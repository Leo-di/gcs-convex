"""Animate the complete two-room cup pickup, transfer, and placement task."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import argparse
from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform

from .build_scene import SceneHandles, build_scene, set_robot_configuration
from .collision_checker import evaluate_collision_status
from .delivery_planner import plan_delivery
from .gcs_base_planner import plan_base_gcs
from .gcs_whole_body_planner import (
    plan_carry_whole_body_gcs,
    plan_to_dict as whole_body_plan_to_dict,
)
from .grasp_planner import plan_pick, cartesian_arm_path, solve_arm_ik
from .iris_region_generator import (
    cover_to_dict as iris_cover_to_dict,
    generate_carrying_iris_cover,
    generate_doorway_iris_region,
    result_to_dict as iris_result_to_dict,
)
from .scene_spec import PROJECT_ROOT, semantic_regions, load_scene_config
from .semantic_predicates import base_inside_region, base_inside_region_residual
from .semantic_product_planner import plan_semantic_product, plan_to_dict
from .task_semantics import near_pickup_residual, cup_bin_residuals, pick_guard_residuals, place_guard_residuals
from .semantic_monitor import monitor_delivery
from .candidate_planner import CandidateRejected, select_candidate
from .grasp_audit import (
    annotated_scene_html,
    measure_cup_gripper,
    measure_wall_clearance,
    summarize_grasp_audit,
    summarize_wall_clearance,
)


@dataclass(frozen=True)
class TaskFrame:
    phase: str
    base_q: np.ndarray
    arm_q: np.ndarray
    gripper_q: np.ndarray
    cup_pose: RigidTransform
    cup_grasped: bool


def _interpolate(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(
        [
            (1.0 - alpha) * start + alpha * end
            for alpha in np.linspace(0.0, 1.0, count)
        ],
        dtype=float,
    )


def _attached_cup_pose(
    scene: SceneHandles,
    base_q: np.ndarray,
    arm_q: np.ndarray,
    X_GC: RigidTransform,
) -> RigidTransform:
    set_robot_configuration(
        scene,
        base_q,
        arm_q,
        scene.gripper_closed_positions,
    )
    X_WG = scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
    return X_WG @ X_GC


def _cup_inside_dropoff_bin(scene: SceneHandles, pose: RigidTransform) -> bool:
    residuals = cup_bin_residuals(pose.GetAsMatrix4(), scene.config)
    return all(value >= -1e-8 for value in residuals.values())


def _append_fixed_cup_frames(
    frames: list[TaskFrame],
    *,
    phase: str,
    base_q: np.ndarray,
    arm_path: np.ndarray,
    gripper_q: np.ndarray,
    cup_pose: RigidTransform,
    cup_grasped: bool,
) -> None:
    for arm_q in arm_path:
        frames.append(
            TaskFrame(
                phase=phase,
                base_q=np.asarray(base_q, dtype=float),
                arm_q=np.asarray(arm_q, dtype=float),
                gripper_q=np.asarray(gripper_q, dtype=float),
                cup_pose=cup_pose,
                cup_grasped=cup_grasped,
            )
        )


def _append_configuration_frames(
    scene: SceneHandles,
    frames: list[TaskFrame],
    *,
    phase: str,
    configurations: np.ndarray,
    gripper_q: np.ndarray,
    cup_pose: RigidTransform | None = None,
    X_GC: RigidTransform | None = None,
) -> None:
    """Append frames from full 10-D configurations.

    When ``X_GC`` is given the cup rides with the gripper; otherwise the
    cup stays at the supplied fixed pose.
    """

    for q in np.asarray(configurations, dtype=float):
        pose = (
            cup_pose
            if cup_pose is not None
            else _attached_cup_pose(scene, q[:3], q[3:], X_GC)
        )
        frames.append(
            TaskFrame(
                phase=phase,
                base_q=q[:3],
                arm_q=q[3:],
                gripper_q=np.asarray(gripper_q, dtype=float),
                cup_pose=pose,
                cup_grasped=X_GC is not None,
            )
        )


def _audit_manipulation_phase_geometry(
    scene: SceneHandles,
    mode: str,
    configurations: np.ndarray,
    *,
    X_GC: RigidTransform | None = None,
    final_cup_pose: RigidTransform | None = None,
) -> dict:
    """Audit a mode trajectory in the unfiltered main Drake scene.

    The mode-specific IRIS checker deliberately filters expected contact
    pairs.  That checker is suitable for growing regions, but it must not be
    the final authority for a trajectory because a convex shortcut can pass
    through a non-contact part of a finger.  This second gate uses the main
    scene and the same named-fingertip rules as the full delivery audit.
    """

    attached = mode in {"grasp", "release"}
    gripper_q = (
        scene.gripper_closed_positions
        if attached
        else scene.gripper_open_positions
    )
    if attached and X_GC is None:
        raise ValueError(f"Manipulation mode {mode} requires X_GC for audit.")
    if mode == "retreat" and final_cup_pose is None:
        raise ValueError("Manipulation mode retreat requires final_cup_pose for audit.")

    rows = []
    for q in np.asarray(configurations, dtype=float):
        set_robot_configuration(scene, q[:3], q[3:], gripper_q)
        if attached:
            cup_pose = _attached_cup_pose(scene, q[:3], q[3:], X_GC)
        elif mode == "retreat":
            cup_pose = final_cup_pose
        else:
            cup_pose = scene.cup_initial_pose
        scene.plant.SetFreeBodyPose(
            scene.plant_context,
            scene.cup_body,
            cup_pose,
        )
        rows.append(
            {
                "attached": attached,
                "cup_gripper_audit": measure_cup_gripper(scene),
            }
        )
    return summarize_grasp_audit(rows)


def _try_manipulation_phase(
    scene: SceneHandles,
    mode: str,
    reference_q10: np.ndarray,
    attempt: dict,
    scene_cache: dict,
    **kwargs,
):
    """Plan one manipulation phase through mode-specific IRIS regions.

    Returns the phase plan, or ``None`` when the feature is disabled or
    the phase fails; the caller then falls back to the reference path and
    the failure is recorded in the attempt report.
    """

    from .manipulation_gcs import phase_plan_to_dict, plan_manipulation_phase

    if not scene.config.get("manipulation_gcs", {}).get("enabled", False):
        return None
    attempt["stage"] = f"{mode}_manipulation_gcs"
    try:
        plan = plan_manipulation_phase(
            mode=mode,
            config=scene.config,
            reference_q10=reference_q10,
            scene_cache=scene_cache,
            **kwargs,
        )
    except (RuntimeError, ValueError) as error:
        attempt[f"{mode}_manipulation_gcs"] = {
            "status": "fallback_to_reference",
            "error": str(error),
        }
        return None
    summary = phase_plan_to_dict(plan)
    geometry_audit = _audit_manipulation_phase_geometry(
        scene,
        mode,
        plan.configurations,
        X_GC=kwargs.get("X_GC"),
        final_cup_pose=kwargs.get("final_cup_pose"),
    )
    summary["main_scene_geometry_audit"] = geometry_audit
    if not geometry_audit["satisfied"]:
        attempt[f"{mode}_manipulation_gcs"] = {
            "status": "fallback_to_reference",
            "error": (
                "Mode GCS trajectory failed the unfiltered main-scene "
                "cup/gripper geometry audit."
            ),
            "rejected_plan": summary,
        }
        return None
    attempt[f"{mode}_manipulation_gcs"] = summary
    return plan


def _with_fixed_base(base_q: np.ndarray, arm_path: np.ndarray) -> np.ndarray:
    base = np.asarray(base_q, dtype=float)
    return np.asarray(
        [np.concatenate([base, np.asarray(arm, dtype=float)]) for arm in arm_path],
        dtype=float,
    )


def _verify_ordered_events(
    trace: list[dict[str, object]],
    sequence: list[str],
) -> tuple[bool, dict[str, int]]:
    """Return whether Boolean trace fields become true in the required order."""

    next_event = 0
    witnesses: dict[str, int] = {}
    for row in trace:
        if next_event >= len(sequence):
            break
        event = sequence[next_event]
        if bool(row[event]):
            witnesses[event] = int(row["index"])
            next_event += 1
    return next_event == len(sequence), witnesses


def plan_delivery_candidate(scene: SceneHandles, pickup_q: np.ndarray, attempt: dict) -> dict:
    """Plan and check one entire delivery without publishing its animation."""
    start_q = np.asarray(scene.config["robot"]["start"], dtype=float)
    dropoff_q = np.asarray(scene.config["delivery_task"]["transfer_path"][-1], dtype=float)
    attempt["stage"] = "pickup_proximity"
    margin = near_pickup_residual(pickup_q, scene.config)
    if margin < 0:
        raise CandidateRejected("Pickup base is too far from the table", {"near_pickup_residual": margin})
    attempt["stage"] = "semantic_product"
    product_plan = plan_semantic_product(scene.config, start_q, dropoff_q)
    action_steps = [
        step for step in product_plan.steps if step.kind == "action"
    ]
    supported_actions = {"pick_cup", "place_cup"}
    unsupported = set(product_plan.actions) - supported_actions
    if unsupported:
        raise RuntimeError(
            "The product planner selected actions without executors: "
            + ", ".join(sorted(unsupported))
        )
    pick_step = next(step for step in action_steps if step.name == "pick_cup")
    place_step = next(step for step in action_steps if step.name == "place_cup")

    attempt["stage"] = "pickup_region"
    region = next(item for item in scene.config["gcs_base_planning"]["regions"]
                  if item["name"] == pick_step.source.region)
    lower = np.array([*region["lower"], region["yaw_bounds"][0]])
    upper = np.array([*region["upper"], region["yaw_bounds"][1]])
    if np.any(pickup_q < lower) or np.any(pickup_q > upper):
        raise CandidateRejected("Pickup configuration is outside the product action region")
    attempt["stage"] = "outbound_gcs"
    from .geometry_outbound_gcs import (
        plan_geometry_outbound,
        plan_geometry_outbound_se2,
        select_navigation_arm,
    )
    # The discrete product plan fixes the task order but not the exact lane.
    # Use every region valid in this mode (not just the product plan's chosen
    # shortest route) so the continuous planner can detour around a relocated
    # obstacle, e.g. a cart blocking room1_central_lane via room1_upper_lane.
    allowed_region_names = [r["name"] for r in scene.config["gcs_base_planning"]["regions"]]
    # Empty-hand navigation uses a tucked arm posture: the home posture tilts
    # the arm forward so iiwa_link_6 reaches across the doorway plane and hits
    # the shared wall when the base detours above a relocated cart.
    navigation_arm = select_navigation_arm(scene, [start_q, pickup_q])
    attempt["navigation_arm_q"] = navigation_arm.tolist()
    try:
        outbound_gcs, attempt["outbound_geometry"] = plan_geometry_outbound(
            scene, start_q, pickup_q, allowed_region_names=allowed_region_names,
            arm_q=navigation_arm,
        )
    except RuntimeError:
        # The fixed-yaw translation slice is disconnected or has no path here
        # (e.g. a relocated cart blocks the lane). Retry with SE(2) so the base
        # may rotate to route around it.
        outbound_gcs, attempt["outbound_geometry"] = plan_geometry_outbound_se2(
            scene, start_q, pickup_q, allowed_region_names=allowed_region_names,
            arm_q=navigation_arm,
        )
    base_to_pick = outbound_gcs.configurations.copy()
    # Preserve the prescribed endpoint exactly. Solver roundoff just outside a
    # closed box can make subsequent automatic GCS intersection tests omit the
    # source edge, despite tolerant PointInSet accepting the same state.
    if not np.allclose(base_to_pick[-1], pickup_q, rtol=0., atol=1e-7):
        raise CandidateRejected("Outbound GCS endpoint differs from requested pickup")
    base_to_pick[-1] = np.asarray(pickup_q, dtype=float)
    attempt["stage"] = "pickup_ik"
    pick_plan = plan_pick(scene, base_to_pick[-1])
    # The ordinary scene query filters cup/gripper pairs. Audit them explicitly.
    set_robot_configuration(scene, pick_plan.base_q, pick_plan.grasp_q, scene.gripper_closed_positions)
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, scene.cup_initial_pose)
    endpoint_audit = summarize_grasp_audit([{"attached": True, "cup_gripper_audit": measure_cup_gripper(scene)}])
    attempt["grasp_endpoint_audit"] = endpoint_audit
    if scene.config.get("validation", {}).get("require_grasp_geometry", False) and not endpoint_audit["satisfied"]:
        attempt["stage"] = "grasp_geometry"
        raise CandidateRejected("Unfiltered grasp geometry failed", endpoint_audit)
    attempt["stage"] = "delivery_ik"
    delivery_plan = plan_delivery(scene, pick_plan)
    attempt["stage"] = "action_guards"
    # Reject infeasible action endpoints before constructing the animation.
    for action, base_q, arm_q, cup_pose, guard in (
        ("pick", pick_plan.base_q, pick_plan.grasp_q, scene.cup_initial_pose, pick_guard_residuals),
        ("place", delivery_plan.base_q, delivery_plan.place_q, delivery_plan.place_cup_pose, place_guard_residuals),
    ):
        set_robot_configuration(scene, base_q, arm_q, scene.gripper_closed_positions)
        gripper_pose = scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
        actual_cup = cup_pose if action == "pick" else gripper_pose @ delivery_plan.X_GC
        observation = {"base_q": base_q, "gripper_q": scene.gripper_closed_positions,
                       "cup_pose": actual_cup.GetAsMatrix4(), "gripper_pose": gripper_pose.GetAsMatrix4()}
        residuals = guard(observation, scene.config)
        failed = {name: value for name, value in residuals.items() if not np.isfinite(value) or value < -1e-8}
        if failed:
            raise CandidateRejected(f"{action} geometric guard failed", {"residuals": failed})
    attempt["stage"] = "doorway_iris"
    carrying_iris_scene = build_scene(
        config=scene.config,
        attached_cup_X_GC=delivery_plan.X_GC,
        visualize=False,
    )
    doorway_iris = generate_doorway_iris_region(
        carrying_iris_scene,
        delivery_plan.carry_q,
    )
    frames: list[TaskFrame] = []

    for base_q in base_to_pick:
        frames.append(
            TaskFrame(
                phase="navigate_to_pickup",
                base_q=base_q,
                arm_q=navigation_arm,
                gripper_q=scene.gripper_open_positions,
                cup_pose=scene.cup_initial_pose,
                cup_grasped=False,
            )
        )

    from .safe_pregrasp import plan_pregrasp
    pregrasp_path = (plan_pregrasp(scene, pick_plan.base_q, navigation_arm, pick_plan.pregrasp_q)
                    if scene.config.get("pick_task", {}).get("collision_checked_pregrasp", False)
                    else _interpolate(navigation_arm, pick_plan.pregrasp_q, 45))
    approach_path = (cartesian_arm_path(scene, pick_plan.base_q,
                     pick_plan.pregrasp_q, pick_plan.grasp_q, 35, stage="approach")
                     if scene.config.get("pick_task", {}).get("cartesian_approach", False)
                     else _interpolate(pick_plan.pregrasp_q, pick_plan.grasp_q, 35))
    manipulation_scene_cache: dict = {}
    pregrasp_reference = np.vstack(
        [
            _with_fixed_base(pick_plan.base_q, pregrasp_path),
            _with_fixed_base(pick_plan.base_q, approach_path)[1:],
        ]
    )
    pregrasp_gcs = _try_manipulation_phase(
        scene, "pregrasp", pregrasp_reference, attempt, manipulation_scene_cache,
    )
    if pregrasp_gcs is not None:
        # One GCS trajectory spans both phases; label the split at the
        # sample closest to the pre-grasp keyframe.
        configurations = pregrasp_gcs.configurations
        split = int(
            np.argmin(
                np.linalg.norm(
                    configurations[:, 3:] - np.asarray(pick_plan.pregrasp_q, float),
                    axis=1,
                )
            )
        )
        _append_configuration_frames(
            scene, frames, phase="arm_to_pregrasp",
            configurations=configurations[: split + 1],
            gripper_q=scene.gripper_open_positions,
            cup_pose=scene.cup_initial_pose,
        )
        _append_configuration_frames(
            scene, frames, phase="approach_cup",
            configurations=configurations[split + 1 :],
            gripper_q=scene.gripper_open_positions,
            cup_pose=scene.cup_initial_pose,
        )
    else:
        _append_fixed_cup_frames(
            frames,
            phase="arm_to_pregrasp",
            base_q=pick_plan.base_q,
            arm_path=pregrasp_path,
            gripper_q=scene.gripper_open_positions,
            cup_pose=scene.cup_initial_pose,
            cup_grasped=False,
        )
        _append_fixed_cup_frames(
            frames,
            phase="approach_cup",
            base_q=pick_plan.base_q,
            arm_path=approach_path,
            gripper_q=scene.gripper_open_positions,
            cup_pose=scene.cup_initial_pose,
            cup_grasped=False,
        )

    for gripper_q in _interpolate(
        scene.gripper_open_positions,
        scene.gripper_closed_positions,
        25,
    ):
        frames.append(
            TaskFrame(
                phase="close_gripper",
                base_q=pick_plan.base_q,
                arm_q=pick_plan.grasp_q,
                gripper_q=gripper_q,
                cup_pose=scene.cup_initial_pose,
                cup_grasped=False,
            )
        )

    lift_reference = _interpolate(pick_plan.grasp_q, pick_plan.lift_q, 45)
    lift_gcs = _try_manipulation_phase(
        scene, "grasp", _with_fixed_base(pick_plan.base_q, lift_reference),
        attempt, manipulation_scene_cache, X_GC=delivery_plan.X_GC,
    )
    lift_configurations = (
        lift_gcs.configurations
        if lift_gcs is not None
        else _with_fixed_base(pick_plan.base_q, lift_reference)
    )
    _append_configuration_frames(
        scene, frames, phase="lift_cup",
        configurations=lift_configurations,
        gripper_q=scene.gripper_closed_positions,
        X_GC=delivery_plan.X_GC,
    )

    carrying_region_names = product_plan.regions_in_mode(place_step.source.mode)
    attempt["stage"] = "bootstrap_carry_gcs"
    bootstrap_whole_body_carry = plan_carry_whole_body_gcs(
        scene.config,
        start_base_q=pick_plan.base_q,
        start_arm_q=pick_plan.lift_q,
        carry_arm_q=delivery_plan.carry_q,
        goal_base_q=delivery_plan.base_q,
        goal_arm_q=delivery_plan.preplace_q,
        allowed_region_names=carrying_region_names,
        region_overrides={"doorway_lane": doorway_iris.region},
    )
    attempt["stage"] = "carrying_iris_cover"
    from .safe_carry_bootstrap import safe_bootstrap
    safe_path, bootstrap_audit = safe_bootstrap(
        carrying_iris_scene, bootstrap_whole_body_carry.configurations, carrying_region_names)
    attempt["bootstrap_audit"] = bootstrap_audit
    if safe_path is None:
        raise CandidateRejected("No collision-checked carrying bootstrap found", bootstrap_audit)
    carrying_iris_cover = generate_carrying_iris_cover(
        carrying_iris_scene,
        safe_path,
        carrying_region_names,
    )
    attempt["stage"] = "whole_body_gcs"
    whole_body_carry = plan_carry_whole_body_gcs(
        scene.config,
        start_base_q=pick_plan.base_q,
        start_arm_q=pick_plan.lift_q,
        carry_arm_q=delivery_plan.carry_q,
        goal_base_q=delivery_plan.base_q,
        goal_arm_q=delivery_plan.preplace_q,
        allowed_region_names=carrying_region_names,
        region_overrides=carrying_iris_cover.regions_by_semantic_label(),
    )
    if scene.config.get("iris_np2_cover", {}).get("counterexample_refinement", True):
        from .refine_carry_regions import refine_carry
        def solve_refined(cover):
            return plan_carry_whole_body_gcs(scene.config,
                start_base_q=pick_plan.base_q, start_arm_q=pick_plan.lift_q,
                carry_arm_q=delivery_plan.carry_q, goal_base_q=delivery_plan.base_q,
                goal_arm_q=delivery_plan.preplace_q, allowed_region_names=carrying_region_names,
                region_overrides=cover.regions_by_semantic_label())
        whole_body_carry, attempt["counterexample_refinement"] = refine_carry(
            carrying_iris_scene, carrying_iris_cover, solve_refined)
    for q in whole_body_carry.configurations[1:]:
        base_q = q[:3]
        arm_q = q[3:]
        frames.append(
            TaskFrame(
                phase="whole_body_carry_to_dropoff",
                base_q=base_q,
                arm_q=arm_q,
                gripper_q=scene.gripper_closed_positions,
                cup_pose=_attached_cup_pose(
                    scene,
                    base_q,
                    arm_q,
                    delivery_plan.X_GC,
                ),
                cup_grasped=True,
            )
        )

    # Lower the cup into the bin: mode-specific IRIS regions first, with the
    # collision-checked Cartesian branch as the reference and fallback.
    # Retain the achieved endpoint for release and retreat.
    from dataclasses import replace
    lowering = cartesian_arm_path(scene, delivery_plan.base_q,
                                  delivery_plan.preplace_q, delivery_plan.place_q, 35,
                                  stage="lower")
    release_gcs = _try_manipulation_phase(
        scene, "release", _with_fixed_base(delivery_plan.base_q, lowering),
        attempt, manipulation_scene_cache, X_GC=delivery_plan.X_GC,
    )
    lowering_configurations = (
        release_gcs.configurations
        if release_gcs is not None
        else _with_fixed_base(delivery_plan.base_q, lowering)
    )
    final_lower_q = lowering_configurations[-1]
    set_robot_configuration(scene, final_lower_q[:3], final_lower_q[3:], scene.gripper_closed_positions)
    achieved = _attached_cup_pose(scene, final_lower_q[:3], final_lower_q[3:], delivery_plan.X_GC)
    delivery_plan = replace(delivery_plan, place_q=final_lower_q[3:], place_cup_pose=achieved)
    _append_configuration_frames(
        scene, frames, phase="lower_into_bin",
        configurations=lowering_configurations,
        gripper_q=scene.gripper_closed_positions,
        X_GC=delivery_plan.X_GC,
    )

    for gripper_q in _interpolate(
        scene.gripper_closed_positions,
        scene.gripper_open_positions,
        25,
    ):
        frames.append(
            TaskFrame(
                phase="release_cup",
                base_q=delivery_plan.base_q,
                arm_q=delivery_plan.place_q,
                gripper_q=gripper_q,
                cup_pose=delivery_plan.place_cup_pose,
                cup_grasped=False,
            )
        )

    # Retract away from the released cup before lifting.  A direct vertical
    # retreat passed the kinematic audit with only millimetres of palm
    # clearance, but tracking error caused real SAP contact during execution.
    # Build a clearance waypoint along the cup-to-gripper separation vector,
    # then lift from that already-separated pose.
    if scene.config.get("validation", {}).get("require_grasp_geometry", False):
        task = scene.config["delivery_task"]
        set_robot_configuration(
            scene,
            delivery_plan.base_q,
            delivery_plan.place_q,
            scene.gripper_open_positions,
        )
        X_WG_place = scene.gripper_body.body_frame().CalcPoseInWorld(
            scene.plant_context
        )
        cup_to_gripper = (
            X_WG_place.translation() - delivery_plan.place_cup_pose.translation()
        )
        separation_norm = float(np.linalg.norm(cup_to_gripper))
        if separation_norm < 1e-6:
            raise CandidateRejected(
                "Cannot construct release retreat direction",
                {"cup_to_gripper_norm_m": separation_norm},
            )
        away_direction = cup_to_gripper / separation_norm
        clearance_translation = (
            X_WG_place.translation()
            + float(task.get("retreat_separation_m", 0.08)) * away_direction
        )
        grasp_point = np.asarray(
            scene.config["gripper"]["grasp_point_in_body"], dtype=float
        )
        clearance_target = (
            RigidTransform(X_WG_place.rotation(), clearance_translation)
            .multiply(grasp_point)
        )
        clearance_q = solve_arm_ik(
            scene,
            stage="retreat_separate",
            base_q=delivery_plan.base_q,
            target_position=clearance_target,
            initial_arm_q=delivery_plan.place_q,
            position_tolerance=0.0002,
            minimum_clearance=float(task["minimum_clearance"]),
            target_orientation=X_WG_place.rotation(),
            orientation_tolerance=0.005,
        )
        set_robot_configuration(
            scene,
            delivery_plan.base_q,
            clearance_q,
            scene.gripper_open_positions,
        )
        X_WG_clearance = scene.gripper_body.body_frame().CalcPoseInWorld(
            scene.plant_context
        )
        lift_target = X_WG_clearance.multiply(grasp_point) + np.array(
            [0.0, 0.0, float(task.get("retreat_lift_m", 0.18))]
        )
        retreat_q = solve_arm_ik(
            scene,
            stage="retreat_lift",
            base_q=delivery_plan.base_q,
            target_position=lift_target,
            initial_arm_q=clearance_q,
            position_tolerance=0.0002,
            minimum_clearance=float(task["minimum_clearance"]),
            target_orientation=X_WG_clearance.rotation(),
            orientation_tolerance=0.005,
        )
        separate_path = cartesian_arm_path(
            scene,
            delivery_plan.base_q,
            delivery_plan.place_q,
            clearance_q,
            25,
            stage="retreat_separate",
        )
        lift_path = cartesian_arm_path(
            scene,
            delivery_plan.base_q,
            clearance_q,
            retreat_q,
            35,
            stage="retreat_lift",
        )
        retreat_path = np.vstack([separate_path, lift_path[1:]])
    else:
        retreat_path = _interpolate(
            delivery_plan.place_q, delivery_plan.preplace_q, 35
        )
    retreat_gcs = _try_manipulation_phase(
        scene, "retreat", _with_fixed_base(delivery_plan.base_q, retreat_path),
        attempt, manipulation_scene_cache, final_cup_pose=delivery_plan.place_cup_pose,
    )
    retreat_configurations = (
        retreat_gcs.configurations
        if retreat_gcs is not None
        else _with_fixed_base(delivery_plan.base_q, retreat_path)
    )
    _append_configuration_frames(
        scene, frames, phase="retreat_arm",
        configurations=retreat_configurations,
        gripper_q=scene.gripper_open_positions,
        cup_pose=delivery_plan.place_cup_pose,
    )
    frames.extend([frames[-1]] * 30)

    attempt["stage"] = "sampled_validation"
    regions = semantic_regions(scene.config)
    minimum_distance = float("inf")
    collision_free = True
    trace: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        set_robot_configuration(
            scene,
            frame.base_q,
            frame.arm_q,
            frame.gripper_q,
        )
        scene.plant.SetFreeBodyPose(
            scene.plant_context,
            scene.cup_body,
            frame.cup_pose,
        )
        status = evaluate_collision_status(
            scene,
            frame.base_q,
            arm_q=frame.arm_q,
            gripper_q=frame.gripper_q,
        )
        minimum_distance = min(minimum_distance, status.minimum_signed_distance)
        collision_free = collision_free and status.collision_free
        trace.append(
            {
                "index": index,
                "phase": frame.phase,
                "base_q": frame.base_q.tolist(),
                "arm_q": frame.arm_q.tolist(),
                "base_in_room1": base_inside_region(frame.base_q, regions["room1"]),
                "base_in_room2": base_inside_region(frame.base_q, regions["room2"]),
                "near_pickup_table": near_pickup_residual(frame.base_q, scene.config) >= 0,
                "near_pickup_residual": near_pickup_residual(frame.base_q, scene.config),
                "gripper_q": frame.gripper_q.tolist(),
                "cup_pose": frame.cup_pose.GetAsMatrix4().tolist(),
                "gripper_pose": scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context).GetAsMatrix4().tolist(),
                "attached": bool(frame.cup_grasped),
                "cup_gripper_audit": measure_cup_gripper(scene),
                "wall_clearance": measure_wall_clearance(scene),
                "room1_residual": base_inside_region_residual(frame.base_q, regions["room1"]),
                "room2_residual": base_inside_region_residual(frame.base_q, regions["room2"]),
                "cup_grasped": frame.cup_grasped,
                "cup_in_dropoff_bin": _cup_inside_dropoff_bin(scene, frame.cup_pose),
                "collision_residual": status.residual,
                "closest_geometry_a": status.closest_geometry_a,
                "closest_geometry_b": status.closest_geometry_b,
            }
        )

    semantic_sequence = list(scene.config["scenario"]["semantic_sequence"])
    semantic_sequence_satisfied, semantic_witness_frames = _verify_ordered_events(
        trace,
        semantic_sequence,
    )
    monitor_report = monitor_delivery(trace, scene.config)
    grasp_audit = summarize_grasp_audit(trace)
    required_wall_clearance = float(
        scene.config.get("validation", {}).get("minimum_wall_clearance_m", 0.0)
    )
    wall_clearance_audit = summarize_wall_clearance(trace, required_wall_clearance)
    if scene.config.get("validation", {}).get("require_grasp_geometry", False) and not grasp_audit["satisfied"]:
        violation = grasp_audit.get("first_violation")
        if violation is not None:
            grasp_audit["first_violation_frame"] = trace[violation["index"]]
        raise CandidateRejected("Unfiltered grasp geometry failed along the path", grasp_audit)
    if (
        not collision_free
        or not wall_clearance_audit["satisfied"]
        or not monitor_report["satisfied"]
        or not semantic_sequence_satisfied
    ):
        raise CandidateRejected("Full delivery validation failed", {
            "collision_free": collision_free,
            "wall_clearance_audit": wall_clearance_audit,
            "first_collision": next((r for r in trace if r["collision_residual"] < 0), None),
            "first_violation": monitor_report["first_violation"],
            "semantic_sequence_satisfied": semantic_sequence_satisfied,
        })
    attempt["validation"] = {
        "frames": len(frames), "collision_free": collision_free,
        "semantic_monitor_satisfied": monitor_report["satisfied"],
        "minimum_signed_distance": minimum_distance,
        "grasp_geometry_satisfied": grasp_audit["satisfied"],
        "wall_clearance_satisfied": wall_clearance_audit["satisfied"],
        "physical_grasp_validated": False,
    }
    return {
        "frames": frames, "trace": trace, "monitor_report": monitor_report,
        "grasp_audit": grasp_audit,
        "wall_clearance_audit": wall_clearance_audit,
        "product_plan": product_plan, "outbound_gcs": outbound_gcs,
        "whole_body_carry": whole_body_carry, "doorway_iris": doorway_iris,
        "carrying_iris_cover": carrying_iris_cover, "minimum_distance": minimum_distance,
        "collision_free": collision_free, "semantic_sequence_satisfied": semantic_sequence_satisfied,
        "semantic_sequence": semantic_sequence, "semantic_witness_frames": semantic_witness_frames,
        "manipulation_gcs": {
            mode: attempt.get(f"{mode}_manipulation_gcs")
            for mode in ("pregrasp", "grasp", "release", "retreat")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-wait", action="store_true", help="Exit after validation")
    parser.add_argument("--config", type=Path, help="Scene YAML, including pickup candidates")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "results")
    args = parser.parse_args()
    config = load_scene_config(args.config) if args.config else load_scene_config()
    scene = build_scene(config=config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scene_config_snapshot.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    selected, candidate_report = select_candidate(
        scene, plan_delivery_candidate, report_path=args.output_dir / "candidate_attempts.json")
    frames, trace = selected["frames"], selected["trace"]
    monitor_report, product_plan = selected["monitor_report"], selected["product_plan"]
    outbound_gcs, whole_body_carry = selected["outbound_gcs"], selected["whole_body_carry"]
    doorway_iris, carrying_iris_cover = selected["doorway_iris"], selected["carrying_iris_cover"]
    minimum_distance = selected["minimum_distance"]
    semantic_sequence, semantic_witness_frames = selected["semantic_sequence"], selected["semantic_witness_frames"]
    collision_free = selected["collision_free"]
    semantic_sequence_satisfied = selected["semantic_sequence_satisfied"]
    final_frame = frames[-1]
    phase_counts = Counter(frame.phase for frame in frames)

    # Only publish the fully validated candidate; rejected plans remain diagnostics.
    scene.meshcat.StartRecording()
    for index, frame in enumerate(frames):
        set_robot_configuration(scene, frame.base_q, frame.arm_q, frame.gripper_q)
        scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, frame.cup_pose)
        scene.root_context.SetTime(index * 0.05)
        scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()
    (args.output_dir / "review_scene.html").write_text(
        annotated_scene_html(
            scene.meshcat.StaticHtml(),
            selected["grasp_audit"],
            selected["wall_clearance_audit"],
        ),
        encoding="utf-8",
    )

    result = {
        "selected_candidate": candidate_report["selected_candidate"],
        "validation_scope": "sampled_kinematic_scene_with_collision_filters",
        "grasp_geometry_satisfied": selected["grasp_audit"]["satisfied"],
        "physical_grasp_validated": False,
        "wall_clearance_audit": selected["wall_clearance_audit"],
        "candidate_attempts": candidate_report,
        "scenario": scene.config["scenario"]["name"],
        "instruction": scene.config["scenario"]["instruction"],
        "frames": len(frames),
        "phase_counts": dict(phase_counts),
        "collision_free": collision_free,
        "minimum_signed_distance": minimum_distance,
        "final_base_q": final_frame.base_q.tolist(),
        "final_cup_position": final_frame.cup_pose.translation().tolist(),
        "cup_in_dropoff_bin": _cup_inside_dropoff_bin(scene, final_frame.cup_pose),
        "semantic_sequence": semantic_sequence,
        "semantic_sequence_satisfied": semantic_sequence_satisfied,
        "semantic_monitor_satisfied": monitor_report["satisfied"],
        "semantic_witness_frames": semantic_witness_frames,
        "semantic_product_graph": plan_to_dict(product_plan),
        "iris_np2_doorway": iris_result_to_dict(doorway_iris),
        "iris_np2_carrying_cover": iris_cover_to_dict(carrying_iris_cover),
        "manipulation_gcs": selected["manipulation_gcs"],
        "gcs": {
            "outbound": {
                "solve_time_seconds": outbound_gcs.solve_time_seconds,
                "trajectory_duration": outbound_gcs.trajectory_duration,
                "path_length": outbound_gcs.path_length,
                "angular_travel": outbound_gcs.angular_travel,
                "optimal_cost": outbound_gcs.optimal_cost,
                "solver": outbound_gcs.solver_name,
                "number_of_segments": outbound_gcs.number_of_segments,
                "region_sequence": outbound_gcs.region_sequence,
            },
            "whole_body_carry": whole_body_plan_to_dict(
                whole_body_carry,
                include_configurations=False,
            ),
        },
    }
    results_path = args.output_dir / "delivery_result.json"
    trace_path = args.output_dir / "delivery_trace.json"
    product_path = args.output_dir / "semantic_product_plan.json"
    whole_body_path = args.output_dir / "whole_body_plan.json"
    iris_path = args.output_dir / "iris_np2_doorway_region.json"
    cover_path = args.output_dir / "iris_np2_carrying_cover.json"
    monitor_path = args.output_dir / "semantic_monitor.json"
    monitor_path.write_text(json.dumps(monitor_report, indent=2), encoding="utf-8")
    (args.output_dir / "grasp_geometry_audit.json").write_text(json.dumps(selected["grasp_audit"], indent=2), encoding="utf-8")
    (args.output_dir / "wall_clearance_audit.json").write_text(
        json.dumps(selected["wall_clearance_audit"], indent=2), encoding="utf-8"
    )
    (results_path.parent / "scene_config_snapshot.json").write_text(json.dumps(scene.config, indent=2), encoding="utf-8")
    results_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    trace_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
    product_path.write_text(
        json.dumps(plan_to_dict(product_plan), indent=2),
        encoding="utf-8",
    )
    whole_body_path.write_text(
        json.dumps(whole_body_plan_to_dict(whole_body_carry), indent=2),
        encoding="utf-8",
    )
    iris_path.write_text(
        json.dumps(iris_result_to_dict(doorway_iris), indent=2),
        encoding="utf-8",
    )
    cover_path.write_text(
        json.dumps(iris_cover_to_dict(carrying_iris_cover), indent=2),
        encoding="utf-8",
    )

    print("Selected pickup candidate:", candidate_report["selected_candidate"])
    print(f"Meshcat URL: {scene.meshcat.web_url()}")
    print(f"Saved delivery result: {results_path}")
    print(f"Saved semantic trace: {trace_path}")
    print(f"Saved semantic product plan: {product_path}")
    print(f"Saved whole-body GCS plan: {whole_body_path}")
    print(f"Saved IRIS-NP2 doorway region: {iris_path}")
    print(f"Saved IRIS-NP2 carrying cover: {cover_path}")
    print("Collision free:", collision_free)
    print("Minimum signed distance:", minimum_distance)
    print(
        "Payload/robot wall clearance:",
        selected["wall_clearance_audit"]["minimum_payload_wall_distance_m"],
        selected["wall_clearance_audit"]["minimum_robot_wall_distance_m"],
    )
    print("Cup in dropoff bin:", result["cup_in_dropoff_bin"])
    print("Semantic sequence satisfied:", semantic_sequence_satisfied)
    print(
        "Semantic product actions:",
        " -> ".join(product_plan.actions),
    )
    print("Outbound GCS regions:", " -> ".join(outbound_gcs.region_sequence))
    print(
        "Whole-body GCS regions:",
        " -> ".join(whole_body_carry.region_sequence),
    )
    print(
        "IRIS-NP2 doorway audit:",
        f"{doorway_iris.holdout_collisions}/{doorway_iris.holdout_samples} collisions",
    )
    cover_audit = iris_cover_to_dict(carrying_iris_cover)["independent_holdout_audit"]
    print(
        "IRIS-NP2 cover audit:",
        f"{cover_audit['collisions']}/{cover_audit['samples']} collisions",
    )
    for mode in ("pregrasp", "grasp", "release", "retreat"):
        summary = selected["manipulation_gcs"][mode]
        if summary is None:
            print(f"Manipulation GCS [{mode}]: disabled")
        elif "error" in summary:
            print(f"Manipulation GCS [{mode}]: fallback ({summary['error'][:60]})")
        else:
            print(
                f"Manipulation GCS [{mode}]: {summary['number_of_regions']} regions, "
                f"audit {summary['trajectory_audit']['collisions']}/"
                f"{summary['trajectory_audit']['samples']} collisions, "
                f"travel {summary['arm_joint_travel']:.3f} rad",
            )
    print("Final cup position:", final_frame.cup_pose.translation())
    print("Semantic monitor satisfied:", monitor_report["satisfied"])
    print("Unfiltered grasp geometry satisfied:", selected["grasp_audit"]["satisfied"])
    print("Physical grasp validated: False (kinematic replay only)")
    if not monitor_report["satisfied"]:
        raise RuntimeError(f"Semantic validation failed: {monitor_report['first_violation']}; see {monitor_path}")
    if not args.no_wait:
        input("Press Enter to close the delivery animation... ")


if __name__ == "__main__":
    main()
