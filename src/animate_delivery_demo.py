"""Animate the complete two-room cup pickup, transfer, and placement task."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

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
from .grasp_planner import plan_pick
from .iris_region_generator import (
    cover_to_dict as iris_cover_to_dict,
    generate_carrying_iris_cover,
    generate_doorway_iris_region,
    result_to_dict as iris_result_to_dict,
)
from .scene_spec import PROJECT_ROOT, semantic_regions
from .semantic_predicates import base_inside_region, base_inside_region_residual
from .semantic_product_planner import plan_semantic_product, plan_to_dict


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
    center = pose.translation()
    bin_spec = scene.config["objects"]["dropoff_bin"]
    bin_center = np.asarray(bin_spec["center"], dtype=float)
    bin_size = np.asarray(bin_spec["size"], dtype=float)
    wall = float(bin_spec["wall_thickness"])
    cup_spec = scene.config["objects"]["cup"]
    radius = float(cup_spec["radius"])
    height = float(cup_spec["height"])

    inner_half_xy = bin_size[:2] / 2.0 - wall - radius
    inside_xy = np.all(np.abs(center[:2] - bin_center[:2]) <= inner_half_xy)
    bin_bottom = bin_center[2] - bin_size[2] / 2.0 + wall
    supported = abs((center[2] - height / 2.0) - bin_bottom) <= 0.008
    return bool(inside_xy and supported)


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


def main() -> None:
    scene = build_scene()
    start_q = np.asarray(scene.config["robot"]["start"], dtype=float)
    pickup_q = np.asarray(scene.config["reference_path"][-1], dtype=float)
    dropoff_q = np.asarray(
        scene.config["delivery_task"]["transfer_path"][-1],
        dtype=float,
    )
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

    outbound_gcs = plan_base_gcs(
        scene.config,
        start_q,
        pickup_q,
        allowed_region_names=product_plan.regions_in_mode(pick_step.source.mode),
    )
    base_to_pick = outbound_gcs.configurations
    pick_plan = plan_pick(scene, base_to_pick[-1])
    delivery_plan = plan_delivery(scene, pick_plan)
    carrying_iris_scene = build_scene(
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
                arm_q=scene.arm_home_positions,
                gripper_q=scene.gripper_open_positions,
                cup_pose=scene.cup_initial_pose,
                cup_grasped=False,
            )
        )

    _append_fixed_cup_frames(
        frames,
        phase="arm_to_pregrasp",
        base_q=pick_plan.base_q,
        arm_path=_interpolate(scene.arm_home_positions, pick_plan.pregrasp_q, 45),
        gripper_q=scene.gripper_open_positions,
        cup_pose=scene.cup_initial_pose,
        cup_grasped=False,
    )
    _append_fixed_cup_frames(
        frames,
        phase="approach_cup",
        base_q=pick_plan.base_q,
        arm_path=_interpolate(pick_plan.pregrasp_q, pick_plan.grasp_q, 35),
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

    for arm_q in _interpolate(pick_plan.grasp_q, pick_plan.lift_q, 45):
        frames.append(
            TaskFrame(
                phase="lift_cup",
                base_q=pick_plan.base_q,
                arm_q=arm_q,
                gripper_q=scene.gripper_closed_positions,
                cup_pose=_attached_cup_pose(
                    scene,
                    pick_plan.base_q,
                    arm_q,
                    delivery_plan.X_GC,
                ),
                cup_grasped=True,
            )
        )

    carrying_region_names = product_plan.regions_in_mode(place_step.source.mode)
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
    carrying_iris_cover = generate_carrying_iris_cover(
        carrying_iris_scene,
        bootstrap_whole_body_carry.configurations,
        carrying_region_names,
    )
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

    for arm_q in _interpolate(delivery_plan.preplace_q, delivery_plan.place_q, 35):
        frames.append(
            TaskFrame(
                phase="lower_into_bin",
                base_q=delivery_plan.base_q,
                arm_q=arm_q,
                gripper_q=scene.gripper_closed_positions,
                cup_pose=_attached_cup_pose(
                    scene,
                    delivery_plan.base_q,
                    arm_q,
                    delivery_plan.X_GC,
                ),
                cup_grasped=True,
            )
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

    _append_fixed_cup_frames(
        frames,
        phase="retreat_arm",
        base_q=delivery_plan.base_q,
        arm_path=_interpolate(delivery_plan.place_q, delivery_plan.preplace_q, 35),
        gripper_q=scene.gripper_open_positions,
        cup_pose=delivery_plan.place_cup_pose,
        cup_grasped=False,
    )
    frames.extend([frames[-1]] * 30)

    regions = semantic_regions(scene.config)
    minimum_distance = float("inf")
    collision_free = True
    trace: list[dict[str, object]] = []
    pickup_xy = np.asarray(
        scene.config["objects"]["table"]["center"][:2],
        dtype=float,
    )
    near_pickup_distance = float(
        scene.config["scenario"]["near_pickup_distance"],
    )
    scene.meshcat.StartRecording()
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
                "near_pickup_table": bool(
                    np.linalg.norm(frame.base_q[:2] - pickup_xy)
                    <= near_pickup_distance
                ),
                "room1_residual": base_inside_region_residual(frame.base_q, regions["room1"]),
                "room2_residual": base_inside_region_residual(frame.base_q, regions["room2"]),
                "cup_grasped": frame.cup_grasped,
                "cup_in_dropoff_bin": _cup_inside_dropoff_bin(scene, frame.cup_pose),
                "collision_residual": status.residual,
                "closest_geometry_a": status.closest_geometry_a,
                "closest_geometry_b": status.closest_geometry_b,
            }
        )
        scene.root_context.SetTime(index * 0.05)
        scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()

    final_frame = frames[-1]
    phase_counts = Counter(frame.phase for frame in frames)
    semantic_sequence = list(scene.config["scenario"]["semantic_sequence"])
    semantic_sequence_satisfied, semantic_witness_frames = _verify_ordered_events(
        trace,
        semantic_sequence,
    )
    result = {
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
        "semantic_witness_frames": semantic_witness_frames,
        "semantic_product_graph": plan_to_dict(product_plan),
        "iris_np2_doorway": iris_result_to_dict(doorway_iris),
        "iris_np2_carrying_cover": iris_cover_to_dict(carrying_iris_cover),
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
    results_path = PROJECT_ROOT / "outputs" / "results" / "delivery_result.json"
    trace_path = PROJECT_ROOT / "outputs" / "results" / "delivery_trace.json"
    product_path = PROJECT_ROOT / "outputs" / "results" / "semantic_product_plan.json"
    whole_body_path = PROJECT_ROOT / "outputs" / "results" / "whole_body_plan.json"
    iris_path = PROJECT_ROOT / "outputs" / "results" / "iris_np2_doorway_region.json"
    cover_path = PROJECT_ROOT / "outputs" / "results" / "iris_np2_carrying_cover.json"
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

    print(f"Meshcat URL: {scene.meshcat.web_url()}")
    print(f"Saved delivery result: {results_path}")
    print(f"Saved semantic trace: {trace_path}")
    print(f"Saved semantic product plan: {product_path}")
    print(f"Saved whole-body GCS plan: {whole_body_path}")
    print(f"Saved IRIS-NP2 doorway region: {iris_path}")
    print(f"Saved IRIS-NP2 carrying cover: {cover_path}")
    print("Collision free:", collision_free)
    print("Minimum signed distance:", minimum_distance)
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
    print("Final cup position:", final_frame.cup_pose.translation())
    input("Press Enter to close the delivery animation... ")


if __name__ == "__main__":
    main()
