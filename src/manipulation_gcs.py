"""Mode-specific IRIS-NP2 regions and 10-D GCS for the manipulation phases.

The cup-delivery task interleaves navigation with four manipulation modes,
each with a different frozen collision model in the 10-D reduced state
``q=(x, y, yaw, q_arm[0:7])``:

  pregrasp: the free cup rests on the pickup table as an obstacle, gripper
            open (fingertip contact is filtered).  Covers the arm motion
            from the tucked navigation posture through pre-grasp down to
            the grasp configuration.
  grasp:    the cup is attached to the gripper via ``X_GC``, gripper
            closed.  Covers the lift off the pickup table.
  release:  the cup is attached via ``X_GC``, gripper closed, at the
            drop-off site.  Covers the lowering into the bin.
  retreat:  the free cup rests in the bin at its final pose as an
            obstacle, gripper open.  Covers the arm retraction.

Each phase grows overlapping IRIS-NP2 regions seeded along its reference
path (the same grow-and-audit procedure as the carrying cover), then
optimizes the arm trajectory through those regions with the official
Drake GCS trajectory optimization.  The two gripper joints are outside
the 10-D planning state, so the open/close transitions remain fixed-arm
replays; this module plans the seven arm joints and the base within each
mode.  Region guarantees are probabilistic (epsilon/delta per region)
and the trajectory gate is a finite-sample numerical audit, not a
continuous-time certificate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Callable

import numpy as np

from pydrake.geometry.optimization import Point
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.multibody.tree import ModelInstanceIndex

from .build_scene import SceneHandles, build_scene, set_robot_configuration
from .iris_region_generator import (
    IrisRegionResult,
    _grow_cover_region,
    _make_collision_checker,
    _make_parameterization,
    _reduced_position_indices,
)


@dataclass(frozen=True)
class ManipulationModeSpec:
    """Frozen collision-model definition for one manipulation mode."""

    mode: str
    attached: bool
    gripper_closed: bool
    cup_at_final_pose: bool
    site: str  # "pickup" | "dropoff"


MODE_SPECS: dict[str, ManipulationModeSpec] = {
    "pregrasp": ManipulationModeSpec(
        mode="pregrasp",
        attached=False,
        gripper_closed=False,
        cup_at_final_pose=False,
        site="pickup",
    ),
    "grasp": ManipulationModeSpec(
        mode="grasp",
        attached=True,
        gripper_closed=True,
        cup_at_final_pose=False,
        site="pickup",
    ),
    "release": ManipulationModeSpec(
        mode="release",
        attached=True,
        gripper_closed=True,
        cup_at_final_pose=False,
        site="dropoff",
    ),
    "retreat": ManipulationModeSpec(
        mode="retreat",
        attached=False,
        gripper_closed=False,
        cup_at_final_pose=True,
        site="dropoff",
    ),
}

SITE_REGION_NAMES: dict[str, str] = {
    "pickup": "room2_pickup_lane",
    "dropoff": "room1_upper_lane",
}


@dataclass(frozen=True)
class ManipulationPhasePlan:
    mode: str
    configurations: np.ndarray  # (N, 10)
    region_sequence: list[str]
    seed_count: int
    bridged: bool
    iris_solve_seconds: float
    gcs_solve_seconds: float
    trajectory_duration: float
    arm_joint_travel: float
    audit_samples: int
    audit_collisions: int
    region_results: tuple[IrisRegionResult, ...] = field(repr=False, default=())


def phase_plan_to_dict(plan: ManipulationPhasePlan) -> dict[str, Any]:
    total_holdout = sum(item.holdout_samples for item in plan.region_results)
    total_collisions = sum(item.holdout_collisions for item in plan.region_results)
    return {
        "mode": plan.mode,
        "algorithm": "Drake IrisNp2 regions + GcsTrajectoryOptimization",
        "dimension": 10,
        "number_of_regions": len(plan.region_results),
        "seed_count": plan.seed_count,
        "bridged": plan.bridged,
        "region_sequence": plan.region_sequence,
        "frames": int(plan.configurations.shape[0]),
        "iris_solve_seconds": plan.iris_solve_seconds,
        "gcs_solve_seconds": plan.gcs_solve_seconds,
        "trajectory_duration": plan.trajectory_duration,
        "arm_joint_travel": plan.arm_joint_travel,
        "trajectory_audit": {
            "samples": plan.audit_samples,
            "collisions": plan.audit_collisions,
        },
        "region_holdout_audit": {
            "samples": total_holdout,
            "collisions": total_collisions,
        },
        "regions": [
            {
                "index": index,
                "faces": item.number_of_faces,
                "solve_time_seconds": item.solve_time_seconds,
                "holdout_collisions": item.holdout_collisions,
                "holdout_samples": item.holdout_samples,
            }
            for index, item in enumerate(plan.region_results)
        ],
        "scope": (
            "Probabilistic per-region epsilon/delta guarantees; finite-sample "
            "trajectory audit; not a continuous-time safety certificate."
        ),
    }


def _manipulation_spec(config: dict[str, Any]) -> dict[str, Any]:
    section = config.get("manipulation_gcs", {})
    cover = config["iris_np2_cover"]
    return {
        "ray_particles": cover["ray_particles"],
        "particles": cover["particles"],
        "max_iterations": cover["max_iterations"],
        "max_separating_iterations": cover["max_separating_iterations"],
        "max_planes_per_iteration": cover["max_planes_per_iteration"],
        "verbose": cover["verbose"],
        "epsilon": cover["epsilon"],
        "configuration_space_margin": cover["configuration_space_margin"],
        "mixing_steps": cover["mixing_steps"],
        "random_seed": cover["random_seed"],
        "starting_radius": cover["starting_radius"],
        "holdout_samples_per_region": cover["holdout_samples_per_region"],
        "holdout_mixing_steps": cover["holdout_mixing_steps"],
        "holdout_random_seed": cover["holdout_random_seed"],
        "joint_radius": float(section.get("joint_radius", 0.30)),
        "seed_distance_threshold": float(
            section.get("seed_distance_threshold", 0.35)
        ),
        "total_delta": float(section.get("total_delta", 0.05)),
        "max_bridges": int(section.get("max_bridges", 8)),
    }


def select_phase_seeds(
    reference: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Greedy weighted-distance seed selection including both endpoints."""

    configurations = np.asarray(reference, dtype=float)
    if configurations.ndim != 2 or configurations.shape[0] < 2:
        raise ValueError("Reference path must contain at least two samples.")
    if np.allclose(configurations, configurations[0]):
        raise ValueError("Reference path is constant; nothing to plan.")
    indices = [0]
    last = configurations[0]
    for index in range(1, configurations.shape[0] - 1):
        weighted = float(np.linalg.norm((configurations[index] - last) * weights))
        if weighted >= threshold:
            indices.append(index)
            last = configurations[index]
    indices.append(configurations.shape[0] - 1)
    return configurations[indices].copy()


def _phase_domain(
    scene: SceneHandles,
    seed: np.ndarray,
    site_region_spec: dict[str, Any],
    spec: dict[str, Any],
) -> Any:
    from pydrake.geometry.optimization import HPolyhedron

    lower = np.concatenate(
        [
            np.asarray(site_region_spec["lower"], dtype=float),
            np.asarray([site_region_spec["yaw_bounds"][0]], dtype=float),
            seed[3:] - float(spec["joint_radius"]),
        ]
    )
    upper = np.concatenate(
        [
            np.asarray(site_region_spec["upper"], dtype=float),
            np.asarray([site_region_spec["yaw_bounds"][1]], dtype=float),
            seed[3:] + float(spec["joint_radius"]),
        ]
    )
    joint_spec = scene.config["whole_body_planning"]
    lower[3:] = np.maximum(lower[3:], joint_spec["joint_lower_limits"])
    upper[3:] = np.minimum(upper[3:], joint_spec["joint_upper_limits"])
    return HPolyhedron.MakeBox(lower, upper)


def _site_region_spec(config: dict[str, Any], site: str) -> dict[str, Any]:
    name = SITE_REGION_NAMES[site]
    for spec in config["gcs_base_planning"]["regions"]:
        if spec["name"] == name:
            return spec
    raise ValueError(f"Unknown manipulation site region: {name}")


@dataclass(frozen=True)
class _ModeMachinery:
    scene: SceneHandles
    parameterization: Callable[[np.ndarray], np.ndarray]
    checker: Any
    reduced_indices: np.ndarray


def _apply_mode_filters(mode_scene: SceneHandles, spec: ManipulationModeSpec) -> None:
    """Add the homogeneous collision filters the mode checker requires.

    Drake's planning collision checker needs uniform per-body-pair filters,
    which the fingertip-only cup filter from ``build_scene`` is not.  The
    manipulator-mode scenes therefore exclude the full gripper body against
    the free cup, and the expected payload-support contact (pickup table at
    grasp, bin at release) against the attached cup.  These exclusions only
    loosen the *region* checker; the per-frame validation replay still runs
    against the unfiltered scene geometry.
    """

    from pydrake.geometry import CollisionFilterDeclaration, GeometrySet

    plant = mode_scene.plant
    scene_graph = mode_scene.scene_graph
    if spec.attached:
        attached_gid = mode_scene.attached_cup_geometry_id
        support_bodies = [plant.GetBodyByName("table"), plant.GetBodyByName("dropoff_bin")]
        declaration = CollisionFilterDeclaration()
        for body in support_bodies:
            declaration.ExcludeBetween(
                GeometrySet([attached_gid]),
                GeometrySet(plant.GetCollisionGeometriesForBody(body)),
            )
    else:
        gripper_geometries = [
            geometry_id
            for body_index in plant.GetBodyIndices(mode_scene.gripper_instance)
            for geometry_id in plant.GetCollisionGeometriesForBody(
                plant.get_body(body_index)
            )
        ]
        declaration = CollisionFilterDeclaration().ExcludeBetween(
            GeometrySet(plant.GetCollisionGeometriesForBody(mode_scene.cup_body)),
            GeometrySet(gripper_geometries),
        )
    scene_graph.collision_filter_manager().Apply(declaration)


def _mode_scene(
    config: dict[str, Any],
    spec: ManipulationModeSpec,
    X_GC: Any,
    scene_cache: dict[str, SceneHandles] | None,
) -> SceneHandles:
    key = "attached" if spec.attached else "free"
    if scene_cache is not None and key in scene_cache:
        return scene_cache[key]
    if spec.attached:
        scene = build_scene(
            config=config,
            attached_cup_X_GC=X_GC,
            visualize=False,
        )
    else:
        scene = build_scene(config=config, visualize=False)
    _apply_mode_filters(scene, spec)
    if scene_cache is not None:
        scene_cache[key] = scene
    return scene


def prepare_mode_machinery(
    *,
    config: dict[str, Any],
    mode: str,
    reference_q10: np.ndarray,
    X_GC: Any = None,
    final_cup_pose: Any = None,
    scene_cache: dict[str, SceneHandles] | None = None,
) -> _ModeMachinery:
    """Freeze the mode's collision model and build its query machinery.

    The 10 reduced coordinates (base + arm) stay free; everything else in
    the full plant state (gripper opening, free-cup pose) is frozen at the
    values prescribed by the mode spec.  Each mode uses a dedicated scene
    with the homogeneous filters it needs, so shared scenes and their
    audits are never weakened.  ``scene_cache`` lets callers reuse one
    free-body scene (pregrasp/retreat) and one attached scene
    (grasp/release) across phases.
    """

    spec = MODE_SPECS[mode]
    mode_scene = _mode_scene(config, spec, X_GC, scene_cache)
    gripper = (
        mode_scene.gripper_closed_positions
        if spec.gripper_closed
        else mode_scene.gripper_open_positions
    )
    reference = np.asarray(reference_q10, dtype=float)
    set_robot_configuration(mode_scene, reference[0, :3], reference[0, 3:], gripper)
    if not spec.attached:
        from pydrake.math import RigidTransform

        if spec.cup_at_final_pose:
            if final_cup_pose is None:
                raise ValueError("Mode retreat requires final_cup_pose.")
            pose = (
                final_cup_pose
                if isinstance(final_cup_pose, RigidTransform)
                else RigidTransform(np.asarray(final_cup_pose, dtype=float))
            )
        else:
            pose = mode_scene.cup_initial_pose
        mode_scene.plant.SetFreeBodyPose(
            mode_scene.plant_context,
            mode_scene.cup_body,
            pose,
        )
    fixed_full_q = mode_scene.plant.GetPositions(mode_scene.plant_context).copy()
    reduced_indices = _reduced_position_indices(mode_scene)
    parameterization = _make_parameterization(fixed_full_q, reduced_indices)
    checker = _make_collision_checker(mode_scene)
    if not checker.CheckConfigCollisionFree(parameterization(reference[0])):
        raise RuntimeError(
            f"Manipulation mode {mode} reference start is in collision."
        )
    if not checker.CheckConfigCollisionFree(parameterization(reference[-1])):
        raise RuntimeError(
            f"Manipulation mode {mode} reference goal is in collision."
        )
    return _ModeMachinery(
        scene=mode_scene,
        parameterization=parameterization,
        checker=checker,
        reduced_indices=reduced_indices,
    )


def _region_labels(
    configurations: np.ndarray,
    regions: list[Any],
    mode: str,
) -> list[str]:
    labels: list[str] = []
    current: str | None = None
    for q in configurations:
        containing = [
            f"{mode}_{index}"
            for index, region in enumerate(regions)
            if region.PointInSet(q, 1e-6)
        ]
        label = current if current in containing else (containing[0] if containing else current)
        if label is None:
            raise RuntimeError(
                f"{mode} GCS trajectory left every IRIS region at {q}."
            )
        if label != current:
            labels.append(label)
            current = label
    return labels


def plan_manipulation_phase(
    *,
    mode: str,
    config: dict[str, Any],
    reference_q10: np.ndarray,
    X_GC: Any = None,
    final_cup_pose: Any = None,
    scene_cache: dict[str, SceneHandles] | None = None,
) -> ManipulationPhasePlan:
    """Grow mode regions along the reference and optimize through them.

    Raises ``RuntimeError`` when the phase cannot be completed (seed
    collision, disconnected cover, or GCS failure); callers decide
    whether to fall back to the reference path.
    """

    spec = MODE_SPECS[mode]
    reference = np.asarray(reference_q10, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 10 or reference.shape[0] < 2:
        raise ValueError("Reference path must have shape (N, 10) with N >= 2.")
    machinery = prepare_mode_machinery(
        config=config,
        mode=mode,
        reference_q10=reference,
        X_GC=X_GC,
        final_cup_pose=final_cup_pose,
        scene_cache=scene_cache,
    )
    spec_options = _manipulation_spec(config)
    weights = np.asarray(
        config["iris_np2_cover"]["seed_metric_weights"], dtype=float
    )
    site_spec = _site_region_spec(config, spec.site)
    cover_seed = select_phase_seeds(
        reference, weights, spec_options["seed_distance_threshold"]
    )

    regions: list[Any] = []
    results: list[IrisRegionResult] = []
    bridged = False

    def grow(seed: np.ndarray, index: int) -> None:
        delta = spec_options["total_delta"] / (
            len(cover_seed) + spec_options["max_bridges"]
        )
        result = _grow_cover_region(
            scene=machinery.scene,
            checker=machinery.checker,
            parameterization=machinery.parameterization,
            seed=seed,
            domain=_phase_domain(machinery.scene, seed, site_spec, spec_options),
            spec=spec_options,
            delta=delta,
            index=index,
        )
        regions.append(result.region)
        results.append(result)

    start = time.perf_counter()
    for index, seed in enumerate(cover_seed):
        grow(seed, index)

    start_q = reference[0]
    goal_q = reference[-1]

    def connected() -> bool:
        reached = {
            index
            for index, region in enumerate(regions)
            if region.PointInSet(start_q, 1e-8)
        }
        goal_regions = {
            index
            for index, region in enumerate(regions)
            if region.PointInSet(goal_q, 1e-8)
        }
        pending = list(reached)
        while pending:
            i = pending.pop()
            for j, region in enumerate(regions):
                if j not in reached and regions[i].IntersectsWith(region):
                    reached.add(j)
                    pending.append(j)
        return bool(reached & goal_regions)

    for bridge in range(spec_options["max_bridges"] + 1):
        if connected():
            break
        if bridge == spec_options["max_bridges"]:
            raise RuntimeError(
                f"Manipulation mode {mode} cover is disconnected after "
                "bounded bridge generation."
            )
        bridged = True
        chosen = next(
            (
                index
                for index, q in enumerate(reference)
                if not any(
                    region.PointInSet(q, 1e-8) for region in regions
                )
                and not any(
                    np.linalg.norm(q - item.seed) < 1e-6 for item in results
                )
            ),
            None,
        )
        if chosen is None:
            raise RuntimeError(
                f"Manipulation mode {mode} has no new reference seed to bridge."
            )
        grow(reference[chosen], len(regions))
    iris_seconds = time.perf_counter() - start

    edges: list[tuple[int, int]] = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if regions[i].IntersectsWith(regions[j]):
                edges.extend([(i, j), (j, i)])

    base_spec = config["gcs_base_planning"]
    planner_spec = config["whole_body_planning"]
    gcs = GcsTrajectoryOptimization(10, continuous_revolute_joints=[2])
    free_space = gcs.AddRegions(
        regions,
        edges,
        order=int(base_spec["bezier_order"]),
        h_min=0.05,
        h_max=30.0,
        name=f"{mode}_free_space",
    )
    source = gcs.AddRegions([Point(start_q)], order=0, name=f"{mode}_start")
    target = gcs.AddRegions([Point(goal_q)], order=0, name=f"{mode}_goal")
    gcs.AddEdges(source, free_space)
    gcs.AddEdges(free_space, target)

    metric = np.diag(
        [
            float(base_spec["path_length_weight"]),
            float(base_spec["path_length_weight"]),
            float(base_spec["yaw_path_weight"]),
            *([float(planner_spec["joint_path_weight"])] * 7),
        ]
    )
    gcs.AddPathLengthCost(metric)
    gcs.AddTimeCost(float(base_spec["time_weight"]))
    maximum_speed = float(base_spec["maximum_speed"])
    maximum_angular_speed = float(base_spec["maximum_angular_speed"])
    maximum_joint_speed = float(planner_spec["maximum_joint_speed"])
    upper_velocity = np.array(
        [maximum_speed, maximum_speed, maximum_angular_speed]
        + [maximum_joint_speed] * 7,
        dtype=float,
    )
    gcs.AddVelocityBounds(-upper_velocity, upper_velocity)

    solve_start = time.perf_counter()
    trajectory, result = gcs.SolvePath(source, target)
    gcs_seconds = time.perf_counter() - solve_start
    if not result.is_success():
        raise RuntimeError(
            f"Manipulation GCS failed for mode {mode} with "
            f"{result.get_solution_result()} using {result.get_solver_id().name()}."
        )

    duration = float(trajectory.end_time() - trajectory.start_time())
    sample_count = max(
        2,
        int(np.ceil(duration * float(base_spec["samples_per_second"]))) + 1,
    )
    sample_times = np.linspace(
        trajectory.start_time(), trajectory.end_time(), sample_count
    )
    configurations = np.vstack(
        [np.asarray(trajectory.value(t), dtype=float).reshape(-1) for t in sample_times]
    )
    # Preserve the prescribed endpoints exactly.
    configurations[0] = start_q
    configurations[-1] = goal_q

    audit_collisions = 0
    for q in configurations:
        audit_collisions += int(
            not machinery.checker.CheckConfigCollisionFree(
                machinery.parameterization(q)
            )
        )
    if audit_collisions:
        raise RuntimeError(
            f"Manipulation mode {mode} GCS trajectory failed the independent "
            f"numerical audit ({audit_collisions}/{len(configurations)} samples)."
        )

    return ManipulationPhasePlan(
        mode=mode,
        configurations=configurations,
        region_sequence=_region_labels(configurations, regions, mode),
        seed_count=len(cover_seed),
        bridged=bridged,
        iris_solve_seconds=iris_seconds,
        gcs_solve_seconds=gcs_seconds,
        trajectory_duration=duration,
        arm_joint_travel=float(
            np.linalg.norm(np.diff(configurations[:, 3:], axis=0), axis=1).sum()
        ),
        audit_samples=len(configurations),
        audit_collisions=audit_collisions,
        region_results=tuple(results),
    )
