"""IRIS-NP2 generation for a probabilistically collision-free doorway set."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import numpy as np

from pydrake.common import Parallelism, RandomGenerator
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.planning import (
    CollisionCheckerParams,
    IrisNp2,
    IrisNp2Options,
    IrisParameterizationFunction,
    SceneGraphCollisionChecker,
)
from pydrake.solvers import IpoptSolver

from .build_scene import SceneHandles, set_robot_configuration


@dataclass(frozen=True)
class IrisRegionResult:
    region: HPolyhedron
    seed: np.ndarray
    solve_time_seconds: float
    number_of_faces: int
    epsilon: float
    delta: float
    holdout_samples: int
    holdout_collisions: int
    holdout_collision_fraction: float
    attached_object_included: bool


@dataclass(frozen=True)
class IrisCoverRegion:
    semantic_region: str
    result: IrisRegionResult


@dataclass(frozen=True)
class IrisCoverResult:
    regions: list[IrisCoverRegion]
    total_solve_time_seconds: float
    total_delta: float
    epsilon: float

    def regions_by_semantic_label(self) -> dict[str, list[HPolyhedron]]:
        grouped: dict[str, list[HPolyhedron]] = {}
        for item in self.regions:
            grouped.setdefault(item.semantic_region, []).append(item.result.region)
        return grouped


def _reduced_position_indices(scene: SceneHandles) -> np.ndarray:
    indices = list(
        range(
            scene.base_joint.position_start(),
            scene.base_joint.position_start() + scene.base_joint.num_positions(),
        )
    )
    for joint_index in scene.plant.GetJointIndices(scene.arm_instance):
        joint = scene.plant.get_joint(joint_index)
        if joint.num_positions() == 1:
            indices.append(joint.position_start())
    if len(indices) != 10:
        raise RuntimeError(f"Expected 10 reduced positions, got {len(indices)}.")
    return np.asarray(indices, dtype=int)


def _make_parameterization(
    fixed_full_q: np.ndarray,
    reduced_indices: np.ndarray,
) -> Callable[[np.ndarray], np.ndarray]:
    def parameterization(q_reduced: np.ndarray) -> np.ndarray:
        q_array = np.asarray(q_reduced)
        dtype = object if q_array.dtype == object else float
        q_full = np.asarray(fixed_full_q, dtype=dtype).copy()
        q_full[reduced_indices] = q_reduced
        return q_full

    return parameterization


def _make_collision_checker(scene: SceneHandles) -> SceneGraphCollisionChecker:
    params = CollisionCheckerParams()
    params.model = scene.diagram
    robot_instances = [
        scene.base_instance,
        scene.arm_instance,
        scene.gripper_instance,
    ]
    if scene.attached_cup_instance is not None:
        robot_instances.append(scene.attached_cup_instance)
    params.robot_model_instances = robot_instances
    params.edge_step_size = 0.05
    return SceneGraphCollisionChecker(params)


def _domain(
    scene: SceneHandles,
    seed: np.ndarray,
    spec: dict[str, Any],
) -> HPolyhedron:
    lower = np.concatenate(
        [
            np.asarray(spec["base_lower"], dtype=float),
            seed[3:] - float(spec["joint_radius"]),
        ]
    )
    upper = np.concatenate(
        [
            np.asarray(spec["base_upper"], dtype=float),
            seed[3:] + float(spec["joint_radius"]),
        ]
    )
    joint_spec = scene.config["whole_body_planning"]
    lower[3:] = np.maximum(
        lower[3:],
        np.asarray(joint_spec["joint_lower_limits"], dtype=float),
    )
    upper[3:] = np.minimum(
        upper[3:],
        np.asarray(joint_spec["joint_upper_limits"], dtype=float),
    )
    return HPolyhedron.MakeBox(lower, upper)


def generate_doorway_iris_region(
    scene: SceneHandles,
    carry_arm_q: np.ndarray | list[float],
) -> IrisRegionResult:
    """Grow and independently audit one 10-D doorway region with IRIS-NP2."""

    spec = scene.config["iris_np2_doorway"]
    seed = np.concatenate(
        [
            np.asarray(spec["seed_base_q"], dtype=float),
            np.asarray(carry_arm_q, dtype=float),
        ]
    )
    if seed.shape != (10,):
        raise ValueError("IRIS-NP2 doorway seed must have dimension 10.")

    set_robot_configuration(
        scene,
        seed[:3],
        seed[3:],
        scene.gripper_closed_positions,
    )
    scene.plant.SetFreeBodyPose(
        scene.plant_context,
        scene.cup_body,
        scene.cup_initial_pose,
    )
    fixed_full_q = scene.plant.GetPositions(scene.plant_context).copy()
    reduced_indices = _reduced_position_indices(scene)
    parameterization = _make_parameterization(fixed_full_q, reduced_indices)
    checker = _make_collision_checker(scene)
    if not checker.CheckConfigCollisionFree(parameterization(seed)):
        raise RuntimeError("The IRIS-NP2 doorway seed is in collision.")

    domain = _domain(scene, seed, spec)
    starting_ellipsoid = Hyperellipsoid.MakeHypersphere(
        float(spec["starting_radius"]),
        seed,
    )
    options = IrisNp2Options()
    options.solver = IpoptSolver()
    options.add_hyperplane_if_solve_fails = True
    options.parameterization = IrisParameterizationFunction(parameterization, 10)
    options.ray_sampler_options.num_particles_to_walk_towards = int(
        spec["ray_particles"]
    )
    sampled = options.sampled_iris_options
    sampled.num_particles = int(spec["particles"])
    sampled.max_iterations = int(spec["max_iterations"])
    sampled.max_iterations_separating_planes = int(spec["max_separating_iterations"])
    sampled.max_separating_planes_per_iteration = int(spec["max_planes_per_iteration"])
    sampled.parallelism = Parallelism(1)
    sampled.sample_particles_in_parallel = False
    sampled.verbose = bool(spec["verbose"])
    sampled.delta = float(spec["delta"])
    sampled.epsilon = float(spec["epsilon"])
    sampled.configuration_space_margin = float(spec["configuration_space_margin"])
    sampled.mixing_steps = int(spec["mixing_steps"])
    sampled.random_seed = int(spec["random_seed"])

    start = time.perf_counter()
    region = IrisNp2(checker, starting_ellipsoid, domain, options)
    solve_time = time.perf_counter() - start
    if not region.PointInSet(seed):
        raise RuntimeError("IRIS-NP2 returned a region that excludes its seed.")

    holdout_samples = int(spec["holdout_samples"])
    generator = RandomGenerator(int(spec["holdout_random_seed"]))
    sample = region.ChebyshevCenter()
    collisions = 0
    for _ in range(holdout_samples):
        sample = region.UniformSample(
            generator,
            sample,
            mixing_steps=int(spec["holdout_mixing_steps"]),
        )
        collisions += int(
            not checker.CheckConfigCollisionFree(parameterization(sample))
        )

    return IrisRegionResult(
        region=region,
        seed=seed,
        solve_time_seconds=solve_time,
        number_of_faces=int(region.A().shape[0]),
        epsilon=sampled.epsilon,
        delta=sampled.delta,
        holdout_samples=holdout_samples,
        holdout_collisions=collisions,
        holdout_collision_fraction=collisions / holdout_samples,
        attached_object_included=scene.attached_cup_geometry_id is not None,
    )


def result_to_dict(result: IrisRegionResult) -> dict[str, Any]:
    return {
        "algorithm": "Drake IrisNp2",
        "nonlinear_solver": "Ipopt",
        "add_hyperplane_if_nonlinear_solve_fails": True,
        "ambient_dimension": int(result.region.ambient_dimension()),
        "seed": result.seed.tolist(),
        "A": result.region.A().tolist(),
        "b": result.region.b().tolist(),
        "number_of_faces": result.number_of_faces,
        "solve_time_seconds": result.solve_time_seconds,
        "attached_object_included": result.attached_object_included,
        "probabilistic_guarantee": {
            "epsilon": result.epsilon,
            "delta": result.delta,
            "statement": (
                "With probability at least 1-delta, the in-collision volume "
                "fraction is no greater than epsilon."
            ),
        },
        "independent_holdout_audit": {
            "samples": result.holdout_samples,
            "collisions": result.holdout_collisions,
            "collision_fraction": result.holdout_collision_fraction,
        },
    }


def _base_region_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(spec["name"]): spec
        for spec in config["gcs_base_planning"]["regions"]
    }


def _base_in_region(q: np.ndarray, spec: dict[str, Any], tolerance: float = 1e-7) -> bool:
    lower = np.asarray(spec["lower"], dtype=float)
    upper = np.asarray(spec["upper"], dtype=float)
    yaw_lower, yaw_upper = spec["yaw_bounds"]
    return bool(
        np.all(q[:2] >= lower - tolerance)
        and np.all(q[:2] <= upper + tolerance)
        and float(yaw_lower) - tolerance <= q[2] <= float(yaw_upper) + tolerance
    )


def _path_region_labels(
    configurations: np.ndarray,
    specs: dict[str, dict[str, Any]],
    allowed_region_names: list[str],
) -> list[str]:
    labels: list[str] = []
    current: str | None = None
    for q in configurations:
        containing = [
            name
            for name in allowed_region_names
            if _base_in_region(q, specs[name])
        ]
        if current in containing:
            label = current
        elif containing:
            label = containing[0]
        elif current is not None:
            # An IRIS override may contain a narrow collision-free extension
            # beyond its original axis-aligned semantic domain. Keep the
            # current label through that transition band.
            label = current
        else:
            raise RuntimeError(f"Bootstrap path left every semantic region at {q[:3]}.")
        labels.append(label)
        current = label
    return labels


def _select_cover_seeds(
    configurations: np.ndarray,
    labels: list[str],
    spec: dict[str, Any],
    spatial_specs: dict[str, dict[str, Any]],
) -> list[tuple[str, np.ndarray]]:
    weights = np.asarray(spec["seed_metric_weights"], dtype=float)
    threshold = float(spec["seed_distance_threshold"])
    selected_indices = [0]
    last = configurations[0]
    for index, q in enumerate(configurations[1:-1], start=1):
        if not _base_in_region(q, spatial_specs[labels[index]]):
            continue
        label_changed = labels[index] != labels[index - 1]
        weighted_distance = float(np.linalg.norm((q - last) * weights))
        if label_changed or weighted_distance >= threshold:
            selected_indices.append(index)
            last = q
    selected_indices.append(len(configurations) - 1)

    # Preserve both sampled sides of every transition. Each seed is known to
    # be safe and lies in its own semantic domain; dense neighboring regions
    # are expected to inflate into the pre-existing spatial overlap.
    transition_seeds: list[tuple[str, np.ndarray]] = []
    for index in range(1, len(labels)):
        if labels[index] == labels[index - 1]:
            continue
        previous_label = labels[index - 1]
        next_label = labels[index]
        previous_candidates = [
            candidate
            for candidate in range(index - 1, -1, -1)
            if _base_in_region(configurations[candidate], spatial_specs[previous_label])
        ]
        next_candidates = [
            candidate
            for candidate in range(index, len(labels))
            if _base_in_region(configurations[candidate], spatial_specs[next_label])
        ]
        if not previous_candidates or not next_candidates:
            raise RuntimeError(
                f"Could not find safe in-domain seeds for {previous_label} -> {next_label}."
            )
        transition_seeds.append(
            (previous_label, configurations[previous_candidates[0]].copy())
        )
        transition_seeds.append(
            (next_label, configurations[next_candidates[0]].copy())
        )

    seeds = [(labels[index], configurations[index].copy()) for index in selected_indices]
    seeds.extend(transition_seeds)
    deduplicated: list[tuple[str, np.ndarray]] = []
    for label, seed in seeds:
        if any(
            existing_label == label and np.linalg.norm(existing_seed - seed) < 1e-6
            for existing_label, existing_seed in deduplicated
        ):
            continue
        deduplicated.append((label, seed))
    return deduplicated


def _cover_domain(
    scene: SceneHandles,
    seed: np.ndarray,
    spatial_spec: dict[str, Any],
    cover_spec: dict[str, Any],
) -> HPolyhedron:
    lower = np.concatenate(
        [
            np.asarray(spatial_spec["lower"], dtype=float),
            np.asarray([spatial_spec["yaw_bounds"][0]], dtype=float),
            seed[3:] - float(cover_spec["joint_radius"]),
        ]
    )
    upper = np.concatenate(
        [
            np.asarray(spatial_spec["upper"], dtype=float),
            np.asarray([spatial_spec["yaw_bounds"][1]], dtype=float),
            seed[3:] + float(cover_spec["joint_radius"]),
        ]
    )
    joint_spec = scene.config["whole_body_planning"]
    lower[3:] = np.maximum(lower[3:], joint_spec["joint_lower_limits"])
    upper[3:] = np.minimum(upper[3:], joint_spec["joint_upper_limits"])
    return HPolyhedron.MakeBox(lower, upper)


def _grow_cover_region(
    *,
    scene: SceneHandles,
    checker: SceneGraphCollisionChecker,
    parameterization: Callable[[np.ndarray], np.ndarray],
    seed: np.ndarray,
    domain: HPolyhedron,
    spec: dict[str, Any],
    delta: float,
    index: int,
    add_hyperplane_if_solve_fails: bool = True,
) -> IrisRegionResult:
    if not checker.CheckConfigCollisionFree(parameterization(seed)):
        error = RuntimeError(f"IRIS cover seed {index} is in collision.")
        error.details = {"seed_index": index, "seed": seed.tolist(),
                         "scope": "Rejected by padded carrying collision checker; not a proof that every route fails."}
        set_robot_configuration(scene, seed[:3], seed[3:], scene.gripper_closed_positions)
        query = scene.scene_graph.get_query_output_port().Eval(
            scene.scene_graph.GetMyContextFromRoot(scene.root_context))
        pairs = query.ComputeSignedDistancePairwiseClosestPoints(.1)
        inspector = query.inspector()
        error.details["raw_geometry"] = [
            {"distance_m": float(p.distance), "a": inspector.GetName(p.id_A),
             "b": inspector.GetName(p.id_B)}
            for p in sorted(pairs, key=lambda p: p.distance)[:8]]
        raise error
    options = IrisNp2Options()
    options.solver = IpoptSolver()
    options.add_hyperplane_if_solve_fails = add_hyperplane_if_solve_fails
    options.parameterization = IrisParameterizationFunction(parameterization, 10)
    options.ray_sampler_options.num_particles_to_walk_towards = int(
        spec["ray_particles"]
    )
    sampled = options.sampled_iris_options
    sampled.num_particles = int(spec["particles"])
    sampled.max_iterations = int(spec["max_iterations"])
    sampled.max_iterations_separating_planes = int(spec["max_separating_iterations"])
    sampled.max_separating_planes_per_iteration = int(spec["max_planes_per_iteration"])
    sampled.parallelism = Parallelism(1)
    sampled.sample_particles_in_parallel = False
    sampled.verbose = bool(spec["verbose"])
    sampled.delta = delta
    sampled.epsilon = float(spec["epsilon"])
    sampled.configuration_space_margin = float(spec["configuration_space_margin"])
    sampled.mixing_steps = int(spec["mixing_steps"])
    sampled.random_seed = int(spec["random_seed"]) + index

    start = time.perf_counter()
    region = IrisNp2(
        checker,
        Hyperellipsoid.MakeHypersphere(float(spec["starting_radius"]), seed),
        domain,
        options,
    )
    solve_time = time.perf_counter() - start
    if not region.PointInSet(seed):
        raise RuntimeError(f"IRIS cover region {index} excludes its seed.")

    holdout_samples = int(spec["holdout_samples_per_region"])
    generator = RandomGenerator(int(spec["holdout_random_seed"]) + index)
    sample = region.ChebyshevCenter()
    collisions = 0
    for _ in range(holdout_samples):
        sample = region.UniformSample(
            generator,
            sample,
            mixing_steps=int(spec["holdout_mixing_steps"]),
        )
        collisions += int(
            not checker.CheckConfigCollisionFree(parameterization(sample))
        )
    return IrisRegionResult(
        region=region,
        seed=seed,
        solve_time_seconds=solve_time,
        number_of_faces=int(region.A().shape[0]),
        epsilon=sampled.epsilon,
        delta=sampled.delta,
        holdout_samples=holdout_samples,
        holdout_collisions=collisions,
        holdout_collision_fraction=collisions / holdout_samples,
        attached_object_included=scene.attached_cup_geometry_id is not None,
    )


def generate_carrying_iris_cover(
    scene: SceneHandles,
    bootstrap_configurations: np.ndarray,
    allowed_region_names: list[str],
) -> IrisCoverResult:
    """Grow an overlapping attached-object IRIS cover along a safe path."""

    configurations = np.asarray(bootstrap_configurations, dtype=float)
    if configurations.ndim != 2 or configurations.shape[1] != 10:
        raise ValueError("Bootstrap configurations must have shape (N, 10).")
    cover_spec = scene.config["iris_np2_cover"]
    spatial_specs = _base_region_specs(scene.config)
    labels = _path_region_labels(configurations, spatial_specs, allowed_region_names)
    seeds = _select_cover_seeds(
        configurations,
        labels,
        cover_spec,
        spatial_specs,
    )
    # Reserve a fixed budget for bridging disconnected components.
    max_bridges = 12
    individual_delta = float(cover_spec["total_delta"]) / (len(seeds) + max_bridges)

    set_robot_configuration(
        scene,
        configurations[0, :3],
        configurations[0, 3:],
        scene.gripper_closed_positions,
    )
    scene.plant.SetFreeBodyPose(
        scene.plant_context,
        scene.cup_body,
        scene.cup_initial_pose,
    )
    fixed_full_q = scene.plant.GetPositions(scene.plant_context).copy()
    parameterization = _make_parameterization(
        fixed_full_q,
        _reduced_position_indices(scene),
    )
    checker = _make_collision_checker(scene)

    regions: list[IrisCoverRegion] = []
    for index, (label, seed) in enumerate(seeds):
        result = _grow_cover_region(
            scene=scene,
            checker=checker,
            parameterization=parameterization,
            seed=seed,
            domain=_cover_domain(
                scene,
                seed,
                spatial_specs[label],
                cover_spec,
            ),
            spec=cover_spec,
            delta=individual_delta,
            index=index,
        )
        regions.append(IrisCoverRegion(label, result))
    def connected_component():
        sets = [item.result.region for item in regions]
        reached = {i for i, region in enumerate(sets) if region.PointInSet(configurations[0], 1e-8)}
        pending = list(reached)
        while pending:
            i = pending.pop()
            for j, region in enumerate(sets):
                if j not in reached and sets[i].IntersectsWith(region):
                    reached.add(j); pending.append(j)
        return reached, any(sets[i].PointInSet(configurations[-1], 1e-8) for i in reached)
    for bridge in range(max_bridges + 1):
        reached, connected = connected_component()
        if connected:
            break
        if bridge == max_bridges or not reached:
            raise RuntimeError("IRIS cover remains disconnected after bounded bridge generation")
        # First point along the checked guide outside the start component.
        chosen = next((i for i, q in enumerate(configurations)
                       if not any(regions[j].result.region.PointInSet(q, 1e-8) for j in reached)
                       and not any(np.linalg.norm(q-item.result.seed)<1e-6 for item in regions)), None)
        if chosen is None:
            raise RuntimeError("No new guide seed available to bridge disconnected IRIS cover")
        seed = configurations[chosen]; label = labels[chosen]
        result = _grow_cover_region(scene=scene, checker=checker, parameterization=parameterization,
            seed=seed, domain=_cover_domain(scene, seed, spatial_specs[label], cover_spec),
            spec=cover_spec, delta=individual_delta, index=len(regions))
        regions.append(IrisCoverRegion(label, result))
    return IrisCoverResult(
        regions=regions,
        total_solve_time_seconds=sum(item.result.solve_time_seconds for item in regions),
        total_delta=float(cover_spec["total_delta"]),
        epsilon=float(cover_spec["epsilon"]),
    )


def cover_to_dict(cover: IrisCoverResult) -> dict[str, Any]:
    total_holdout = sum(item.result.holdout_samples for item in cover.regions)
    total_collisions = sum(item.result.holdout_collisions for item in cover.regions)
    return {
        "algorithm": "Drake IrisNp2 trajectory-seeded cover",
        "nonlinear_solver": "Ipopt",
        "add_hyperplane_if_nonlinear_solve_fails": True,
        "ambient_dimension": 10,
        "number_of_regions": len(cover.regions),
        "total_solve_time_seconds": cover.total_solve_time_seconds,
        "epsilon_per_region": cover.epsilon,
        "total_delta_union_bound": cover.total_delta,
        "all_regions_attached_object_aware": all(
            item.result.attached_object_included for item in cover.regions
        ),
        "independent_holdout_audit": {
            "samples": total_holdout,
            "collisions": total_collisions,
            "collision_fraction": total_collisions / total_holdout,
        },
        "regions": [
            {
                "semantic_region": item.semantic_region,
                **result_to_dict(item.result),
            }
            for item in cover.regions
        ],
    }
