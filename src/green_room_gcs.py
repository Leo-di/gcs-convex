"""Staged green-room GCS: tucked navigation, contained reach, close and lift.

The operation cover uses configured official IRIS-NP2 polytopes or the
legacy 7-joint box corridor. Both are restricted to domains with conservative
whole-robot room bounds. IRIS mode requires clean independent collision and
unfiltered grasp holdouts; this is not continuous collision certification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point
from pydrake.math import RigidTransform
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver

from .build_scene import build_scene, set_robot_configuration
from .collision_checker import evaluate_collision_status
from .geometry_outbound_gcs import plan_geometry_outbound
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .grasp_planner import cartesian_arm_path, plan_pick
from .safe_pregrasp import plan_pregrasp
from .scene_spec import load_scene_config
from .semantic_policy import policy_from_config, accepts
from .whole_robot_room import WholeRobotRoomOracle, arm_point_reach_bound, shape_support


def room_oracle(scene):
    spec = scene.config["green_room"]
    return WholeRobotRoomOracle.from_xy_bounds(
        scene, spec["x_bounds"], spec["y_bounds"], margin=spec["margin_m"]
    )


def operation_oracle(scene, X_GC=None):
    """Spatial constraints used while constructing manipulation domains."""
    oracle = room_oracle(scene)
    if scene.config.get("robot_keep_clear") or scene.config.get(
        "human_passage", {}
    ).get("enforce"):
        from .robot_keep_clear import OperationSpatialOracle

        return OperationSpatialOracle(oracle, X_GC)
    return oracle


def guide_edge_validator(scene, base, arm, gripper, *, semantic, X_GC=None):
    from .green_room_iris import make_checker
    from .contact_boundary import support_boundary

    mode_scene, checker, mapping = make_checker(scene.config, base, arm, gripper, X_GC)
    oracle = operation_oracle(scene, X_GC)

    def valid_edge(a, b):
        count = max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.005)) + 1)
        for t in np.linspace(0, 1, count):
            q = (1 - t) * a + t * b
            full = mapping(q)
            if not checker.CheckConfigCollisionFree(full):
                if X_GC is None or not support_boundary(mode_scene, full):
                    return False
            if semantic and not oracle.query(np.r_[base, q], gripper)["satisfied"]:
                return False
        return True

    return valid_edge


def semantic_box_bound(oracle, base, lower, upper, gripper, reach):
    """Bound spatial-rule residuals over a fixed-base arm box.

    The base residual is unchanged. Arm motion consumes at most
    ``reach * sum(half_widths)`` of the center residual.
    """
    center = (lower + upper) / 2
    query = oracle.query(np.r_[base, center], gripper)
    residual = query["residual_m"]
    displacement = float(reach * np.sum((upper - lower) / 2))
    # The base is fixed in this seven-dimensional operation domain. Its exact
    # margin does not decrease when arm joints vary. Apply the Lipschitz bound
    # only to arm/gripper geometry, then take the whole-robot minimum.
    bound = min(
        query["base_residual_m"], query["arm_gripper_residual_m"] - displacement
    )
    return {
        "center_residual_m": residual,
        "point_displacement_bound_m": displacement,
        "fixed_base_residual_m": query["base_residual_m"],
        "arm_gripper_center_residual_m": query["arm_gripper_residual_m"],
        "whole_box_semantic_lower_bound_m": bound,
    }


def make_operation_cover(scene, base, reference, gripper, *, semantic=True, X_GC=None):
    """Construct local input domains, reducing padding or bisecting as needed."""
    spec = scene.config["green_room"]
    oracle = operation_oracle(scene, X_GC)
    reach = arm_point_reach_bound(scene, oracle)
    joint = scene.config["whole_body_planning"]
    domain_lo, domain_hi = (
        np.array(joint["joint_lower_limits"]),
        np.array(joint["joint_upper_limits"]),
    )
    stride, padding = int(spec["corridor_stride"]), float(spec["corridor_padding_rad"])
    if stride < 1 or padding <= 0:
        raise ValueError("Positive corridor stride and padding required")
    indices = sorted(set([*range(0, len(reference), stride), len(reference) - 1]))
    regions, records = [], []

    def add(a, b, depth=0):
        local_padding = padding
        if semantic:
            edge_lo = np.maximum(np.minimum(a, b), domain_lo)
            edge_hi = np.minimum(np.maximum(a, b), domain_hi)
            edge_bound = semantic_box_bound(
                oracle, base, edge_lo, edge_hi, gripper, reach
            )
            # Subdivision shortens the guide edge, but cannot reduce a fixed
            # padding floor. Allocate padding from the remaining arm-motion
            # budget so valid near-boundary guides can have positive domains.
            available = (
                edge_bound["arm_gripper_center_residual_m"] - 1e-6
            ) / reach - float(np.sum((edge_hi - edge_lo) / 2))
            if available > 0:
                local_padding = min(padding, available / (2 * len(edge_lo)))
        lo = np.maximum(np.minimum(a, b) - local_padding, domain_lo)
        hi = np.minimum(np.maximum(a, b) + local_padding, domain_hi)
        if (
            np.any(lo > hi)
            or np.any(a < lo)
            or np.any(a > hi)
            or np.any(b < lo)
            or np.any(b > hi)
        ):
            raise ValueError("Reference violates joint bounds")
        bound = semantic_box_bound(oracle, base, lo, hi, gripper, reach)
        if semantic and bound["whole_box_semantic_lower_bound_m"] < 1e-6:
            if depth >= 16:
                raise RuntimeError(
                    "Cannot construct a contained operation box; no whole-C-space infeasibility claim"
                )
            midpoint = (a + b) / 2
            add(a, midpoint, depth + 1)
            add(midpoint, b, depth + 1)
            return
        regions.append(HPolyhedron.MakeBox(lo, hi))
        records.append(
            {
                "lower": lo.tolist(),
                "upper": hi.tolist(),
                "guide_start": np.asarray(a).tolist(),
                "guide_end": np.asarray(b).tolist(),
                "corridor_padding_rad": float(local_padding),
                **bound,
            }
        )

    for i, j in zip(indices, indices[1:]):
        add(np.asarray(reference[i]), np.asarray(reference[j]))
    if not regions:
        raise ValueError("Need a nonempty operation reference")
    return regions, records, reach


def solve_operation(
    scene, base, reference, gripper, name, *, semantic=True, X_GC=None, artifacts=None
):
    began = time.perf_counter()
    method = scene.config["green_room"].get("region_method", "reference_corridor")
    if method == "iris_np2":
        from .operation_refinement import build_audited_cover

        refinement_artifact = (
            Path(artifacts).with_name(Path(artifacts).stem + "_refinement.json")
            if artifacts is not None
            else None
        )
        regions, records, reach, refinement = build_audited_cover(
            scene,
            base,
            reference,
            gripper,
            X_GC=X_GC,
            semantic=semantic,
            artifact=refinement_artifact,
        )
    elif method == "reference_corridor":
        regions, records, reach = make_operation_cover(
            scene, base, reference, gripper, semantic=semantic, X_GC=X_GC
        )
    else:
        raise ValueError(f"Unknown operation region method: {method}")
    planner = GcsTrajectoryOptimization(7)
    edges = (
        [
            (i, j)
            for i in range(len(regions))
            for j in range(len(regions))
            if i != j and regions[i].IntersectsWith(regions[j])
        ]
        if method == "iris_np2"
        else [(i, i + 1) for i in range(len(regions) - 1)]
    )
    sources = [i for i, r in enumerate(regions) if r.PointInSet(reference[0], 1e-7)]
    targets = [i for i, r in enumerate(regions) if r.PointInSet(reference[-1], 1e-7)]
    if artifacts is not None:
        Path(artifacts).write_text(
            json.dumps(
                {
                    "region_method": method,
                    "regions": records,
                    "edges": edges,
                    "sources": sources,
                    "targets": targets,
                },
                indent=2,
            )
        )
    if not sources or not targets:
        raise RuntimeError("Operation IRIS cover excludes endpoint; no fallback")
    reached = set(sources)
    while True:
        new = reached | {j for i, j in edges if i in reached}
        if new == reached:
            break
        reached = new
    if not reached.intersection(targets):
        raise RuntimeError(
            "Operation region cover disconnected; no fallback or infeasibility claim"
        )
    cover = planner.AddRegions(regions, edges, order=1, h_min=0.01, h_max=20, name=name)
    source = planner.AddRegions([Point(reference[0])], order=0, name=f"{name}_start")
    target = planner.AddRegions([Point(reference[-1])], order=0, name=f"{name}_goal")
    planner.AddEdges(source, cover, edges_between_regions=[(0, i) for i in sources])
    planner.AddEdges(cover, target, edges_between_regions=[(i, 0) for i in targets])
    planner.AddPathLengthCost(1.0)
    planner.AddTimeCost(0.05)
    speed = float(scene.config["whole_body_planning"]["maximum_joint_speed"])
    planner.AddVelocityBounds(np.full(7, -speed), np.full(7, speed))
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.max_rounded_paths = 5
    from .planner_randomness import planning_seed

    options.rounding_seed = planning_seed(scene.config, "gcs_rounding", 0)
    options.solver = ClarabelSolver()
    options.restriction_solver = ClarabelSolver()
    prepared = time.perf_counter() - began
    tick = time.perf_counter()
    trajectory, result = planner.SolvePath(source, target, options)
    if not result.is_success():
        raise RuntimeError(f"{name} GCS failed: {result.get_solution_result()}")
    seconds = time.perf_counter() - tick
    path, selected_boxes = [], []
    continuity = 0.0
    for i in range(trajectory.get_number_of_segments()):
        segment = trajectory.segment(i)
        a = segment.value(segment.start_time()).ravel()
        b = segment.value(segment.end_time()).ravel()
        if path:
            continuity = max(continuity, float(np.max(np.abs(path[-1] - a))))
        path.extend([a, b])
        if segment.end_time() > segment.start_time() + 1e-10:
            containing = [
                j
                for j, r in enumerate(regions)
                if r.PointInSet(a, 1e-6) and r.PointInSet(b, 1e-6)
            ]
            if not containing:
                raise RuntimeError("GCS segment is not contained in an operation box")
            selected_boxes.append(containing[0])
    path = np.asarray(path)
    endpoint_error = max(
        float(np.max(abs(path[0] - reference[0]))),
        float(np.max(abs(path[-1] - reference[-1]))),
    )
    if max(endpoint_error, continuity) > 1e-6:
        raise RuntimeError("GCS endpoint/continuity check failed")
    from .operation_path import shortcut_path, cover_edge_validator

    raw_path = path.copy()
    inside_cover = cover_edge_validator(regions)
    path, simplification = shortcut_path(path, inside_cover)
    if not all(inside_cover(a, b) for a, b in zip(path, path[1:])):
        raise RuntimeError("Simplified operation edge leaves audited region union")
    simplification.update(
        all_output_edges_in_cover=True,
        scope="Post-GCS shortcuts inside the entire audited polytope union; independent trajectory and contact audits still required",
    )
    if artifacts is not None:
        Path(artifacts).with_name(Path(artifacts).stem + "_path.json").write_text(
            json.dumps(
                {
                    "gcs_path": raw_path.tolist(),
                    "execution_path": path.tolist(),
                    "simplification": simplification,
                },
                indent=2,
            )
        )
    return (
        path,
        {
            "name": name,
            "region_method": method,
            "region_count": len(regions),
            "edges": edges,
            "independent_variables": 7,
            "order": 1,
            "semantic_enforced": semantic,
            "prepare_seconds": prepared,
            "solve_seconds": seconds,
            "objective": float(result.get_optimal_cost()),
            "reach_bound_m": reach,
            "minimum_whole_box_semantic_bound_m": min(
                r["whole_box_semantic_lower_bound_m"] for r in records
            ),
            "endpoint_error_rad": endpoint_error,
            "continuity_error_rad": continuity,
            "selected_boxes": selected_boxes,
            "boxes": records,
            "path_simplification": simplification,
            "region_audit": refinement["final_audit"] if method == "iris_np2" else None,
            "refinement": refinement if method == "iris_np2" else None,
            "scope": "Ordered guide-seeded 7D regions; analytic room bound; sampled collision audits; no route discovery",
        },
        planner.GetGraphvizString(),
    )


def append_motion(frames, base_path, arm_path, gripper_path, phase):
    """Retime a piecewise-linear path into uniform execution reference frames."""
    if not (len(base_path) == len(arm_path) == len(gripper_path)):
        raise ValueError("Motion arrays must have equal lengths")
    triples = list(zip(base_path, arm_path, gripper_path))
    if any(
        np.concatenate(t).shape != (12,) or not np.all(np.isfinite(np.concatenate(t)))
        for t in triples
    ):
        raise ValueError("Finite 12D motion configurations required")
    if not triples:
        raise ValueError("Empty phase")
    if frames:
        previous = np.r_[
            frames[-1]["base_q"], frames[-1]["arm_q"], frames[-1]["gripper_q"]
        ]
        first = np.concatenate(triples[0])
        if np.max(abs(previous - first)) > 1e-6:
            raise RuntimeError(f"Configuration jump entering {phase}")
    for i, (base, arm, grip) in enumerate(triples):
        target = np.r_[base, arm, grip]
        previous = target if i == 0 else np.concatenate(triples[i - 1])
        count = max(
            1,
            int(
                np.ceil(
                    max(
                        np.max(abs(target[:3] - previous[:3])) / 0.02,
                        np.max(abs(target[3:10] - previous[3:10])) / 0.025,
                        np.max(abs(target[10:] - previous[10:])) / 0.0005,
                    )
                )
            ),
        )
        for alpha in np.linspace(0, 1, count + 1)[1:]:
            q = (1 - alpha) * previous + alpha * target
            frames.append(
                {
                    "base_q": q[:3].tolist(),
                    "arm_q": q[3:10].tolist(),
                    "gripper_q": q[10:].tolist(),
                    "phase": phase,
                }
            )


def cup_room_residual(scene, oracle, cup_pose):
    inspector = scene.scene_graph.model_inspector()
    values = []
    for gid in scene.plant.GetCollisionGeometriesForBody(scene.cup_body):
        pose = cup_pose @ inspector.GetPoseInFrame(gid)
        shape = inspector.GetShape(gid)
        for n, b in zip(oracle.normals, oracle.offsets):
            values.append(
                float(
                    b
                    - n @ pose.translation()
                    - shape_support(shape, pose.rotation().matrix().T @ n)
                    - oracle.margin
                )
            )
    if not values:
        raise ValueError("Cup has no collision volume")
    return min(values)


def audit_reference(
    scene, frames, dock, tucked, X_GC, *, phase_local=False, max_samples=200000
):
    from .semantic_plan import configuration

    if not frames or type(max_samples) is not int or max_samples < 1:
        raise ValueError("Nonempty reference and positive sample budget required")
    qs = [configuration(f) for f in frames]
    if phase_local and len({f["phase"] for f in frames}) != 1:
        raise ValueError("Local acceptance requires exactly one phase")
    lower = np.r_[
        scene.config["robot"]["lower_limits"],
        scene.config["whole_body_planning"]["joint_lower_limits"],
    ]
    upper = np.r_[
        scene.config["robot"]["upper_limits"],
        scene.config["whole_body_planning"]["joint_upper_limits"],
    ]
    limits_ok = all(
        np.all(q[:10] >= lower - 1e-8) and np.all(q[:10] <= upper + 1e-8) for q in qs
    )
    oracle = room_oracle(scene)
    spec = scene.config["green_room"]
    steps = np.r_[
        [spec["maximum_audit_base_step_m"]] * 3,
        [spec["maximum_audit_joint_step_rad"]] * 7,
        [0.0005] * 2,
    ]
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("Positive finite reference audit spacings required")
    rows, grasp_rows = [], []
    minimum_collision = float("inf")
    nav_arm_error = op_base_error = 0.0
    operation_min = float("inf")
    cup_min = float("inf")
    for i, b in enumerate(frames):
        a = frames[max(0, i - 1)]
        qa = np.r_[a["base_q"], a["arm_q"], a["gripper_q"]]
        qb = np.r_[b["base_q"], b["arm_q"], b["gripper_q"]]
        count = max(1, int(np.ceil(np.max(np.abs(qb - qa) / steps))))
        if len(rows) + count + 1 > max_samples:
            raise RuntimeError("Reference audit sample budget exhausted")
        for alpha in np.linspace(0, 1, count + 1):
            q = (1 - alpha) * qa + alpha * qb
            active = b["phase"] != "navigate_to_pickup"
            attached = b["phase"] in ("lift_cup", "whole_body_carry_to_dropoff")
            inside = oracle.query(q[:10], q[10:])
            pose = scene.cup_initial_pose
            if attached:
                pose = (
                    scene.plant.EvalBodyPoseInWorld(
                        scene.plant_context, scene.gripper_body
                    )
                    @ X_GC
                )
            scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, pose)
            collision = evaluate_collision_status(
                scene, q[:3], arm_q=q[3:10], gripper_q=q[10:], influence_distance=0.5
            )
            raw = measure_cup_gripper(scene)
            grasp_rows.append({"attached": attached, "cup_gripper_audit": raw})
            minimum_collision = min(
                minimum_collision, collision.minimum_signed_distance
            )
            cup_value = cup_room_residual(scene, oracle, pose)
            cup_min = min(cup_min, cup_value)
            if active:
                operation_min = min(operation_min, inside["residual_m"])
                op_base_error = max(op_base_error, float(np.max(abs(q[:3] - dock))))
            else:
                nav_arm_error = max(nav_arm_error, float(np.max(abs(q[3:10] - tucked))))
            rows.append(
                {
                    "edge": i,
                    "alpha": float(alpha),
                    "phase": b["phase"],
                    "semantic_active": active,
                    "whole_robot_residual_m": inside["residual_m"],
                    "worst": inside["worst"],
                    "collision_distance_m": collision.minimum_signed_distance,
                    "cup_room_residual_m": cup_value,
                }
            )
    grasp = summarize_grasp_audit(grasp_rows)
    operations = [f for f in frames if f["phase"] != "navigate_to_pickup"]
    navigation = [f for f in frames if f["phase"] == "navigate_to_pickup"]
    if not phase_local and (not operations or not navigation):
        raise ValueError("Complete cup reference requires navigation and operation")
    switch_error = (
        float(np.max(abs(configuration(navigation[-1]) - configuration(operations[0]))))
        if navigation and operations
        else 0.0
    )
    nav_inside = (
        oracle.query(
            configuration(navigation[-1])[:10], configuration(navigation[-1])[10:]
        )
        if navigation
        else {"satisfied": True, "scope": "Checked by navigation phase"}
    )
    report = {
        "samples": len(rows),
        "minimum_collision_distance_m": minimum_collision,
        "minimum_operation_room_residual_m": operation_min,
        "minimum_cup_room_residual_m": cup_min,
        "navigation_endpoint_inside": nav_inside,
        "phase_switch_error": switch_error,
        "navigation_arm_error_rad": nav_arm_error,
        "operation_base_error_m_rad": op_base_error,
        "unfiltered_grasp_audit": grasp,
        "scope": "Dense samples of every executable linear reference edge; not continuous collision certification",
    }
    report["joint_limits_satisfied"] = bool(limits_ok)
    report["geometry_satisfied"] = bool(
        limits_ok
        and minimum_collision >= -1e-7
        and max(switch_error, nav_arm_error, op_base_error) < 1e-6
        and grasp["satisfied"]
    )
    endpoint_frames = [
        f
        for j, f in enumerate(frames)
        if f["phase"] != "navigate_to_pickup"
        and (
            j == 0
            or j == len(frames) - 1
            or frames[j - 1]["phase"] != f["phase"]
            or frames[min(j + 1, len(frames) - 1)]["phase"] != f["phase"]
        )
    ]
    report["goal_semantics_satisfied"] = bool(
        nav_inside["satisfied"]
        and all(
            oracle.query(np.r_[f["base_q"], f["arm_q"]], f["gripper_q"])["satisfied"]
            for f in endpoint_frames
        )
    )
    report["motion_semantics_satisfied"] = bool(
        operation_min >= -1e-8 and cup_min >= -1e-8
    )
    report["satisfied"] = bool(
        limits_ok
        and minimum_collision >= -1e-7
        and operation_min >= -1e-8
        and cup_min >= -1e-8
        and nav_inside["satisfied"]
        and max(switch_error, nav_arm_error, op_base_error) < 1e-6
        and grasp["satisfied"]
    )
    from .semantic_plan import audit_cup_plan

    if phase_local and scene.config.get("grounding_semantic_plan") is not None:
        from .semantic_plan import SemanticPlanEvaluator

        plan = scene.config["grounding_semantic_plan"]
        phases = [p for p in plan["phases"] if p["id"] == frames[0]["phase"]]
        report["semantic_plan_audit"] = SemanticPlanEvaluator(
            scene, dict(plan, phases=phases), bindings={"dock": dock, "tucked": tucked}
        ).audit(frames, max_samples=max_samples)
    else:
        report["semantic_plan_audit"] = audit_cup_plan(scene, frames, dock, tucked)
    report["supplied_plan_satisfied"] = (
        report["semantic_plan_audit"] is None
        or report["semantic_plan_audit"]["satisfied"]
    )
    report["satisfied"] = bool(
        report["satisfied"] and report["supplied_plan_satisfied"]
    )
    if not operations:
        report["minimum_operation_room_residual_m"] = None
    return report, rows


def region_holdout(scene, base, records, gripper, *, X_GC=None, seed=None):
    seed = int(scene.config["green_room"]["random_seed"] if seed is None else seed)
    rng = np.random.default_rng(seed)
    oracle = operation_oracle(scene, X_GC)
    collision_bad = semantic_bad = raw_bad = 0
    counterexamples = []
    failures = []
    minimum = float("inf")
    per_region = int(scene.config["green_room"]["random_region_samples"])
    if per_region < 1 or not records:
        raise ValueError(
            "Region audit requires nonempty regions and positive sample counts"
        )
    for region_index, record in enumerate(records):
        before = (collision_bad, semantic_bad, raw_bad)
        local_examples = []
        if "A" in record:
            from pydrake.common import RandomGenerator

            region = HPolyhedron(np.asarray(record["A"]), np.asarray(record["b"]))
            generator = RandomGenerator(int(rng.integers(1, 2**30)))
            sample = region.ChebyshevCenter()
            samples = []
            for _ in range(per_region):
                sample = region.UniformSample(generator, sample, mixing_steps=20)
                samples.append(sample)
        else:
            samples = rng.uniform(
                record["lower"], record["upper"], size=(per_region, 7)
            )
        for arm in samples:
            inside = oracle.query(np.r_[base, arm], gripper)
            pose = (
                scene.cup_initial_pose
                if X_GC is None
                else scene.plant.EvalBodyPoseInWorld(
                    scene.plant_context, scene.gripper_body
                )
                @ X_GC
            )
            scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, pose)
            collision = evaluate_collision_status(
                scene, base, arm_q=arm, gripper_q=gripper, influence_distance=0.2
            )
            collision_bad += int(collision.minimum_signed_distance < -1e-7)
            semantic_bad += int(not inside["satisfied"])
            minimum = min(minimum, inside["residual_m"])
            raw = summarize_grasp_audit(
                [
                    {
                        "attached": X_GC is not None,
                        "cup_gripper_audit": measure_cup_gripper(scene),
                    }
                ]
            )
            raw_bad += int(not raw["satisfied"])
            if (
                not raw["satisfied"]
                or not inside["satisfied"]
                or collision.minimum_signed_distance < -1e-7
            ):
                example = {
                    "region_index": region_index,
                    "arm_q": arm.tolist(),
                    "collision_distance_m": collision.minimum_signed_distance,
                    "whole_robot_room_residual_m": inside["residual_m"],
                    "raw_grasp_violation": raw["first_violation"],
                }
                if len(counterexamples) < 20:
                    counterexamples.append(example)
                if len(local_examples) < 5:
                    local_examples.append(example)
        counts = dict(
            zip(
                ("collisions", "semantic_violations", "raw_grasp_violations"),
                (
                    collision_bad - before[0],
                    semantic_bad - before[1],
                    raw_bad - before[2],
                ),
            )
        )
        if any(counts.values()):
            failures.append(
                {
                    "region_index": region_index,
                    **counts,
                    "counterexamples": local_examples,
                }
            )
    return {
        "samples": len(records) * per_region,
        "collisions": collision_bad,
        "raw_grasp_violations": raw_bad,
        "semantic_violations": semantic_bad,
        "counterexamples": counterexamples,
        "region_failures": failures,
        "seed": seed,
        "minimum_room_residual_m": minimum,
        "scope": "Independent holdout (hit-and-run for IRIS); counterexamples retained; not a safety certificate",
    }


def run(config_path, output, *, semantic=True, dynamics=True, visualize=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    config = load_scene_config(config_path)
    if not semantic:
        config["semantic_policy"] = "geometry_only"
    policy = policy_from_config(config)
    semantic = policy == "full_motion"

    def progress(stage):
        (output / "progress.json").write_text(json.dumps({"stage": stage}))

    progress("build_scene")
    (output / "scene_config_snapshot.json").write_text(json.dumps(config, indent=2))
    scene = build_scene(config=config, visualize=False)
    spec = config["green_room"]
    dock = np.asarray(config["pick_task"]["candidates"][0]["base_q"], float)
    start = np.asarray(config["robot"]["start"], float)
    tucked = np.asarray(config["pick_task"]["navigation_arm_q"], float)
    oracle = room_oracle(scene)
    start_value = oracle.query(np.r_[start, tucked])
    dock_value = oracle.query(np.r_[dock, tucked])
    progress("dock_semantics")
    if policy != "geometry_only" and not dock_value["satisfied"]:
        raise RuntimeError(
            f"Frozen docking configuration is outside room: {dock_value}"
        )
    print(f"Dock accepted under {policy}; solving grasp IK...", flush=True)
    progress("grasp_ik")
    pick = plan_pick(scene, dock)
    goal_checks = {
        name: oracle.query(np.r_[dock, arm])
        for name, arm in [
            ("pregrasp", pick.pregrasp_q),
            ("grasp", pick.grasp_q),
            ("lift", pick.lift_q),
        ]
    }
    (output / "goal_semantics.json").write_text(json.dumps(goal_checks, indent=2))
    if policy != "geometry_only" and not all(
        item["satisfied"] for item in goal_checks.values()
    ):
        raise RuntimeError(
            "IK goal violates whole-robot room semantics; try another candidate"
        )
    progress("operation_guides")
    transition = plan_pregrasp(
        scene,
        dock,
        tucked,
        pick.pregrasp_q,
        state_constraint=(lambda q: oracle.query(np.r_[dock, q])["satisfied"])
        if semantic
        else None,
    )
    approach = cartesian_arm_path(
        scene,
        dock,
        pick.pregrasp_q,
        pick.grasp_q,
        25,
        stage="green_room_approach",
        edge_validator=guide_edge_validator(
            scene,
            dock,
            pick.pregrasp_q,
            scene.gripper_open_positions,
            semantic=semantic,
        ),
    )
    reach_reference = np.vstack([transition, approach[1:]])
    # Use the continuation branch endpoint as the exact shared grasp state.
    grasp = reach_reference[-1]
    set_robot_configuration(scene, dock, grasp, scene.gripper_closed_positions)
    X_GC = (
        scene.plant.EvalBodyPoseInWorld(
            scene.plant_context, scene.gripper_body
        ).inverse()
        @ scene.cup_initial_pose
    )
    lift_reference = cartesian_arm_path(
        scene,
        dock,
        grasp,
        pick.lift_q,
        20,
        stage="green_room_lift",
        edge_validator=guide_edge_validator(
            scene,
            dock,
            grasp,
            scene.gripper_closed_positions,
            semantic=semantic,
            X_GC=X_GC,
        ),
    )
    print("Solving fixed-base contained reach and lift GCS...", flush=True)
    progress("operation_gcs")
    reach, reach_report, reach_dot = solve_operation(
        scene,
        dock,
        reach_reference,
        scene.gripper_open_positions,
        "reach_inside_room",
        semantic=semantic,
        artifacts=output / "reach_cover.json",
    )
    lift, lift_report, lift_dot = solve_operation(
        scene,
        dock,
        lift_reference,
        scene.gripper_closed_positions,
        "lift_inside_room",
        semantic=semantic,
        X_GC=X_GC,
        artifacts=output / "lift_cover.json",
    )
    from .accepted_motion import (
        accept_cup_phase,
        compose_cup_motion,
        check_cup_boundary,
    )

    repeat = lambda q, n: np.repeat(np.asarray(q)[None, :], n, axis=0)
    opened, closed = scene.gripper_open_positions, scene.gripper_closed_positions

    def segment(name, bases, arms, grips, dwell=0):
        local = []
        append_motion(local, bases, arms, grips, name)
        local.extend([dict(local[-1]) for _ in range(dwell)])
        return accept_cup_phase(
            scene,
            local,
            dock,
            tucked,
            X_GC,
            start=np.r_[bases[0], arms[0], grips[0]],
            end=np.r_[bases[-1], arms[-1], grips[-1]],
        )

    operations = [
        segment(
            "approach_cup", repeat(dock, len(reach)), reach, repeat(opened, len(reach))
        ),
        segment(
            "close_gripper",
            repeat(dock, 2),
            repeat(grasp, 2),
            np.array([opened, closed]),
            dwell=12,
        ),
        segment("lift_cup", repeat(dock, len(lift)), lift, repeat(closed, len(lift))),
        segment(
            "whole_body_carry_to_dropoff",
            repeat(dock, 2),
            repeat(lift[-1], 2),
            repeat(closed, 2),
            dwell=12,
        ),
    ]
    for left, right in zip(operations, operations[1:]):
        check_cup_boundary(left, right)
    scene.plant.SetFreeBodyPose(
        scene.plant_context, scene.cup_body, scene.cup_initial_pose
    )
    print("Solving tucked navigation GCS...", flush=True)
    progress("motion_bridge_gcs")
    nav, nav_report = plan_geometry_outbound(
        scene,
        start,
        dock,
        allowed_region_names=spec["navigation_regions"],
        arm_q=tucked,
        spacing=float(spec.get("navigation_spacing_m", 0.30)),
        prune_contained=bool(spec.get("navigation_prune_contained", False)),
        refinement_spacing=spec.get("navigation_refinement_spacing_m"),
        refinement_radius_threshold=spec.get(
            "navigation_refinement_radius_threshold_m"
        ),
    )
    nav_indices = sorted(
        set([*range(0, len(nav.configurations), 10), len(nav.configurations) - 1])
    )
    nav_path = nav.configurations[nav_indices]
    navigation = segment(
        "navigate_to_pickup",
        nav_path,
        repeat(tucked, len(nav_path)),
        repeat(opened, len(nav_path)),
    )
    frames, composition = compose_cup_motion(
        [navigation, *operations],
        start=np.r_[start, tucked, opened],
        end=np.r_[dock, lift[-1], closed],
    )
    print(f"Auditing {len(frames)} executable reference frames...", flush=True)
    progress("reference_audit")
    audit, trace = audit_reference(scene, frames, dock, tucked, X_GC)
    region_audits = {
        "reach": reach_report["region_audit"]
        if reach_report.get("region_audit") is not None
        else region_holdout(
            scene, dock, reach_report["boxes"], scene.gripper_open_positions
        ),
        "lift": lift_report["region_audit"]
        if lift_report.get("region_audit") is not None
        else region_holdout(
            scene, dock, lift_report["boxes"], scene.gripper_closed_positions, X_GC=X_GC
        ),
    }
    audit["policy_satisfied"] = accepts(
        policy,
        geometry=audit["geometry_satisfied"],
        goals=audit["goal_semantics_satisfied"],
        motion=audit["motion_semantics_satisfied"],
        plan=audit["supplied_plan_satisfied"],
    )
    report = {
        "task": "green_room_whole_robot_pick",
        "composition": composition,
        "phase_acceptance": {s.phase: s.report for s in [navigation, *operations]},
        "semantic_policy": policy,
        "semantic_enforced": semantic,
        "start_inside": start_value,
        "dock_inside": dock_value,
        "dock_q": dock.tolist(),
        "goal_semantics": goal_checks,
        "fixed_tucked_arm_q": tucked.tolist(),
        "navigation": nav_report,
        "reach": reach_report,
        "lift": lift_report,
        "reference_audit": audit,
        "region_holdout_audits": region_audits,
        "reference_frames": len(frames),
        "gcs_solve_calls": 3,
        "architecture": "Staged 2D navigation + 7D reach + 7D lift; exact configuration sharing; fixed yaw",
        "trajectory_representation": "Executable piecewise linear path; operation segments from order-1 GCS, navigation resampled from order-1 GCS and independently re-audited",
        "scope": "Guide-seeded corridors; supplied semantic sequence; cup separately checked; XY-prism room; no automatic missing-mode discovery",
        "planning_seconds": time.perf_counter() - began,
    }
    (output / "delivery_trace.json").write_text(json.dumps(frames))
    (output / "semantic_trace.json").write_text(json.dumps(trace))
    (output / "reach_graph.dot").write_text(reach_dot)
    (output / "lift_graph.dot").write_text(lift_dot)
    (output / "result.json").write_text(json.dumps(report, indent=2))
    if not audit["policy_satisfied"]:
        raise RuntimeError(f"Green-room reference failed independent audit: {audit}")
    if spec.get("require_clean_region_holdout", False) and any(
        r["collisions"]
        or r["raw_grasp_violations"]
        or (semantic and r["semantic_violations"])
        for r in region_audits.values()
    ):
        progress("region_holdout_rejected")
        raise RuntimeError(
            "Independent region holdout rejected this cover; no physical execution or box fallback"
        )
    if dynamics:
        progress("contact_dynamics")
        from .execute_delivery_dynamics import execute

        print("Reference passed; executing free-cup SAP simulation...", flush=True)
        physics, dyn_scene = execute(
            config,
            frames,
            time_step=0.001,
            time_scale=2.0,
            visualize=visualize,
            grip=True,
        )
        # The delivery evaluator expects return/release. This task instead
        # ends with a stationary lift/hold in the green room.
        dynamic_oracle = room_oracle(dyn_scene)
        active_values, cup_values, fixed_base_errors = [], [], []
        for row in physics["trace"]:
            if row["phase"] == "navigate_to_pickup":
                continue
            value = dynamic_oracle.query(
                np.r_[row["base_q"], row["arm_q"]], row["gripper_q"]
            )
            row["whole_robot_room"] = value
            active_values.append(value["residual_m"])
            cup_values.append(
                cup_room_residual(
                    dyn_scene,
                    dynamic_oracle,
                    RigidTransform(np.asarray(row["cup_pose"])),
                )
            )
            fixed_base_errors.append(
                float(np.linalg.norm(np.asarray(row["base_q"])[:2] - dock[:2]))
            )
        task_audit = {
            "minimum_operation_room_residual_m": min(active_values),
            "minimum_cup_room_residual_m": min(cup_values),
            "maximum_operation_base_position_error_m": max(fixed_base_errors),
            "bilateral_lift": physics["lifted_with_bilateral_contact"],
            "hold_bilateral_fraction": physics["held_bilateral_contact_fraction"],
            "unexpected_contacts": physics["unexpected_contact_count"],
            "effort_limits_satisfied": physics["actuator_effort_limits_satisfied"],
            "slip_satisfied": physics["carry_slip_satisfied"],
            "scope": "Room geometry re-audited at saved 50 ms actual-state samples; contacts at 10 ms; no continuous-time certificate",
        }
        task_audit["satisfied"] = bool(
            min(active_values) >= 0
            and min(cup_values) >= 0
            and max(fixed_base_errors) <= 0.01
            and task_audit["bilateral_lift"]
            and task_audit["hold_bilateral_fraction"] == 1.0
            and task_audit["unexpected_contacts"] == 0
            and task_audit["effort_limits_satisfied"]
            and task_audit["slip_satisfied"]
        )
        actual = [
            row for row in physics["trace"] if row["phase"] != "navigate_to_pickup"
        ]
        endpoints = [
            row
            for j, row in enumerate(actual)
            if j == 0
            or j == len(actual) - 1
            or actual[j - 1]["phase"] != row["phase"]
            or actual[min(j + 1, len(actual) - 1)]["phase"] != row["phase"]
        ]
        task_audit["geometry_satisfied"] = bool(
            max(fixed_base_errors) <= 0.01
            and task_audit["bilateral_lift"]
            and task_audit["hold_bilateral_fraction"] == 1.0
            and task_audit["unexpected_contacts"] == 0
            and task_audit["effort_limits_satisfied"]
            and task_audit["slip_satisfied"]
        )
        task_audit["goal_semantics_satisfied"] = all(
            row["whole_robot_room"]["satisfied"] for row in endpoints
        )
        task_audit["motion_semantics_satisfied"] = bool(
            min(active_values) >= 0 and min(cup_values) >= 0
        )
        from .semantic_plan import audit_cup_plan

        task_audit["semantic_plan_audit"] = audit_cup_plan(
            dyn_scene, physics["trace"], dock, tucked, execution=True
        )
        task_audit["supplied_plan_satisfied"] = (
            task_audit["semantic_plan_audit"] is None
            or task_audit["semantic_plan_audit"]["satisfied"]
        )
        task_audit["satisfied"] = bool(
            task_audit["satisfied"] and task_audit["supplied_plan_satisfied"]
        )
        task_audit["policy_satisfied"] = accepts(
            policy,
            geometry=task_audit["geometry_satisfied"],
            goals=task_audit["goal_semantics_satisfied"],
            motion=task_audit["motion_semantics_satisfied"],
            plan=task_audit["supplied_plan_satisfied"],
        )
        (output / "dynamics_trace.json").write_text(json.dumps(physics.pop("trace")))
        physics["generic_delivery_audit_applicable"] = False
        physics["generic_delivery_satisfied"] = physics.pop("satisfied")
        physics["satisfied"] = task_audit["satisfied"]
        physics["green_room_task_audit"] = task_audit
        (output / "physics.json").write_text(json.dumps(physics, indent=2))
        report["physics"] = task_audit
        if visualize:
            (output / "scene.html").write_text(dyn_scene.meshcat.StaticHtml())
    report["semantic_task_satisfied"] = bool(
        audit["satisfied"] and (not dynamics or report["physics"]["satisfied"])
    )
    report["satisfied"] = bool(
        audit["policy_satisfied"]
        and (not dynamics or report["physics"]["policy_satisfied"])
    )
    report["reference_validated"] = bool(audit["policy_satisfied"])
    report["physical_execution_validated"] = bool(
        dynamics and report["physics"]["policy_satisfied"]
    )
    report["dynamics_executed"] = dynamics
    report["total_seconds"] = time.perf_counter() - began
    report["source_sha256"] = {
        str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [
            Path(__file__),
            Path(__file__).with_name("whole_robot_room.py"),
            Path(config_path).resolve(),
        ]
    }
    (output / "result.json").write_text(json.dumps(report, indent=2))
    progress("complete" if report["satisfied"] else "execution_rejected")
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "satisfied",
                    "reference_frames",
                    "planning_seconds",
                    "dynamics_executed",
                ]
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["satisfied"]:
        raise RuntimeError("Green-room execution failed task audit; artifacts retained")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scene_green_room.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--no-dynamics", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--no-semantic", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        args.output,
        semantic=not args.no_semantic,
        dynamics=not args.no_dynamics,
        visualize=not args.no_visualize,
    )


if __name__ == "__main__":
    main()
