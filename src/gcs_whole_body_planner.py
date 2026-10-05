"""Whole-body GCS planning for the cup-carrying transit mode.

The state is ``q=(x, y, yaw, q_arm[0:7])``.  Each whole-body convex set is
the Cartesian product of an SE(2) base region and a joint-space box.  The
pickup and drop-off sets permit arm reconfiguration, while the doorway and
central transit sets keep the arm close to a compact carry posture.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from pydrake.geometry.optimization import HPolyhedron, Point
from pydrake.planning import GcsTrajectoryOptimization


@dataclass(frozen=True)
class WholeBodyPlan:
    configurations: np.ndarray
    solve_time_seconds: float
    trajectory_duration: float
    base_path_length: float
    angular_travel: float
    arm_joint_travel: float
    optimal_cost: float
    solver_name: str
    number_of_segments: int
    region_sequence: list[str]
    dimension: int = 10

    @property
    def base_configurations(self) -> np.ndarray:
        return self.configurations[:, :3]

    @property
    def arm_configurations(self) -> np.ndarray:
        return self.configurations[:, 3:]


def plan_to_dict(plan: WholeBodyPlan, *, include_configurations: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "dimension": plan.dimension,
        "solve_time_seconds": plan.solve_time_seconds,
        "trajectory_duration": plan.trajectory_duration,
        "base_path_length": plan.base_path_length,
        "angular_travel": plan.angular_travel,
        "arm_joint_travel": plan.arm_joint_travel,
        "optimal_cost": plan.optimal_cost,
        "solver": plan.solver_name,
        "number_of_segments": plan.number_of_segments,
        "region_sequence": plan.region_sequence,
    }
    if include_configurations:
        result["configurations"] = plan.configurations.tolist()
    return result


def _selected_region_specs(
    config: dict[str, Any],
    allowed_region_names: list[str],
) -> list[dict[str, Any]]:
    allowed = set(allowed_region_names)
    specs = [
        spec
        for spec in config["gcs_base_planning"]["regions"]
        if spec["name"] in allowed
    ]
    if set(spec["name"] for spec in specs) != allowed:
        missing = allowed - set(spec["name"] for spec in specs)
        raise ValueError(f"Unknown whole-body regions: {sorted(missing)}")
    return specs


def _arm_bounds(
    region_name: str,
    planner_spec: dict[str, Any],
    anchors: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    anchor_names = list(planner_spec["region_arm_anchors"][region_name])
    values = np.vstack([anchors[name] for name in anchor_names])
    margin_key = "transit_arm_margin" if len(anchor_names) == 1 else "arm_envelope_margin"
    margin = float(planner_spec[margin_key])
    lower = np.min(values, axis=0) - margin
    upper = np.max(values, axis=0) + margin
    joint_lower = np.asarray(planner_spec["joint_lower_limits"], dtype=float)
    joint_upper = np.asarray(planner_spec["joint_upper_limits"], dtype=float)
    return np.maximum(lower, joint_lower), np.minimum(upper, joint_upper)


def _whole_body_regions(
    config: dict[str, Any],
    allowed_region_names: list[str],
    anchors: dict[str, np.ndarray],
) -> tuple[list[HPolyhedron], list[dict[str, Any]]]:
    planner_spec = config["whole_body_planning"]
    region_specs = _selected_region_specs(config, allowed_region_names)
    regions: list[HPolyhedron] = []
    for spec in region_specs:
        arm_lower, arm_upper = _arm_bounds(
            str(spec["name"]),
            planner_spec,
            anchors,
        )
        lower = np.concatenate(
            [
                np.asarray(spec["lower"], dtype=float),
                np.asarray([spec["yaw_bounds"][0]], dtype=float),
                arm_lower,
            ]
        )
        upper = np.concatenate(
            [
                np.asarray(spec["upper"], dtype=float),
                np.asarray([spec["yaw_bounds"][1]], dtype=float),
                arm_upper,
            ]
        )
        regions.append(HPolyhedron.MakeBox(lower, upper))
    return regions, region_specs


def _label_sequence(
    configurations: np.ndarray,
    regions: list[HPolyhedron],
    specs: list[dict[str, Any]],
) -> list[str]:
    result: list[str] = []
    current: str | None = None
    for q in configurations:
        containing = [
            str(spec["name"])
            for region, spec in zip(regions, specs, strict=True)
            if region.PointInSet(q, 1e-6)
        ]
        label = current if current in containing else (containing[0] if containing else "transition")
        if label != current:
            result.append(label)
            current = label
    return result


def plan_carry_whole_body_gcs(
    config: dict[str, Any],
    *,
    start_base_q: np.ndarray | list[float],
    start_arm_q: np.ndarray | list[float],
    carry_arm_q: np.ndarray | list[float],
    goal_base_q: np.ndarray | list[float],
    goal_arm_q: np.ndarray | list[float],
    allowed_region_names: list[str],
    region_overrides: dict[str, HPolyhedron | list[HPolyhedron]] | None = None,
    gcs_options=None,
) -> WholeBodyPlan:
    """Plan retraction, transit, and pre-placement in one 10-D GCS."""

    start_base_q = np.asarray(start_base_q, dtype=float)
    goal_base_q = np.asarray(goal_base_q, dtype=float)
    anchors = {
        "start": np.asarray(start_arm_q, dtype=float),
        "carry": np.asarray(carry_arm_q, dtype=float),
        "goal": np.asarray(goal_arm_q, dtype=float),
    }
    if start_base_q.shape != (3,) or goal_base_q.shape != (3,):
        raise ValueError("Base configurations must have shape (3,).")
    if any(value.shape != (7,) for value in anchors.values()):
        raise ValueError("Arm configurations must have shape (7,).")

    start_q = np.concatenate([start_base_q, anchors["start"]])
    goal_q = np.concatenate([goal_base_q, anchors["goal"]])
    regions, region_specs = _whole_body_regions(
        config,
        allowed_region_names,
        anchors,
    )
    if region_overrides:
        expanded_regions: list[HPolyhedron] = []
        expanded_specs: list[dict[str, Any]] = []
        for region, spec in zip(regions, region_specs, strict=True):
            override = region_overrides.get(str(spec["name"]))
            replacements = (
                [region]
                if override is None
                else (override if isinstance(override, list) else [override])
            )
            if not replacements:
                raise ValueError(f"Region override for {spec['name']} is empty.")
            for replacement in replacements:
                if replacement.ambient_dimension() != 10:
                    raise ValueError("Whole-body region overrides must be 10-D.")
                expanded_regions.append(replacement)
                expanded_specs.append(spec)
        regions = expanded_regions
        region_specs = expanded_specs
    for label, q in (("start", start_q), ("goal", goal_q)):
        if not any(region.PointInSet(q) for region in regions):
            raise ValueError(f"The whole-body {label} is outside all selected regions.")

    planner_spec = config["whole_body_planning"]
    base_spec = config["gcs_base_planning"]
    gcs = GcsTrajectoryOptimization(10, continuous_revolute_joints=[2])
    free_space = gcs.AddRegions(
        regions,
        order=int(base_spec["bezier_order"]),
        h_min=0.05,
        h_max=30.0,
        name="whole_body_carry_free_space",
    )
    source = gcs.AddRegions([Point(start_q)], order=0, name="carry_start")
    target = gcs.AddRegions([Point(goal_q)], order=0, name="preplace_goal")
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
    trajectory, result = (gcs.SolvePath(source, target) if gcs_options is None
                          else gcs.SolvePath(source, target, gcs_options))
    solve_time = time.perf_counter() - solve_start
    if not result.is_success():
        error = RuntimeError(
            "Whole-body GCS failed with "
            f"{result.get_solution_result()} using {result.get_solver_id().name()}."
        )
        error.details = {
            "start": start_q.tolist(), "goal": goal_q.tolist(),
            "regions": [{"name": str(spec["name"]),
                         "start_max_violation": float(np.max(region.A() @ start_q - region.b())),
                         "goal_max_violation": float(np.max(region.A() @ goal_q - region.b()))}
                        for region, spec in zip(regions, region_specs)],
            "intersections": [[i, j] for i in range(len(regions))
                              for j in range(i + 1, len(regions))
                              if regions[i].IntersectsWith(regions[j])],
        }
        raise error

    duration = float(trajectory.end_time() - trajectory.start_time())
    sample_count = max(
        2,
        int(np.ceil(duration * float(base_spec["samples_per_second"]))) + 1,
    )
    sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(), sample_count)
    configurations = np.vstack(
        [np.asarray(trajectory.value(t), dtype=float).reshape(-1) for t in sample_times]
    )
    differences = np.diff(configurations, axis=0)
    return WholeBodyPlan(
        configurations=configurations,
        solve_time_seconds=solve_time,
        trajectory_duration=duration,
        base_path_length=float(np.linalg.norm(differences[:, :2], axis=1).sum()),
        angular_travel=float(np.abs(differences[:, 2]).sum()),
        arm_joint_travel=float(np.linalg.norm(differences[:, 3:], axis=1).sum()),
        optimal_cost=float(result.get_optimal_cost()),
        solver_name=result.get_solver_id().name(),
        number_of_segments=int(trajectory.get_number_of_segments()),
        region_sequence=_label_sequence(configurations, regions, region_specs),
    )
