"""Graph-of-Convex-Sets planning for a planar mobile base in SE(2)."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from pydrake.geometry.optimization import HPolyhedron, Point
from pydrake.planning import GcsTrajectoryOptimization


@dataclass(frozen=True)
class GcsBasePlan:
    configurations: np.ndarray
    solve_time_seconds: float
    trajectory_duration: float
    path_length: float
    angular_travel: float
    optimal_cost: float
    solver_name: str
    number_of_segments: int
    region_sequence: list[str]


def _region_sets(
    planner_spec: dict[str, Any],
) -> tuple[list[HPolyhedron], list[dict[str, Any]]]:
    region_specs = list(planner_spec["regions"])
    regions = [
        HPolyhedron.MakeBox(
            np.asarray([*spec["lower"], spec["yaw_bounds"][0]], dtype=float),
            np.asarray([*spec["upper"], spec["yaw_bounds"][1]], dtype=float),
        )
        for spec in region_specs
    ]
    return regions, region_specs


def _label_region_sequence(
    q_path: np.ndarray,
    region_specs: list[dict[str, Any]],
) -> list[str]:
    sequence: list[str] = []
    current: str | None = None
    for q in q_path:
        containing = [
            spec["name"]
            for spec in region_specs
            if np.all(q[:2] >= np.asarray(spec["lower"], dtype=float) - 1e-7)
            and np.all(q[:2] <= np.asarray(spec["upper"], dtype=float) + 1e-7)
            and float(spec["yaw_bounds"][0]) - 1e-7 <= q[2]
            and q[2] <= float(spec["yaw_bounds"][1]) + 1e-7
        ]
        if current in containing:
            label = current
        else:
            label = containing[0] if containing else "transition"
        if label != current:
            sequence.append(label)
            current = label
    return sequence


def plan_base_gcs(
    config: dict[str, Any],
    start_q: np.ndarray | list[float],
    goal_q: np.ndarray | list[float],
    *,
    allowed_region_names: list[str] | None = None,
) -> GcsBasePlan:
    """Solve and sample a 3-D GCS path for ``q=(x, y, yaw)``."""

    start_q = np.asarray(start_q, dtype=float)
    goal_q = np.asarray(goal_q, dtype=float)
    if start_q.shape != (3,) or goal_q.shape != (3,):
        raise ValueError("Expected planar configurations q=(x, y, yaw).")

    planner_spec = config["gcs_base_planning"]
    regions, region_specs = _region_sets(planner_spec)
    if allowed_region_names is not None:
        allowed = set(allowed_region_names)
        selected = [
            (region, spec)
            for region, spec in zip(regions, region_specs, strict=True)
            if spec["name"] in allowed
        ]
        if not selected:
            raise ValueError("The semantic plan selected no GCS regions.")
        regions = [region for region, _ in selected]
        region_specs = [spec for _, spec in selected]
    for label, q in (("start", start_q), ("goal", goal_q)):
        if not any(region.PointInSet(q) for region in regions):
            raise ValueError(
                f"{label} configuration {q.tolist()} is outside all GCS regions."
            )

    gcs = GcsTrajectoryOptimization(3, continuous_revolute_joints=[2])
    free_space = gcs.AddRegions(
        regions,
        order=int(planner_spec["bezier_order"]),
        h_min=0.05,
        h_max=20.0,
        name="base_free_space",
    )
    source = gcs.AddRegions(
        [Point(start_q)],
        order=0,
        name="start",
    )
    target = gcs.AddRegions(
        [Point(goal_q)],
        order=0,
        name="goal",
    )
    gcs.AddEdges(source, free_space)
    gcs.AddEdges(free_space, target)
    position_weight = float(planner_spec["path_length_weight"])
    yaw_weight = float(planner_spec["yaw_path_weight"])
    gcs.AddPathLengthCost(
        np.diag([position_weight, position_weight, yaw_weight])
    )
    gcs.AddTimeCost(float(planner_spec["time_weight"]))
    maximum_speed = float(planner_spec["maximum_speed"])
    maximum_angular_speed = float(planner_spec["maximum_angular_speed"])
    gcs.AddVelocityBounds(
        np.array([-maximum_speed, -maximum_speed, -maximum_angular_speed]),
        np.array([maximum_speed, maximum_speed, maximum_angular_speed]),
    )

    solve_start = time.perf_counter()
    trajectory, result = gcs.SolvePath(source, target)
    solve_time = time.perf_counter() - solve_start
    if not result.is_success():
        raise RuntimeError(
            "GCS base planning failed with "
            f"{result.get_solution_result()} using {result.get_solver_id().name()}."
        )

    duration = float(trajectory.end_time() - trajectory.start_time())
    sample_count = max(
        2,
        int(np.ceil(duration * float(planner_spec["samples_per_second"]))) + 1,
    )
    sample_times = np.linspace(
        trajectory.start_time(),
        trajectory.end_time(),
        sample_count,
    )
    configurations = np.vstack(
        [np.asarray(trajectory.value(t), dtype=float).reshape(-1) for t in sample_times]
    )
    path_length = float(
        np.linalg.norm(np.diff(configurations[:, :2], axis=0), axis=1).sum()
    )
    angular_travel = float(np.abs(np.diff(configurations[:, 2])).sum())

    return GcsBasePlan(
        configurations=configurations,
        solve_time_seconds=solve_time,
        trajectory_duration=duration,
        path_length=path_length,
        angular_travel=angular_travel,
        optimal_cost=float(result.get_optimal_cost()),
        solver_name=result.get_solver_id().name(),
        number_of_segments=int(trajectory.get_number_of_segments()),
        region_sequence=_label_region_sequence(configurations, region_specs),
    )
