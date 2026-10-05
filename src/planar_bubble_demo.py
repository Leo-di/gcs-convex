"""A separate 2D finite-radius robot experiment: safe disks × task modes → Drake GCS.

Run: .venv/bin/python -m src.planar_bubble_demo
A fixed task sequence connects two mode-specific GCS solves at the pickup guard.
No manipulation physics is claimed. Every line segment is contained in a disk.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry.optimization import Hyperellipsoid, Point, GraphOfConvexSetsOptions
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver


LOWER = np.array([-4., -3.])
UPPER = np.array([3., 3.])
# Axis-aligned closed obstacles: xmin, ymin, xmax, ymax.
OBSTACLES = np.array([
    [-.10, -3., .10, -.35], [-.10, .35, .10, .9],
    [-.10, 2.4, .10, 3.],
    [.10, 2.5, 3., 3.], [.10, -3., 3., -2.5],
    [-1.8, -1.8, -1.1, -.8], [1., -1.8, 1.6, -.9],
])
START = np.array([-3., -.3])
PICKUP = np.array([2., -.1])
DROPOFF = np.array([-2.8, .1])
MARGIN = .06
BODY_RADII = {"empty": .12, "carrying": .42}
PASSAGES = {"narrow": [-.35, .35], "wide": [.9, 2.4]}


def clearance(point: np.ndarray, obstacles: np.ndarray = OBSTACLES) -> float:
    """Exact Euclidean distance to rectangles/boundary for a free point.

    Returns zero inside an obstacle; negative outside the bounding rectangle.
    """
    p = np.asarray(point)
    boundary = min(np.min(p - LOWER), np.min(UPPER - p))
    distances = np.linalg.norm(
        np.maximum(np.maximum(obstacles[:, :2] - p, p - obstacles[:, 2:]), 0),
        axis=1,
    )
    return float(min(boundary, distances.min()))


def make_cover(mode: str = "empty", *, obstacles: np.ndarray = OBSTACLES, body_radius: float | None = None) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    # Deterministic seed lattice; this is a partial cover, not a completeness claim.
    if body_radius is None:
        body_radius = BODY_RADII[mode]
    seeds = [START, PICKUP, DROPOFF]
    seeds += [np.array([x, y]) for x in np.arange(-3.5, 2.8, .55)
              for y in np.arange(-2.5, 2.6, .55)]
    seeds += [np.array([x, y]) for x in np.arange(-1., 1.01, .2) for y in [0., 1.65]]
    centers, radii = [], []
    for seed in seeds:
        r = min(.95, clearance(seed, obstacles) - body_radius - MARGIN)
        if r >= .08 and not any(np.linalg.norm(seed-c) < 1e-8 for c in centers):
            centers.append(seed)
            radii.append(r)
    centers, radii = np.array(centers), np.array(radii)
    edges = [(i, j) for i in range(len(centers)) for j in range(len(centers))
             if i != j and np.linalg.norm(centers[i]-centers[j]) < radii[i]+radii[j]-.015]
    return centers, radii, edges


def graph_connects(cover, start, goal) -> bool:
    centers, radii, edges = cover
    starts = {i for i, (c, r) in enumerate(zip(centers, radii)) if np.linalg.norm(start-c) <= r}
    goals = {i for i, (c, r) in enumerate(zip(centers, radii)) if np.linalg.norm(goal-c) <= r}
    adjacency = {i: [] for i in range(len(centers))}
    for i, j in edges:
        adjacency[i].append(j)
    seen, todo = set(starts), list(starts)
    while todo:
        for j in adjacency[todo.pop()]:
            if j not in seen:
                seen.add(j)
                todo.append(j)
    return bool(seen & goals)


def solve(case: str = "full") -> dict:
    if case not in {"full", "no_task", "ignore_payload", "blocked"}:
        raise ValueError(case)
    obstacles = OBSTACLES.copy()
    if case == "blocked":
        obstacles = np.vstack([obstacles, [-.10, .9, .10, 2.4]])
    planning_radii = dict(BODY_RADII)
    if case == "ignore_payload":
        planning_radii["carrying"] = BODY_RADII["empty"]
    covers = {mode: make_cover(mode, obstacles=obstacles, body_radius=radius)
              for mode, radius in planning_radii.items()}
    legs = [("empty", START, DROPOFF)] if case == "no_task" else [
        ("empty", START, PICKUP), ("carrying", PICKUP, DROPOFF)]
    reachable = {mode: graph_connects(covers[mode], start, goal) for mode, start, goal in legs}
    base = {
        "case": case, "body_radii": BODY_RADII, "planning_radii": planning_radii,
        "passages": PASSAGES, "bounds": [LOWER.tolist(), UPPER.tolist()],
        "obstacles": obstacles.tolist(), "start": START.tolist(),
        "pickup": PICKUP.tolist(), "dropoff": DROPOFF.tolist(),
        "covers": {mode: {"centers": c.tolist(), "radii": r.tolist(), "edges": e}
                   for mode, (c, r, e) in covers.items()},
        "graph_reachable": reachable,
    }
    if not all(reachable.values()):
        # Geometric certificate only for this explicit fully-spanning divider:
        # every continuous left/right path must cross x=0; the sole opening
        # has width .70, smaller than the actual loaded diameter .84.
        certificate = case == "blocked" and 2*BODY_RADII["carrying"] > .70
        return {**base, "status": "geometrically_infeasible" if certificate else "cover_disconnected",
                "certificate": "At x=0 only (-0.35,0.35) is open; loaded diameter 0.84 exceeds width 0.70; outer boundary prevents going around." if certificate else None,
                "segments": [], "trace": [], "crossings": {}, "metrics": None,
                "audit": {"task_order": None, "actual_body_safe": None, "accepted": False},
                "planning_method": "graph connectivity precheck; no GCS solve attempted"}

    disks = {mode: [Hyperellipsoid.MakeHypersphere(r, c) for c, r in zip(cover[0], cover[1])]
             for mode, cover in covers.items()}
    # The action location is fixed, so the two additive-cost motion problems
    # separate at PICKUP. Solve each mode's GCS and concatenate at that guard.
    # This is a decomposition at a fixed action guard, not
    # a joint optimization over action locations or alternative task orders.
    trajectories = {}
    build_seconds = 0.
    solve_seconds = 0.
    for mode, start, goal in legs:
        build_start = time.perf_counter()
        centers, radii, edges = covers[mode]
        planner = GcsTrajectoryOptimization(2)
        free = planner.AddRegions(disks[mode], edges, order=1, h_min=.01, h_max=20, name=mode)
        source = planner.AddRegions([Point(start)], order=0, name="start")
        target = planner.AddRegions([Point(goal)], order=0, name="goal")
        def containing(p):
            return [i for i, (c, r) in enumerate(zip(centers, radii)) if np.linalg.norm(p-c) <= r]
        planner.AddEdges(source, free, edges_between_regions=[(0, i) for i in containing(start)])
        planner.AddEdges(free, target, edges_between_regions=[(i, 0) for i in containing(goal)])
        planner.AddPathLengthCost(1.)
        planner.AddTimeCost(.01)
        planner.AddVelocityBounds(np.array([-1., -1.]), np.array([1., 1.]))
        build_seconds += time.perf_counter()-build_start
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 20
        options.rounding_seed = 0
        options.solver = ClarabelSolver()
        options.restriction_solver = ClarabelSolver()
        solve_start = time.perf_counter()
        trajectory, result = planner.SolvePath(source, target, options)
        solve_seconds += time.perf_counter()-solve_start
        if not result.is_success():
            raise RuntimeError(f"{mode}: {result.get_solution_result()}")
        trajectories[mode] = trajectory

    # Audit every segment endpoint against one *common* safe disk. For order 1,
    # convexity gives whole-segment containment, stronger than dense sampling alone.
    segments, trace = [], []
    picked = case != "no_task"
    for mode, trajectory in trajectories.items():
        for i in range(trajectory.get_number_of_segments()):
            segment = trajectory.segment(i)
            a = segment.value(segment.start_time()).ravel()
            b = segment.value(segment.end_time()).ravel()
            centers, radii, _ = covers[mode]
            candidates = [j for j, (c, r) in enumerate(zip(centers, radii))
                          if max(np.linalg.norm(a-c), np.linalg.norm(b-c)) <= r+1e-6]
            if not candidates:
                raise AssertionError("A segment lacks a common containing safe disk")
            segments.append({"a": a.tolist(), "b": b.tolist(), "mode": mode, "disk": candidates[0]})
            for u in np.linspace(0., 1., 21):
                p = (1-u)*a+u*b
                trace.append({"xy": p.tolist(), "mode": mode, "room": "room1" if p[0]<-.1 else "room2" if p[0]>.1 else "doorway"})
    empty_segments = [s for s in segments if s["mode"] == "empty"]
    carry_segments = [s for s in segments if s["mode"] == "carrying"]
    if picked:
        assert np.linalg.norm(np.array(empty_segments[-1]["b"])-PICKUP) < 1e-5
        assert np.linalg.norm(np.array(carry_segments[0]["a"])-PICKUP) < 1e-5
    assert np.linalg.norm(np.array(segments[-1]["b"])-DROPOFF)<1e-5
    assert np.linalg.norm(np.array(segments[0]["a"])-START)<1e-5
    assert all(np.linalg.norm(np.array(a["b"])-b["a"])<1e-5 for a,b in zip(segments,segments[1:]))
    min_clearance = min(clearance(np.array(row["xy"]), obstacles)-BODY_RADII[row["mode"]] for row in trace)
    if case != "ignore_payload":
        assert min_clearance >= MARGIN-2e-6
    crossings = {mode: [] for mode in BODY_RADII}
    for s in segments:
        a, b = np.array(s["a"]), np.array(s["b"])
        if (a[0] < 0 <= b[0]) or (b[0] < 0 <= a[0]):
            y = float(a[1]+(-a[0])/(b[0]-a[0])*(b[1]-a[1]))
            label = next((name for name, (lo, hi) in PASSAGES.items() if lo <= y <= hi), "wall")
            crossings[s["mode"]].append({"passage": label, "y": y})
    # Independent checks at all samples and exact wall crossings, using the
    # ACTUAL payload radius, not the deliberately wrong ablation radius.
    crossing_clearance = min((clearance(np.array([0., x["y"]]), obstacles)-BODY_RADII[mode]
                              for mode, values in crossings.items() for x in values), default=float("inf"))
    actual_safe = min(min_clearance, crossing_clearance) >= MARGIN-2e-6
    return {
        **base, "status": "solved", "crossings": crossings,
        "audit": {"task_order": picked, "actual_body_safe": bool(actual_safe),
                  "accepted": bool(picked and actual_safe),
                  "crossing_min_body_clearance": crossing_clearance if np.isfinite(crossing_clearance) else None},
        "planning_method": "direct empty-mode GCS (task ignored)" if case == "no_task" else "two mode-specific GCS solves separated at a fixed pickup guard",
        "segments": segments, "trace": trace,
        "metrics": {"disks_by_mode": {mode: len(v) for mode, v in disks.items()},
                    "mode_region_vertices": sum(len(v) for v in disks.values()),
                    "geometric_directed_edges": {mode: len(v[2]) for mode, v in covers.items()}, "build_seconds": build_seconds,
                    "solve_seconds": solve_seconds, "solver": result.get_solver_id().name(),
                    "length": sum(np.linalg.norm(np.array(s["b"])-s["a"]) for s in segments),
                    "sampled_min_body_clearance": min_clearance, "disk_margin": MARGIN,
                    "whole_segment_disk_audit": True, "pickup_before_dropoff": picked},
        "limitations": "2D circular robot with mode-dependent radius; fixed action points; abstract pickup/dropoff; partial disk cover; convex relaxation and rounding, no global optimality claim."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/results/grasp_fix_web/planar"))
    args = parser.parse_args()
    data = solve()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "result.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    template = Path(__file__).with_name("planar_bubble_view.html").read_text(encoding="utf-8")
    (args.output / "index.html").write_text(template.replace("__DATA__", json.dumps(data)), encoding="utf-8")
    print(json.dumps(data["metrics"], indent=2))
    print(args.output / "index.html")


if __name__ == "__main__":
    main()
