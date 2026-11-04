"""Matched Inside -> IRIS -> GCS example with two constraint interfaces."""

import copy
import hashlib
import json
import logging
from pathlib import Path
import time
import importlib.metadata
import numpy as np
from pydrake.common import Parallelism, RandomGenerator
from pydrake.geometry.optimization import (
    HPolyhedron,
    Hyperellipsoid,
    Point,
    GraphOfConvexSetsOptions,
)
from pydrake.planning import (
    IrisNp2,
    IrisNp2Options,
    IrisParameterizationFunction,
    GcsTrajectoryOptimization,
)
from pydrake.solvers import IpoptSolver, ClarabelSolver

from .scene_spec import PROJECT_ROOT, load_scene_config
from .build_scene import build_scene
from .green_room_iris import make_checker
from .semantic_compiler import compile_predicates
from .whole_robot_room import WholeRobotRoomOracle


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def interpolate(points, step=0.003):
    result = [np.asarray(points[0])]
    for a, b in zip(points, points[1:]):
        n = max(1, int(np.ceil(np.max(abs(b - a)) / step)))
        result.extend(a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:])
    return np.asarray(result)


class WarningCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def check_points(compiled, oracle, checker, mapping, points):
    largest = -np.inf
    independent = -np.inf
    bad = []
    for i, q in enumerate(points):
        g = float(np.max(compiled.g(q)))
        raw = -float(
            oracle.query(np.r_[compiled.base, q], compiled.gripper)["residual_m"]
        )
        if abs(g - raw) > 1e-8:
            raise RuntimeError("Compiled and independent FK evaluators disagree")
        free = checker.CheckConfigCollisionFree(mapping(q))
        largest = max(largest, g)
        independent = max(independent, raw)
        if (g > 1e-7 or not free) and len(bad) < 8:
            bad.append(dict(index=i, g_max_m=g, collision_free=free, q=q.tolist()))
    return dict(
        satisfied=not bad,
        samples=len(points),
        maximum_g_m=largest,
        independent_maximum_g_m=independent,
        counterexamples=bad,
    )


def run(problem_path, output, *, methods=("conservative", "nonlinear")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    problem = json.loads(Path(problem_path).read_text())
    cfg = load_scene_config(PROJECT_ROOT / problem["scene"])
    start, goal = np.array(problem["start_arm_q"]), np.array(problem["goal_arm_q"])
    guide = np.array(
        [
            (1 - t) * start + t * goal
            for t in np.linspace(0, 1, problem["guide_intervals"] + 1)
        ]
    )
    guide[:, 1] -= problem["guide_shoulder_retraction_rad"] * np.sin(
        np.linspace(0, np.pi, len(guide))
    )
    shared = dict(
        problem=problem,
        guide=guide.tolist(),
        seed=20261104,
        particles=200,
        iris_iterations=3,
        separating_iterations=30,
        configuration_space_margin_rad=1e-5,
        holdout_samples_per_region=128,
        holdout_mixing_steps=20,
        gcs=dict(
            order=1,
            velocity_rad_s=0.5,
            length_weight=1.0,
            time_weight=0.05,
            rounded_paths=5,
            solver="Clarabel",
            max_iter=500,
            tolerance=1e-6,
            normalization="q = origin + scale * u; scale = guide range + 0.12 rad",
        ),
        drake_version=importlib.metadata.version("drake"),
        source_sha256={
            name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
            for name in [
                "src/semantic_dsl.py",
                "src/semantic_compiler.py",
                "src/inside_iris_comparison.py",
                problem["scene"],
            ]
        },
        scope="Fixed-base seven-joint reference planning; no contact execution or task-order comparison",
    )
    shared["input_sha256"] = hashlib.sha256(
        json.dumps(shared, sort_keys=True).encode()
    ).hexdigest()
    save(output / "protocol.json", shared)
    results = []
    for method in methods:
        folder = output / method
        folder.mkdir()
        began = time.monotonic()
        try:
            result = solve_method(cfg, problem, guide, shared, folder, method, began)
        except Exception as exc:
            save(
                folder / "failure.json",
                dict(error=str(exc), wall_seconds=time.monotonic() - began),
            )
            raise
        results.append(result)
        save(output / "results.json", results)
    return results


def solve_method(cfg, problem, guide, protocol, folder, method, began):
    if method not in ("conservative", "nonlinear"):
        raise ValueError("Unknown constraint interface")
    scene, checker, mapping = make_checker(
        copy.deepcopy(cfg), problem["base_q"], guide[0], [-0.055, 0.055]
    )
    compiled = compile_predicates(
        problem["dsl"],
        scene,
        base=problem["base_q"],
        gripper=scene.gripper_open_positions,
    )
    fresh = build_scene(config=copy.deepcopy(cfg), visualize=False)
    region = compiled.regions[compiled.predicates[0].region]
    oracle = WholeRobotRoomOracle(
        fresh, region.normals, region.offsets, margin=compiled.predicates[0].margin_m
    )
    save(folder / "compiled.json", compiled.metadata())
    guide_check = check_points(compiled, oracle, checker, mapping, interpolate(guide))
    if not guide_check["satisfied"]:
        raise RuntimeError("Shared guide is not feasible")
    direct_line = check_points(
        compiled, oracle, checker, mapping, interpolate([guide[0], guide[-1]])
    )
    save(folder / "straight_line_control.json", direct_line)
    if direct_line["satisfied"]:
        raise RuntimeError("Diagnostic case does not activate Inside")
    records = []
    regions = []
    attempts = []
    bound_calls = 0
    limits = cfg["whole_body_planning"]
    jlo = np.array(limits["joint_lower_limits"])
    jhi = np.array(limits["joint_upper_limits"])
    program = compiled.to_program() if method == "nonlinear" else None
    native_iris_seconds = 0.0
    audit_seconds = 0.0

    def grow(a, b, depth=0):
        nonlocal bound_calls, native_iris_seconds, audit_seconds
        if time.monotonic() - began > 1200 or len(records) >= 128 or depth > 10:
            raise RuntimeError("Local cover budget exhausted")
        padding = float(problem["padding_rad"]) * (0.65**depth)
        if method == "conservative":

            def upper(pad):
                nonlocal bound_calls
                bound_calls += 1
                return float(
                    np.max(
                        compiled.box_upper_bound(
                            np.maximum(np.minimum(a, b) - pad, jlo),
                            np.minimum(np.maximum(a, b) + pad, jhi),
                        )
                    )
                )

            if upper(0) > -1e-6:
                middle = (a + b) / 2
                grow(a, middle, depth + 1)
                grow(middle, b, depth + 1)
                return
            low, high = 0.0, padding
            for _ in range(35):
                mid = (low + high) / 2
                if upper(mid) <= -1e-6:
                    low = mid
                else:
                    high = mid
            padding = low
        lo = np.maximum(np.minimum(a, b) - padding, jlo)
        hi = np.minimum(np.maximum(a, b) + padding, jhi)
        seed = (lo + hi) / 2
        domain = HPolyhedron.MakeBox(lo, hi)
        if (
            not checker.CheckConfigCollisionFree(mapping(seed))
            or np.max(compiled.g(seed)) >= -1e-6
        ):
            mid = (a + b) / 2
            grow(a, mid, depth + 1)
            grow(mid, b, depth + 1)
            return
        options = IrisNp2Options()
        options.parameterization = IrisParameterizationFunction(mapping, 7)
        options.solver = IpoptSolver()
        options.ray_sampler_options.num_particles_to_walk_towards = 100
        s = options.sampled_iris_options
        s.num_particles = 200
        s.max_iterations = 3
        s.max_iterations_separating_planes = 30
        s.max_separating_planes_per_iteration = 20
        s.parallelism = Parallelism(1)
        s.sample_particles_in_parallel = False
        s.configuration_space_margin = 1e-5
        s.mixing_steps = 20
        s.random_seed = 20261104 + len(attempts)
        s.delta = 0.01 / 128
        s.epsilon = 0.01
        s.containment_points = np.column_stack([a, seed, b])
        if program is not None:
            s.prog_with_additional_constraints = program
        row = dict(
            method=method,
            lower=lo.tolist(),
            upper=hi.tolist(),
            seed=seed.tolist(),
            guide_start=a.tolist(),
            guide_end=b.tolist(),
            padding_rad=padding,
            extra_nonlinear_program=program is not None,
            status="running",
        )
        attempts.append(row)
        save(folder / "attempts.json", attempts)
        warnings = WarningCapture()
        logger = logging.getLogger("drake")
        logger.addHandler(warnings)
        tick = time.monotonic()
        try:
            poly = IrisNp2(
                checker,
                Hyperellipsoid.MakeHypersphere(
                    min(0.0001, float(np.min(hi - lo) / 8)), seed
                ),
                domain,
                options,
            )
        except RuntimeError as exc:
            row.update(status="native_iris_rejected", error=str(exc))
            poly = None
        finally:
            logger.removeHandler(warnings)
            elapsed = time.monotonic() - tick
            native_iris_seconds += elapsed
            row.update(iris_seconds=elapsed, warnings=warnings.messages)
        invalid = poly is None or any("voids" in w for w in warnings.messages)
        if not invalid and (
            not poly.PointInSet(a, 1e-7) or not poly.PointInSet(b, 1e-7)
        ):
            invalid = True
            row["status"] = "guide_endpoint_rejected"
        if not invalid:
            tick = time.monotonic()
            rng = RandomGenerator(9200 + len(attempts))
            q = poly.UniformSample(rng, seed, mixing_steps=100)
            samples = []
            for _ in range(128):
                q = poly.UniformSample(rng, q, mixing_steps=20)
                samples.append(q)
            audit = check_points(compiled, oracle, checker, mapping, samples)
            audit_seconds += time.monotonic() - tick
            row["holdout"] = audit
            invalid = not audit["satisfied"]
        if invalid:
            row["status"] = "rejected"
            save(folder / "attempts.json", attempts)
            middle = (a + b) / 2
            grow(a, middle, depth + 1)
            grow(middle, b, depth + 1)
            return
        row.update(
            status="accepted",
            faces=len(poly.b()),
            A=poly.A().tolist(),
            b=poly.b().tolist(),
            domain_maximum_g_bound_m=float(np.max(compiled.box_upper_bound(lo, hi))),
        )
        regions.append(poly)
        records.append(copy.deepcopy(row))
        save(folder / "regions.json", records)
        save(folder / "attempts.json", attempts)
        print(
            method,
            "region",
            len(records),
            "faces",
            len(poly.b()),
            "padding",
            round(padding, 6),
            flush=True,
        )

    for a, b in zip(guide, guide[1:]):
        grow(a, b)
    prepare = time.monotonic() - began
    points, duration, gcs_seconds, edge_count = solve_gcs(regions, guide, folder)
    dense = interpolate(points, step=0.001)
    audit = check_points(compiled, oracle, checker, mapping, dense)
    if not audit["satisfied"]:
        save(folder / "path_audit.json", audit)
        raise RuntimeError("Independent GCS path audit rejected")
    save(
        folder / "path.json",
        dict(
            waypoints=points.tolist(),
            dense=dense.tolist(),
            audit=audit,
            duration_s=duration,
        ),
    )
    answer = dict(
        method=method,
        input_sha256=protocol["input_sha256"],
        satisfied=True,
        regions=len(regions),
        directed_edges=edge_count,
        iris_calls=len(attempts),
        rejected_iris_regions=sum(r["status"] != "accepted" for r in attempts),
        native_iris_seconds=native_iris_seconds,
        region_audit_seconds=audit_seconds,
        prepare_seconds=prepare,
        gcs_seconds=gcs_seconds,
        total_seconds=time.monotonic() - began,
        domain_bound_calls=bound_calls,
        path_length_rad=float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
        path_audit=audit,
        straight_line_control=direct_line,
        scope="One diagnostic scene; independently sampled region/path checks; no continuous-time or global-optimality claim",
    )
    save(folder / "result.json", answer)
    return answer


def solve_gcs(regions, guide, folder):
    edges = [
        (i, j)
        for i, r in enumerate(regions)
        for j, t in enumerate(regions)
        if i != j and r.IntersectsWith(t)
    ]
    origin = (guide[0] + guide[-1]) / 2
    scales = np.ptp(guide, axis=0) + 0.12
    normalized = []
    for region in regions:
        A = region.A() * scales[None, :]
        b = region.b() - region.A() @ origin
        lengths = np.linalg.norm(A, axis=1)
        normalized.append(HPolyhedron(A / lengths[:, None], b / lengths))
    save(
        folder / "gcs_coordinates.json",
        dict(
            origin=origin.tolist(),
            scale=scales.tolist(),
            mapping="q = origin + scale * u; path metric and velocity bounds transformed exactly",
        ),
    )
    planner = GcsTrajectoryOptimization(7)
    cover = planner.AddRegions(normalized, edges, order=1, h_min=0.01, h_max=20)
    source = planner.AddRegions([Point((guide[0] - origin) / scales)], order=0)
    target = planner.AddRegions([Point((guide[-1] - origin) / scales)], order=0)
    planner.AddEdges(source, cover)
    planner.AddEdges(cover, target)
    planner.AddPathLengthCost(np.diag(scales))
    planner.AddTimeCost(0.05)
    planner.AddVelocityBounds(-0.5 / scales, 0.5 / scales)
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.max_rounded_paths = 5
    options.rounding_seed = 20261104
    options.solver = ClarabelSolver()
    options.restriction_solver = ClarabelSolver()
    from pydrake.solvers import SolverOptions

    settings = SolverOptions()
    for key, value in [
        ("max_iter", 500),
        ("tol_gap_abs", 1e-6),
        ("tol_gap_rel", 1e-6),
        ("tol_feas", 1e-6),
    ]:
        settings.SetOption(ClarabelSolver.id(), key, value)
    options.solver_options = settings
    options.restriction_solver_options = settings
    tick = time.monotonic()
    trajectory, result = planner.SolvePath(source, target, options)
    gcs_seconds = time.monotonic() - tick
    if not result.is_success():
        raise RuntimeError("GCS failed: " + str(result.get_solution_result()))
    points = []
    for i in range(trajectory.get_number_of_segments()):
        segment = trajectory.segment(i)
        a = origin + scales * segment.value(segment.start_time()).ravel()
        b = origin + scales * segment.value(segment.end_time()).ravel()
        if points and np.max(abs(points[-1] - a)) > 1e-6:
            raise RuntimeError("GCS path discontinuity")
        if not points:
            points.append(a)
        if np.max(abs(points[-1] - b)) > 1e-10:
            points.append(b)
    points = np.asarray(points)
    return (
        points,
        float(trajectory.end_time() - trajectory.start_time()),
        gcs_seconds,
        len(edges),
    )
