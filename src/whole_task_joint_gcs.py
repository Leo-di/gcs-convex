"""One task-wide GCS with shared, variable contact configurations.

Task order and placement goals define the mode graph. Motion modes share
robot states at variable grasp/place vertices with exact FK constraints.
Ten-dimensional IRIS regions cover free motion; near-support domains use
explicit collision constraints in the nonlinear restriction problems.
"""

from __future__ import annotations
import copy, json, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from pydrake.autodiffutils import (
    AutoDiffXd,
    InitializeAutoDiff,
    ExtractGradient,
    ExtractValue,
)
from pydrake.common import Parallelism
from pydrake.geometry.optimization import (
    HPolyhedron,
    Hyperellipsoid,
    Point,
    GraphOfConvexSets,
    GraphOfConvexSetsOptions,
)
from pydrake.math import RigidTransform
from pydrake.multibody.inverse_kinematics import MinimumDistanceLowerBoundConstraint
from pydrake.planning import (
    GcsTrajectoryOptimization,
    IrisNp2,
    IrisNp2Options,
    IrisParameterizationFunction,
)
from pydrake.solvers import (
    MathematicalProgram,
    Binding,
    Constraint,
    ClarabelSolver,
    IpoptSolver,
)
from .build_scene import set_robot_configuration
from .green_room_iris import make_checker
from .green_room_gcs import operation_oracle
from .iris_region_generator import _make_parameterization, _reduced_position_indices
from .joint_approach_gcs import motion_bound, interpolate_q10
from .room_entry import compress_collinear
from .stateful_transport_grounding import world_config
from .whole_robot_room import arm_point_reach_bound

RESTRICTION = {GraphOfConvexSets.Transcription.kRestriction}
Q = lambda f: np.r_[f["base_q"], f["arm_q"]]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False))


class RobotFK:
    def __init__(self, scene, gripper):
        self.scene = scene
        self.mapping = _make_parameterization(
            scene.plant.GetPositions(scene.plant_context).copy(),
            _reduced_position_indices(scene),
        )
        self.ad = scene.plant.ToAutoDiffXd()
        self.context = self.ad.CreateDefaultContext()
        self.body = self.ad.get_body(scene.gripper_body.index())

    def transform(self, q):
        self.scene.plant.SetPositions(
            self.scene.plant_context, self.mapping(np.asarray(q, float))
        )
        return self.scene.plant.EvalBodyPoseInWorld(
            self.scene.plant_context, self.scene.gripper_body
        )

    def error(self, q, target):
        q = np.asarray(q).ravel()
        if q.dtype == object:
            self.ad.SetPositions(self.context, self.mapping(q))
            X = self.ad.EvalBodyPoseInWorld(self.context, self.body)
        else:
            X = self.transform(q)
        R = target.rotation().matrix().T @ X.rotation().matrix()
        return np.r_[
            X.translation() - target.translation(),
            (R[2, 1] - R[1, 2]) / 2,
            (R[0, 2] - R[2, 0]) / 2,
            (R[1, 0] - R[0, 1]) / 2,
        ]

    def jacobian(self, q, target):
        return ExtractGradient(
            self.error(InitializeAutoDiff(np.asarray(q, float)).ravel(), target)
        )


@dataclass
class MotionMode:
    name: str
    action: str
    active: str
    held: bool
    guide: np.ndarray
    template: dict
    scene: object
    checker: object
    mapping: object
    oracle: object
    fk: RobotFK
    grip: np.ndarray
    X_GC: object
    supports: list
    regions: list
    records: list
    compiled_inside: object = None
    inside_interface: str | None = None


def make_modes(config, reference, actions):
    modes = []
    contacts = []
    previous_retreat = []
    for action in actions:
        aid = action["id"]
        rows = [f for f in reference if f["phase"].startswith(aid + ".")]
        free = [f for f in rows if f["mode"] in ("arrive", "reach")]
        close = next(f for f in reversed(rows) if f["mode"] == "close")
        held = [f for f in rows if f["mode"] in ("lift", "hold", "carry", "lower")]
        release = next(f for f in rows if f["mode"] == "release")
        retreat = [f for f in rows if f["mode"] == "retreat"]

        def create(name, frames, attached=False, X=None):
            first = frames[0]
            local = world_config(
                config,
                dict(poses=first["world_objects"], supports=first["world_supports"]),
                action["object_id"],
            )
            grip = np.asarray(close["gripper_q"] if attached else free[0]["gripper_q"])
            scene, checker, _ = make_checker(local, Q(first)[:3], Q(first)[3:], grip, X)
            margin = float(
                config.get("whole_task_planning", {}).get("base_tracking_margin_m", 0.0)
            )
            checker.SetPaddingOneRobotBodyAllEnvironmentPairs(
                scene.base_body.index(), margin
            )
            mapping = _make_parameterization(
                scene.plant.GetPositions(scene.plant_context).copy(),
                _reduced_position_indices(scene),
            )
            oracle = operation_oracle(scene, X)
            return MotionMode(
                name,
                aid,
                action["object_id"],
                attached,
                np.array([Q(f) for f in frames]),
                copy.deepcopy(first),
                scene,
                checker,
                mapping,
                oracle,
                RobotFK(scene, grip),
                grip,
                X,
                [action["source_support"], action["destination_support"]],
                [],
                [],
            )

        empty = create(aid + "_empty", previous_retreat + free)
        modes.append(empty)
        target = empty.fk.transform(Q(close))
        X_GC = target.inverse() @ RigidTransform(
            np.array(close["world_objects"][action["object_id"]])
        )
        contacts.append(
            dict(
                name=aid + "_grasp",
                action=aid,
                kind="close",
                q0=Q(close),
                target=target,
                fk=empty.fk,
            )
        )
        carrying = create(aid + "_held", held, True, X_GC)
        modes.append(carrying)
        contacts.append(
            dict(
                name=aid + "_place",
                action=aid,
                kind="release",
                q0=Q(held[-1]),
                target=carrying.fk.transform(Q(held[-1])),
                fk=carrying.fk,
            )
        )
        previous_retreat = retreat
    # Last retreat stays in the SAME task graph, after the last release.
    last = actions[-1]
    first = previous_retreat[0]
    local = world_config(
        config,
        dict(poses=first["world_objects"], supports=first["world_supports"]),
        last["object_id"],
    )
    grip = np.asarray(first["gripper_q"])
    scene, checker, _ = make_checker(local, Q(first)[:3], Q(first)[3:], grip)
    checker.SetPaddingOneRobotBodyAllEnvironmentPairs(
        scene.base_body.index(),
        float(config.get("whole_task_planning", {}).get("base_tracking_margin_m", 0.0)),
    )
    modes.append(
        MotionMode(
            "final_retreat",
            last["id"],
            last["object_id"],
            False,
            np.array([Q(f) for f in previous_retreat]),
            copy.deepcopy(first),
            scene,
            checker,
            _make_parameterization(
                scene.plant.GetPositions(scene.plant_context).copy(),
                _reduced_position_indices(scene),
            ),
            operation_oracle(scene),
            RobotFK(scene, grip),
            grip,
            None,
            [],
            [],
            [],
        )
    )
    return modes, contacts


def support_gap(mode, q):
    if not mode.held:
        return np.inf
    mode.scene.plant.SetPositions(mode.scene.plant_context, mode.mapping(q))
    s = mode.scene
    query = s.scene_graph.get_query_output_port().Eval(
        s.scene_graph.GetMyContextFromRoot(s.root_context)
    )
    gids = s.plant.GetCollisionGeometriesForBody(s.plant.GetBodyByName("attached_cup"))
    return min(
        float(query.ComputeSignedDistancePairClosestPoints(a, b).distance)
        for a in gids
        for name in mode.supports
        for b in s.plant.GetCollisionGeometriesForBody(s.plant.GetBodyByName(name))
    )


def spatial_query(mode, q):
    if hasattr(mode.oracle, "clear"):
        mode.oracle.clear.passage.reset_witness_cache()
    result = mode.oracle.query(q, mode.grip)
    if mode.compiled_inside is not None:
        # Inside is handled by the compiler. Preserve the separate access rule
        # here without also clipping both interfaces to the legacy room bound.
        result["residual_m"] = result.get("keep_clear", {}).get("minimum_residual_m")
        if result["residual_m"] is None:
            result["residual_m"] = float("inf")
    return result


def grow_mode_cover(mode, out, *, seed, max_seconds=1200, domain_policy="guide_local"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    rng = np.random.default_rng(seed)
    reach = arm_point_reach_bound(mode.scene, mode.oracle)
    spec = mode.scene.config["whole_body_planning"]
    limits_lo = np.r_[[-1.45, -2.3, -np.pi], spec["joint_lower_limits"]]
    limits_hi = np.r_[[5.25, 5.05, np.pi], spec["joint_upper_limits"]]
    tuning = mode.scene.config.get("whole_task_planning", {})
    padding = np.array(
        tuning.get("region_padding", [0.015, 0.015, 0.002] + [0.004] * 7)
    )
    inside = mode.compiled_inside

    def add(a, b, depth=0, scale=1.0):
        if time.monotonic() - began > max_seconds or len(mode.regions) > 800:
            raise RuntimeError("Whole-task mode cover budget exceeded")
        lo = np.maximum(np.minimum(a, b) - padding * scale, limits_lo)
        hi = np.minimum(np.maximum(a, b) + padding * scale, limits_hi)
        center = (lo + hi) / 2
        expansion = None
        if domain_policy == "geometry_cap":
            caps = np.r_[[0.10, 0.10, 0.04], [0.04] * 7]
            lo = np.maximum(np.minimum(a, b) - caps * scale, limits_lo)
            hi = np.minimum(np.maximum(a, b) + caps * scale, limits_hi)
            center = (lo + hi) / 2
            bound = 0.0
        elif domain_policy == "slack_adaptive":
            from .semantic_domain_expansion import expand_domain

            center = (a + b) / 2
            q = spatial_query(mode, center)
            expanded = expand_domain(
                a,
                b,
                limits_lo,
                limits_hi,
                float(q["residual_m"]),
                reach,
                cap_scale=scale,
            )
            if expanded is not None:
                lo, hi, expansion = expanded
            else:
                lo, hi = np.minimum(a, b), np.maximum(a, b)
            bound = -1.0 if expanded is None else expansion["certified_lower_bound_m"]
        else:
            q = spatial_query(mode, center)
            bound = float(q["residual_m"]) - motion_bound((hi - lo) / 2, reach)
        if domain_policy != "geometry_cap" and bound < 1e-6:
            if depth >= 14:
                raise RuntimeError("No spatially valid joint domain")
            mid = (a + b) / 2
            add(a, mid, depth + 1, scale * 0.6)
            add(mid, b, depth + 1, scale * 0.6)
            return
        inside_bound = None
        if inside is not None:
            inside_bound = float(np.max(inside.box_upper_bound(lo, hi)))
            if any(np.max(inside.g(q)) > 1e-7 for q in (a, b)):
                raise RuntimeError(
                    "Guide endpoint violates compiled Inside in " + mode.name
                )
            invalid_inside = (
                inside_bound > -1e-6
                if mode.inside_interface == "conservative"
                else np.max(inside.g(center)) >= -1e-6
            )
            if invalid_inside:
                if depth >= 14:
                    raise RuntimeError("No Inside-feasible joint input domain")
                mid = (a + b) / 2
                add(a, mid, depth + 1, scale * 0.6)
                add(mid, b, depth + 1, scale * 0.6)
                return
        domain = HPolyhedron.MakeBox(lo, hi)
        active_inside_rows = (
            inside.active_rows(lo, hi) if inside is not None else np.array([], int)
        )
        inside_program = (
            inside.to_program(active_inside_rows)
            if inside is not None and mode.inside_interface == "nonlinear"
            else None
        )
        from .base_clearance_domains import guard

        tracking_margin = float(
            mode.scene.config.get("whole_task_planning", {}).get(
                "base_tracking_margin_m", 0.0
            )
        )
        base_guard = guard(mode, lo, hi, a, b, tracking_margin)
        if base_guard is None:
            if depth >= 14:
                raise RuntimeError("No conservative base tracking-clearance domain")
            mid = (a + b) / 2
            add(a, mid, depth + 1, scale * 0.6)
            add(mid, b, depth + 1, scale * 0.6)
            return
        base_A, base_b, base_proof = base_guard
        domain = HPolyhedron(np.vstack([domain.A(), base_A]), np.r_[domain.b(), base_b])
        near = mode.held and min(support_gap(mode, a), support_gap(mode, b)) < 0.004
        start = time.monotonic()
        kind = "support_contact_patch" if near else "official_IRIS_NP2_10D"
        if near:
            region = domain
        else:
            if not mode.checker.CheckConfigCollisionFree(mode.mapping(center)):
                if depth >= 14:
                    raise RuntimeError(
                        "Colliding domain center after bounded refinement"
                    )
                mid = (a + b) / 2
                add(a, mid, depth + 1, scale * 0.6)
                add(mid, b, depth + 1, scale * 0.6)
                return
            options = IrisNp2Options()
            options.parameterization = IrisParameterizationFunction(mode.mapping, 10)
            options.solver = IpoptSolver()
            options.add_hyperplane_if_solve_fails = False
            options.ray_sampler_options.num_particles_to_walk_towards = 100
            p = options.sampled_iris_options
            p.num_particles = 200
            p.max_iterations = 3
            p.max_iterations_separating_planes = 15
            p.max_separating_planes_per_iteration = 10
            p.parallelism = Parallelism(1)
            p.sample_particles_in_parallel = False
            p.verbose = False
            p.delta = 0.01 / 12000
            p.epsilon = 0.01
            p.configuration_space_margin = 1e-6
            p.mixing_steps = 20
            p.random_seed = seed + len(mode.regions)
            p.containment_points = np.column_stack([a, center, b])
            if inside_program is not None:
                p.prog_with_additional_constraints = inside_program
            from .inside_iris_comparison import WarningCapture
            import logging

            capture = WarningCapture()
            logger = logging.getLogger("drake")
            logger.addHandler(capture)
            failure = None
            try:
                region = IrisNp2(
                    mode.checker,
                    Hyperellipsoid.MakeHypersphere(
                        min(0.0001, float(np.min(hi - lo) / 8)), center
                    ),
                    domain,
                    options,
                )
                if any("voids" in message for message in capture.messages):
                    failure = "IRIS probabilistic test was voided"
            except RuntimeError as exc:
                failure = str(exc)
            finally:
                logger.removeHandler(capture)
            if failure is not None:
                write(
                    out / f"native_rejected_{len(mode.regions)}_{depth}.json",
                    dict(
                        error=failure,
                        warnings=capture.messages,
                        lower=lo.tolist(),
                        upper=hi.tolist(),
                        seconds=time.monotonic() - start,
                    ),
                )
                if depth >= 14:
                    raise RuntimeError(
                        "IRIS failed after bounded refinement: " + failure
                    )
                mid = (a + b) / 2
                add(a, mid, depth + 1, scale * 0.5)
                add(mid, b, depth + 1, scale * 0.5)
                return
            if not region.PointInSet(a, 1e-7) or not region.PointInSet(b, 1e-7):
                raise RuntimeError("Global cover excludes source guide endpoint")
            tested = 0
            bad = []
            for _ in range(6000):
                sample = rng.uniform(lo, hi)
                if not region.PointInSet(sample, 1e-10):
                    continue
                tested += 1
                if not mode.checker.CheckConfigCollisionFree(mode.mapping(sample)):
                    bad.append(sample.tolist())
                elif inside is not None and np.max(inside.g(sample)) > 1e-7:
                    bad.append(sample.tolist())
                if tested == 32:
                    break
            if tested < 32:
                raise RuntimeError("Insufficient global region holdout samples")
            if bad:
                write(
                    out / f"rejected_{len(mode.regions)}_{depth}.json",
                    dict(counterexamples=bad, lower=lo.tolist(), upper=hi.tolist()),
                )
                if depth >= 14:
                    raise RuntimeError(
                        "Region holdout rejection after bounded refinement"
                    )
                mid = (a + b) / 2
                add(a, mid, depth + 1, scale * 0.5)
                add(mid, b, depth + 1, scale * 0.5)
                return
        record = dict(
            inside_interface=mode.inside_interface,
            inside_active_nonlinear_rows=active_inside_rows.tolist(),
            inside_certified_rows=(
                len(inside.labels) - len(active_inside_rows)
                if inside is not None
                else 0
            ),
            iris_warnings=[] if near else capture.messages,
            inside_maximum_domain_bound_m=inside_bound,
            inside_nonlinear_program_in_iris=inside_program is not None and not near,
            base_clearance_guard=base_proof,
            domain_policy=domain_policy,
            semantic_expansion=expansion,
            kind=kind,
            lower=lo.tolist(),
            upper=hi.tolist(),
            A=region.A().tolist(),
            b=region.b().tolist(),
            guide_start=a.tolist(),
            guide_end=b.tolist(),
            seed=center.tolist(),
            semantic_lower_bound_m=None if domain_policy == "geometry_cap" else bound,
            growth_seconds=time.monotonic() - start,
            contact_nonlinear_required=near,
        )
        mode.regions.append(region)
        mode.records.append(record)
        write(out / "regions.json", mode.records)
        print(
            mode.name,
            len(mode.regions),
            kind,
            round(record["growth_seconds"], 2),
            flush=True,
        )

    compact = compress_collinear(mode.guide, tolerance=1e-8)
    for a, b in zip(compact, compact[1:]):
        n = max(
            1,
            int(
                np.ceil(
                    max(
                        np.linalg.norm(b[:2] - a[:2])
                        / tuning.get("cover_translation_step_m", 0.22),
                        np.max(np.abs(b[2:] - a[2:]))
                        / tuning.get("cover_rotation_step_rad", 0.15),
                    )
                )
            ),
        )
        for i in range(n):
            add(a + (b - a) * i / n, a + (b - a) * (i + 1) / n)
    write(
        out / "complete.json",
        dict(
            base_tracking_margin_m=float(
                mode.scene.config.get("whole_task_planning", {}).get(
                    "base_tracking_margin_m", 0.0
                )
            ),
            domain_policy=domain_policy,
            regions=len(mode.regions),
            guide_sha256=__import__("hashlib").sha256(mode.guide.tobytes()).hexdigest(),
        ),
    )
    return mode


def contact_region(contact, radius=None):
    q = contact["q0"]
    fk = contact["fk"]
    J = fk.jacobian(q, contact["target"])
    rank = int(np.linalg.matrix_rank(J, tol=1e-6))
    if rank != 6:
        raise RuntimeError("Contact FK Jacobian must leave four local robot DOFs")
    radius = np.full(10, 0.012 if radius is None else float(radius))
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("Positive contact trust radius required")
    lo = q - radius
    hi = q + radius
    A = np.vstack([np.eye(10), -np.eye(10), J, -J])
    tol = np.r_[[0.0003] * 3, [0.0005] * 3]
    b = np.r_[hi, -lo, J @ q + tol, -J @ q + tol]
    return HPolyhedron(A, b), dict(
        name=contact["name"],
        constraint="exact gripper FK in global nonlinear restriction",
        jacobian_rank=rank,
        local_free_dimensions=10 - rank,
        robot_point_fixed=False,
        lower=lo.tolist(),
        upper=hi.tolist(),
        anchor=q.tolist(),
        target=contact["target"].GetAsMatrix4().tolist(),
    )


def callable_binding(fun, lower, upper, variables):
    factory = MathematicalProgram()
    x = factory.NewContinuousVariables(len(variables))
    binding = factory.AddConstraint(fun, np.asarray(lower), np.asarray(upper), x)
    return Binding[Constraint](binding.evaluator(), variables)


class PairwiseContactClearance:
    """Individual robot-body distances with native analytic Jacobians.

    A smooth aggregate minimum-distance predicate can reject multiple exact
    support contacts even when every pair is nonpenetrating. Preserve those
    physical zero-gap constraints individually instead of relaxing their bound.
    """

    def __init__(self, mode, record):
        self.mode = mode
        self.indices = _reduced_position_indices(mode.scene)
        center = (np.array(record["lower"]) + record["upper"]) / 2
        displacement = motion_bound(
            (np.array(record["upper"]) - record["lower"]) / 2,
            arm_point_reach_bound(mode.scene, mode.oracle),
        )
        self.influence = 4 * displacement + 0.01
        clearance = mode.checker.CalcRobotClearance(
            mode.mapping(center), 2 * displacement + 0.005
        )
        self.pairs = sorted(
            {
                tuple(sorted((int(a), int(b))))
                for a, b in zip(clearance.robot_indices(), clearance.other_indices())
            }
        )

    def __call__(self, q):
        q = np.asarray(q).ravel()
        ad = isinstance(q[0], AutoDiffXd)
        values = ExtractValue(q).ravel() if ad else q.astype(float)
        clearance = self.mode.checker.CalcRobotClearance(
            self.mode.mapping(values), self.influence
        )
        found = {}
        for i, (a, b) in enumerate(
            zip(clearance.robot_indices(), clearance.other_indices())
        ):
            pair = tuple(sorted((int(a), int(b))))
            distance = float(clearance.distances()[i])
            jac = clearance.jacobians()[i, self.indices]
            if pair not in found or distance < found[pair][0]:
                found[pair] = (distance, jac)
        derivative = ExtractGradient(q) if ad else None
        result = []
        for pair in self.pairs:
            value, jac = found.get(pair, (self.influence, np.zeros(10)))
            result.append(AutoDiffXd(value, jac @ derivative) if ad else value)
        return np.array(result)


def optimize_task(
    modes,
    contacts,
    out,
    *,
    max_solver_seconds=900,
    contact_radius=None,
    rounded_paths=1,
):
    out = Path(out)
    planner = GcsTrajectoryOptimization(10)
    subgraphs = []
    metadata = []
    handles = []
    for i, mode in enumerate(modes):
        edges = []
        for a, r in enumerate(mode.regions):
            for b in range(a + 1, len(mode.regions)):
                ra, rb = mode.records[a], mode.records[b]
                if np.any(np.asarray(ra["upper"]) < rb["lower"]) or np.any(
                    np.asarray(rb["upper"]) < ra["lower"]
                ):
                    continue
                if r.IntersectsWith(mode.regions[b]):
                    edges.extend([(a, b), (b, a)])
        g = planner.AddRegions(
            mode.regions, edges, order=1, h_min=0.01, h_max=50, name=mode.name
        )
        subgraphs.append(g)
        g.AddPathEnergyCost(np.diag([1.0, 1.0, 0.25] + [0.35] * 7))
        for j, (v, record) in enumerate(zip(g.Vertices(), mode.records)):
            handles.append((v, mode, j))
            delta = np.abs(np.asarray(record["guide_end"]) - record["guide_start"])
            region_time = float(
                max(np.max(delta[:2]) / 0.10, np.max(delta[2:]) / 0.4) + 0.20
            )
            v.AddConstraint(v.x()[-1] == region_time)
            # A tiny SOFT regularizer removes flat directions. It does not pin any
            # intermediate state, including the shared six contact configurations.
            v.AddCost(
                1e-6
                * sum(
                    (v.x()[k] - record["guide_start"][k]) ** 2
                    + (v.x()[10 + k] - record["guide_end"][k]) ** 2
                    for k in range(10)
                )
            )
            if record["contact_nonlinear_required"]:
                f = PairwiseContactClearance(mode, record)
                handles.append((None, f, None))
                cp = g.vertex_control_points()
                # Evaluate pairwise collision distances at both endpoints
                # and the midpoint, with the same variables as the global solve.
                for alpha in [0.0, 0.5, 1.0]:

                    def middle(z, alpha=alpha, f=f):
                        return f((1 - alpha) * z[:10] + alpha * z[10:])

                    binding = callable_binding(
                        middle,
                        np.full(len(f.pairs), -1e-10),
                        np.full(len(f.pairs), np.inf),
                        np.r_[cp[:, 0], cp[:, 1]],
                    )
                    v_binding = Binding[Constraint](
                        binding.evaluator(), np.r_[v.x()[:10], v.x()[10:20]]
                    )
                    # Bind via subgraph placeholders only to this contact-patch vertex.
                    v.AddConstraint(v_binding, RESTRICTION)
                if mode.compiled_inside is not None:
                    compiled = mode.compiled_inside
                    active = compiled.active_rows(record["lower"], record["upper"])
                    for alpha in (0.0, 0.5, 1.0):
                        if not len(active):
                            continue

                        def contact_inside(
                            z, alpha=alpha, compiled=compiled, active=active
                        ):
                            return compiled.g((1 - alpha) * z[:10] + alpha * z[10:])[
                                active
                            ]

                        binding = callable_binding(
                            contact_inside,
                            np.full(len(active), -np.inf),
                            np.zeros(len(active)),
                            np.r_[v.x()[:10], v.x()[10:20]],
                        )
                        v.AddConstraint(binding, RESTRICTION)
        metadata.append(
            dict(
                name=mode.name,
                regions=len(mode.regions),
                edges=len(edges),
                held=mode.held,
                contact_patches=sum(
                    r["contact_nonlinear_required"] for r in mode.records
                ),
                dimension=10,
            )
        )
        if i < len(contacts):
            contact = contacts[i]
            region, record = contact_region(contact, contact_radius)
            gate = planner.AddRegions([region], order=0, name=contact["name"])
            cp = gate.vertex_control_points().ravel()
            binding = callable_binding(
                lambda q, c=contact: c["fk"].error(q, c["target"]),
                np.zeros(6),
                np.zeros(6),
                cp,
            )
            gate.AddVertexConstraint(binding, RESTRICTION)
            for neighbor in (modes[i], modes[i + 1]):
                if neighbor.compiled_inside is not None:
                    compiled = neighbor.compiled_inside
                    active = compiled.active_rows(record["lower"], record["upper"])
                    if not len(active):
                        continue
                    binding = callable_binding(
                        lambda q, compiled=compiled, active=active: compiled.g(q)[
                            active
                        ],
                        np.full(len(active), -np.inf),
                        np.zeros(len(active)),
                        cp,
                    )
                    gate.AddVertexConstraint(binding, RESTRICTION)
            subgraphs.append(gate)
            metadata.append(record)
    start = planner.AddRegions(
        [Point(modes[0].guide[0])], order=0, name="whole_task_initial_state"
    )
    goal = planner.AddRegions(
        [Point(modes[-1].guide[-1])], order=0, name="whole_task_final_state"
    )
    planner.AddEdges(start, subgraphs[0])
    planner.AddEdges(subgraphs[-1], goal)
    transition_records = []
    for i, contact in enumerate(contacts):
        # Linearized FK overlap is not sufficient evidence that a contact-mode
        # transition is physically realizable. Keep anchor-connected subsets with
        # a known exact-FK witness on BOTH sides. The entire shared gate state is
        # still variable; this is a conservative graph restriction, not point IK
        # inserted between independently solved trajectories.
        q = contact["q0"]
        incoming = [j for j, r in enumerate(modes[i].regions) if r.PointInSet(q, 1e-7)]
        outgoing = [
            j for j, r in enumerate(modes[i + 1].regions) if r.PointInSet(q, 1e-7)
        ]
        if not incoming or not outgoing:
            raise RuntimeError(
                "No exact-witness contact transition: " + contact["name"]
            )
        planner.AddEdges(
            subgraphs[2 * i],
            subgraphs[2 * i + 1],
            edges_between_regions=[(j, 0) for j in incoming],
        )
        planner.AddEdges(
            subgraphs[2 * i + 1],
            subgraphs[2 * i + 2],
            edges_between_regions=[(0, j) for j in outgoing],
        )
        transition_records.append(
            dict(
                contact=contact["name"],
                incoming=incoming,
                outgoing=outgoing,
                witness_maximum_fk_error=float(
                    np.max(abs(contact["fk"].error(q, contact["target"])))
                ),
                robot_point_fixed=False,
                scope="Conservative exact-witness-connected subset; untested contacts are not declared infeasible",
            )
        )
    write(out / "contact_transition_graph.json", transition_records)
    planner.AddTimeCost(0.02)
    planner.AddVelocityBounds(
        -np.r_[[0.15, 0.15, 0.25], [0.6] * 7], np.r_[[0.15, 0.15, 0.25], [0.6] * 7]
    )
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.max_rounded_paths = rounded_paths
    options.rounding_seed = 20261102
    options.solver = ClarabelSolver()
    options.restriction_solver = IpoptSolver()
    from pydrake.solvers import SolverOptions

    so = SolverOptions()
    so.SetOption(IpoptSolver.id(), "max_iter", 800)
    so.SetOption(IpoptSolver.id(), "tol", 1e-8)
    so.SetOption(IpoptSolver.id(), "constr_viol_tol", 1e-8)
    so.SetOption(IpoptSolver.id(), "max_cpu_time", float(max_solver_seconds))
    so.SetOption(IpoptSolver.id(), "mu_strategy", "adaptive")
    so.SetOption(IpoptSolver.id(), "print_level", 5)
    so.SetOption(IpoptSolver.id(), "output_file", str(out / "ipopt.log"))
    options.restriction_solver_options = so
    # Accept an accurately feasible local solution without claiming strict
    # stationarity/global optimality. Physical and FK bounds stay at 1e-8.
    for key, value in [
        ("acceptable_tol", 0.05),
        ("acceptable_dual_inf_tol", 0.05),
        ("acceptable_constr_viol_tol", 1e-8),
        ("acceptable_compl_inf_tol", 1e-7),
        ("acceptable_obj_change_tol", 1e-6),
        ("acceptable_iter", 10),
    ]:
        so.SetOption(IpoptSolver.id(), key, value)
    options.restriction_solver_options = so
    write(
        out / "global_model.json",
        dict(
            base_tracking_margin_m=float(
                modes[0]
                .scene.config.get("whole_task_planning", {})
                .get("base_tracking_margin_m", 0.0)
            ),
            dimension=10,
            motion_modes=7,
            variable_contact_states=6,
            internal_robot_point_constraints=0,
            point_constraints_only=[
                "whole_task_initial_state",
                "whole_task_final_state",
            ],
            solve_path_calls=1,
            global_shared_objective=True,
            rounded_path_budget=rounded_paths,
            objective="Weighted joint path energy + region duration cost + 1e-6 soft guide regularizer",
            timing="Fixed feasible region durations; this run jointly optimizes spatial paths, not global retiming.",
            subgraphs=metadata,
            scope="One task-wide convex GCS relaxation plus one global nonlinear restriction. Contact patches are constrained mode domains, not collision-free IRIS sets. Fixed task order and guide-local domains.",
        ),
    )
    print(
        "GLOBAL_SOLVE_STARTED",
        sum(len(m.regions) for m in modes),
        "regions",
        flush=True,
    )
    tick = time.monotonic()
    trajectory, result = planner.SolvePath(start, goal, options)
    elapsed = time.monotonic() - tick
    write(
        out / "solve_status.json",
        dict(
            success=result.is_success(),
            status=str(result.get_solution_result()),
            seconds=elapsed,
            solver_calls=1,
        ),
    )
    # Graph vertex solutions identify selected modes and all six shared gates.
    solved_contacts = []
    for contact, gate in zip(contacts, subgraphs[1::2]):
        v = gate.Vertices()[0]
        solution = result.GetSolution(v.x())[:10]
        error = np.array(contact["fk"].error(solution, contact["target"]), float)
        solved_contacts.append(
            dict(
                name=contact["name"],
                q=solution.tolist(),
                anchor=contact["q0"].tolist(),
                maximum_fk_error=float(np.max(np.abs(error))),
                robot_change_from_anchor=float(
                    np.max(np.abs(solution - contact["q0"]))
                ),
            )
        )
    write(out / "optimized_contacts.json", solved_contacts)
    if not result.is_success():
        raise RuntimeError(
            "Whole-task shared solve failed: " + str(result.get_solution_result())
        )
    segments = []
    lookup = {
        v.id(): (mode, j) for v, mode, j in handles if isinstance(mode, MotionMode)
    }
    graph = planner.graph_of_convex_sets()
    current = start.Vertices()[0]
    target = goal.Vertices()[0]
    visited = []
    while current.id() != target.id():
        if current.id() in visited:
            raise RuntimeError("Cycle in rounded task path")
        visited.append(current.id())
        edges = [
            e
            for e in graph.Edges()
            if e.u().id() == current.id() and result.GetSolution(e.phi()) > 0.5
        ]
        if len(edges) != 1:
            raise RuntimeError(
                "Rounded global path must have exactly one outgoing edge"
            )
        current = edges[0].v()
        if current.id() in lookup:
            mode, j = lookup[current.id()]
            value = result.GetSolution(current.x())
            a = value[:10]
            b = value[10:20]
            if not mode.regions[j].PointInSet(a, 1e-6) or not mode.regions[
                j
            ].PointInSet(b, 1e-6):
                raise RuntimeError("Selected global segment violates its mode set")
            segments.append(
                dict(
                    mode=mode.name,
                    region_index=j,
                    kind=mode.records[j]["kind"],
                    a=a.tolist(),
                    b=b.tolist(),
                )
            )
    write(out / "global_segments.json", segments)
    return planner, trajectory, result, subgraphs, solved_contacts, elapsed
