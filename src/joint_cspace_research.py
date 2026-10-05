"""Two real joints in the original scene: official Drake IRIS-NP2 and GCS.

The raster is diagnostic only. It is never used to construct regions or paths.
The semantic planning clone encodes x <= -margin as a virtual forbidden box.
The original scene and independent full-arm support oracle audit every result.
"""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
from pydrake.common import Parallelism, RandomGenerator
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.geometry.optimization import (
    GraphOfConvexSetsOptions, HPolyhedron, Hyperellipsoid, Point, VPolytope,
)
from pydrake.planning import (
    CollisionCheckerParams, SceneGraphCollisionChecker, IrisNp2, IrisNp2Options,
    IrisParameterizationFunction, GcsTrajectoryOptimization,
)
from pydrake.solvers import ClarabelSolver, IpoptSolver

from .build_scene import build_scene, set_robot_configuration
from .arm_boundary_semantics import ArmBoundaryOracle
from .collision_checker import evaluate_collision_status
from .iris_region_generator import _reduced_position_indices, _make_parameterization

OUTPUT = Path('outputs/results/joint_cspace_research_v1')
WEB = Path('outputs/results/grasp_fix_web/planar/joint_cspace')
BASE = np.array([-.6, 0., 0.])
START = np.array([-1.6, .45])
GOAL = np.array([1.6, .45])
SEMANTIC_MARGIN = .002


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False))


def make_scene(config, semantic=False, visualize=False):
    config = copy.deepcopy(config)
    if semantic:
        # A planning constraint only, never added to the physical/display scene.
        # All moving robot geometry is within 20 m of the fixed base; hence this
        # finite box represents the forbidden half-space over this experiment.
        # The base is fixed entirely on the allowed side; the cup is environment.
        config['world']['walls'].append(dict(
            name='planning_only_arm_boundary', size=[20., 40., 40.],
            center=[10.-SEMANTIC_MARGIN, 0., 0.]))
    scene = build_scene(config=config, visualize=visualize)
    # No grasp/contact transition in this experiment. Re-enable ALL cup/robot
    # pairs instead of relaxing the fingertip-specific filters to body-wide ones.
    ids = [g for inst in (scene.base_instance, scene.arm_instance, scene.gripper_instance)
           for b in scene.plant.GetBodyIndices(inst)
           for g in scene.plant.GetCollisionGeometriesForBody(scene.plant.get_body(b))]
    scene.scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().AllowBetween(
            GeometrySet(ids), GeometrySet(scene.plant.GetCollisionGeometriesForBody(scene.cup_body))))
    scene.root_context = scene.diagram.CreateDefaultContext()
    scene.plant_context = scene.plant.GetMyMutableContextFromRoot(scene.root_context)
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, scene.cup_initial_pose)
    set_robot_configuration(scene, BASE, scene.arm_home_positions, scene.gripper_open_positions)
    return scene


def setup_checker(scene):
    params = CollisionCheckerParams()
    params.model = scene.diagram
    params.robot_model_instances = [scene.base_instance, scene.arm_instance, scene.gripper_instance]
    params.edge_step_size = .01
    params.implicit_context_parallelism = Parallelism(1)
    checker = SceneGraphCollisionChecker(params)
    full = scene.plant.GetPositions(scene.plant_context).copy()
    indices = _reduced_position_indices(scene)[3:5]
    parameterization = _make_parameterization(full, indices)
    lower = scene.plant.GetPositionLowerLimits()[indices]
    upper = scene.plant.GetPositionUpperLimits()[indices]
    return checker, parameterization, lower, upper


def state(scene, q):
    arm = scene.arm_home_positions.copy()
    arm[:2] = q
    return np.r_[BASE, arm]


def audit_points(scene, oracle, points):
    minimum_distance = float('inf')
    minimum_semantic = float('inf')
    collisions = semantic_violations = 0
    for q in points:
        s = state(scene, q)
        collision = evaluate_collision_status(scene, s[:3], arm_q=s[3:])
        residual = oracle.query(s)['residual_m']
        minimum_distance = min(minimum_distance, collision.minimum_signed_distance)
        minimum_semantic = min(minimum_semantic, residual)
        collisions += int(not collision.collision_free)
        semantic_violations += int(residual < -1e-8)
    return dict(samples=len(points), collisions=collisions, semantic_violations=semantic_violations,
                minimum_collision_distance_m=minimum_distance, minimum_semantic_residual_m=minimum_semantic,
                scope='Independent finite numerical audit, not a continuous-time or whole-region certificate')


def grow_cover(scene, original, oracle, semantic, max_regions, output):
    checker, parameterize, lower, upper = setup_checker(scene)
    domain = HPolyhedron.MakeBox(lower, upper)
    candidates = np.vstack([START, GOAL, np.random.default_rng(713).uniform(lower, upper, (400, 2))])
    regions, records = [], []
    for seed_index, seed in enumerate(candidates):
        if len(regions) >= max_regions:
            break
        if any(r.PointInSet(seed) for r in regions) or not checker.CheckConfigCollisionFree(parameterize(seed)):
            continue
        options = IrisNp2Options()
        options.parameterization = IrisParameterizationFunction(parameterize, 2)
        options.solver = IpoptSolver()
        options.add_hyperplane_if_solve_fails = False
        sampled = options.sampled_iris_options
        sampled.parallelism = Parallelism(1)
        sampled.num_particles = 1000
        sampled.epsilon = .005
        sampled.delta = .01 / max_regions
        sampled.max_iterations = 5
        sampled.max_iterations_separating_planes = 30
        sampled.max_separating_planes_per_iteration = 10
        sampled.configuration_space_margin = .01
        sampled.random_seed = 4100 + seed_index
        sampled.mixing_steps = 50
        began = time.perf_counter()
        print(f'IRIS semantic={semantic} seed={seed_index} q={seed.tolist()}', flush=True)
        try:
            region = IrisNp2(checker, Hyperellipsoid.MakeHypersphere(.01, seed), domain, options)
            seconds = time.perf_counter()-began
            rng = RandomGenerator(9800+seed_index)
            point = region.ChebyshevCenter()
            points = []
            for _ in range(500):
                point = region.UniformSample(rng, point, mixing_steps=20)
                points.append(point.copy())
            audit = audit_points(original, oracle, points)
            accepted = audit['collisions'] == 0 and (not semantic or audit['semantic_violations'] == 0)
            vertices = VPolytope(region).vertices().T
            center = vertices.mean(axis=0)
            vertices = vertices[np.argsort(np.arctan2(*(vertices-center)[:, ::-1].T))]
            record = dict(seed=seed.tolist(), seed_index=seed_index, seconds=seconds,
                          A=region.A().tolist(), b=region.b().tolist(), polygon=vertices.tolist(),
                          audit=audit, accepted=accepted)
            if accepted:
                regions.append(region)
        except RuntimeError as exc:
            record = dict(seed=seed.tolist(), seed_index=seed_index,
                          seconds=time.perf_counter()-began, accepted=False, error=str(exc))
        records.append(record)
        write_json(output, records)
        print(f'  accepted={record["accepted"]}, regions={len(regions)}, seconds={record["seconds"]:.2f}', flush=True)
        # A strict attempt budget prevents hiding repeated failures behind retries.
        if len(records) >= 2*max_regions:
            break
    return regions, records


def solve_gcs(regions, scene, oracle):
    if not regions:
        return dict(success=False, reason='No accepted IRIS regions')
    gcs = GcsTrajectoryOptimization(2)
    free = gcs.AddRegions(regions, order=3, h_min=.05, h_max=20., name='iris_cover')
    source = gcs.AddRegions([Point(START)], order=0, name='start')
    target = gcs.AddRegions([Point(GOAL)], order=0, name='goal')
    gcs.AddEdges(source, free)
    gcs.AddEdges(free, target)
    gcs.AddPathLengthCost(1.)
    gcs.AddTimeCost(.05)
    gcs.AddVelocityBounds(np.full(2, -.6), np.full(2, .6))
    gcs.AddPathContinuityConstraints(1)
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.preprocessing = True
    options.max_rounded_paths = 20
    options.rounding_seed = 41
    options.solver = ClarabelSolver()
    options.restriction_solver = ClarabelSolver()
    began = time.perf_counter()
    trajectory, result = gcs.SolvePath(source, target, options)
    record = dict(success=result.is_success(), seconds=time.perf_counter()-began,
                  solver=result.get_solver_id().name(), status=str(result.get_solution_result()),
                  objective_description='Bezier control polygon length upper bound + 0.05 * duration',
                  continuity='C0 physical time, C1 normalized path; not necessarily C1 physical time')
    if not result.is_success():
        return record
    duration = trajectory.end_time()
    times = np.linspace(0, duration, max(1001, int(duration/.005)+1))
    points = np.array([trajectory.value(t).ravel() for t in times])
    audit = audit_points(scene, oracle, points)
    record.update(objective=float(result.get_optimal_cost()), duration_s=duration,
                  audit=audit, trajectory_samples=points.tolist(), sample_times=times.tolist(),
                  start_error_rad=float(np.linalg.norm(points[0]-START)),
                  goal_error_rad=float(np.linalg.norm(points[-1]-GOAL)),
                  max_sampled_speed_rad_s=float(max(np.max(np.abs(trajectory.EvalDerivative(t))) for t in times)),
                  sampled_path_length_rad=float(np.linalg.norm(np.diff(points, axis=0),axis=1).sum()))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--regions', type=int, default=8)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path('tests/fixtures/arm_boundary.json').read_text())['config']
    write_json(args.output/'config.json', config)
    original = make_scene(config)
    oracle = ArmBoundaryOracle(original)
    checker, parameterize, lower, upper = setup_checker(original)
    write_json(args.output/'protocol.json', dict(
        drake_version=importlib.metadata.version('drake'),
        region_algorithm='Official Drake IrisNp2, Werner et al. 2024; NOT original 2023 IRIS-NP',
        planner='Official Drake GcsTrajectoryOptimization: cubic Bezier, convex relaxation, rounding, restriction',
        references=['https://arxiv.org/abs/2205.04422','https://arxiv.org/abs/2303.14737','https://arxiv.org/abs/2410.12649'],
        active_joints=['iiwa_joint_1','iiwa_joint_2'], lower=lower.tolist(), upper=upper.tolist(),
        base=BASE.tolist(), fixed_arm_home=original.arm_home_positions.tolist(),
        start=START.tolist(), goal=GOAL.tolist(), max_accepted_regions=args.regions,
        max_attempts_per_condition=2*args.regions, seed_policy='start, goal, uncovered feasible uniform proposals (seed 713); no grid planner',
        semantic_margin_m=SEMANTIC_MARGIN, iris_epsilon=.005, iris_delta_per_region=.01/args.regions,
        iris_particles=1000, iris_iterations=5, iris_separation_iterations=30, holdout_per_region=500,
        collision_policy='Original collision geometry; cup-to-robot collisions fully enabled in no-contact mode',
        semantic_adapter='Planning-only box x=[-0.002,19.998], y,z=[-20,20]. Base fixed strictly inside allowed halfspace. Original full-arm oracle independently audits.',
        scope='2 independent real joints, remaining coordinates fixed; no 10D completeness, contact dynamics, or strict region safety claim'))
    summary = {}
    for semantic in (False, True):
        label = 'semantic' if semantic else 'collision_only'
        planning = make_scene(config, semantic=semantic)
        regions, records = grow_cover(planning, original, oracle, semantic, args.regions, args.output/f'{label}_regions.json')
        result = solve_gcs(regions, original, oracle)
        write_json(args.output/f'{label}_plan.json', result)
        summary[label] = dict(accepted_regions=len(regions), attempts=len(records),
                              generation_seconds=sum(r['seconds'] for r in records),
                              **{k:v for k,v in result.items() if k not in ('trajectory_samples','sample_times')})
        write_json(args.output/'summary.json', summary)
        print(label, json.dumps(summary[label]), flush=True)


if __name__ == '__main__':
    main()
