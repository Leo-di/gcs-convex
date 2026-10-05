"""Seven independent arm joints using official Drake IRIS-NP2 and GCS.

Fixed base, fixed open gripper, stationary cup. No prescribed posture family or
precomputed path is used. All region matrices and failures are retained.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import time

import numpy as np
from pydrake.common import Parallelism, RandomGenerator
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid, Point, GraphOfConvexSetsOptions
from pydrake.planning import CollisionCheckerParams, SceneGraphCollisionChecker, IrisNp2, IrisNp2Options, IrisParameterizationFunction, GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver, IpoptSolver

from .joint_cspace_research import make_scene, BASE, SEMANTIC_MARGIN, write_json
from .iris_region_generator import _reduced_position_indices, _make_parameterization
from .arm_boundary_semantics import ArmBoundaryOracle
from .collision_checker import evaluate_collision_status

OUTPUT = Path('outputs/results/arm_cspace_research_v1')
WEB = Path('outputs/results/grasp_fix_web/planar/arm_cspace')


def setup(scene, parallelism=1):
    params = CollisionCheckerParams()
    params.model = scene.diagram
    params.robot_model_instances = [scene.base_instance, scene.arm_instance, scene.gripper_instance]
    params.edge_step_size = .01
    params.implicit_context_parallelism = Parallelism(parallelism)
    checker = SceneGraphCollisionChecker(params)
    indices = _reduced_position_indices(scene)[3:]
    if len(indices) != 7:
        raise ValueError('Expected seven independent iiwa positions')
    full = scene.plant.GetPositions(scene.plant_context).copy()
    return (checker, _make_parameterization(full, indices),
            scene.plant.GetPositionLowerLimits()[indices], scene.plant.GetPositionUpperLimits()[indices])


def audit(scene, oracle, points):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 7 or len(points) == 0:
        raise ValueError('Expected nonempty N by 7 array')
    started = time.perf_counter()
    clearance, residual = float('inf'), float('inf')
    collisions = violations = 0
    witnesses = []
    for i, q in enumerate(points):
        c = evaluate_collision_status(scene, BASE, arm_q=q)
        r = oracle.query(np.r_[BASE, q])['residual_m']
        clearance = min(clearance, c.minimum_signed_distance)
        residual = min(residual, r)
        collisions += int(not c.collision_free)
        violations += int(r < -1e-8)
        if (not c.collision_free or r < -1e-8) and len(witnesses) < 5:
            witnesses.append(dict(index=i,q=q.tolist(),clearance_m=c.minimum_signed_distance,residual_m=r))
    return dict(samples=len(points), collisions=collisions, semantic_violations=violations,
                minimum_collision_distance_m=clearance, minimum_semantic_residual_m=residual,
                witnesses=witnesses, seconds=time.perf_counter()-started,
                scope='Independent original SceneGraph and full-arm support oracle; finite numerical audit')


def full_dimensional_radius(region):
    center = region.ChebyshevCenter()
    norms = np.linalg.norm(region.A(), axis=1)
    return float(np.min((region.b()-region.A()@center)[norms>1e-12]/norms[norms>1e-12]))


def parallel_seed(config, semantic, seed, candidate_index, max_regions, ellipse_A=None):
    """One independent official IRIS call; private Drake models per process."""
    scene=make_scene(config,semantic=semantic)
    original=make_scene(config)
    oracle=ArmBoundaryOracle(original)
    checker,parameterize,lower,upper=setup(scene)
    seed=np.asarray(seed)
    options=IrisNp2Options()
    options.parameterization=IrisParameterizationFunction(parameterize,7)
    options.solver=IpoptSolver()
    options.add_hyperplane_if_solve_fails=False
    s=options.sampled_iris_options
    s.num_particles=2000;s.epsilon=.0025;s.delta=.01/(2*max_regions)
    s.max_iterations=6;s.max_iterations_separating_planes=30;s.max_separating_planes_per_iteration=10
    s.configuration_space_margin=.01;s.mixing_steps=50;s.parallelism=Parallelism(1)
    s.random_seed=5100+candidate_index
    began=time.perf_counter()
    record=dict(candidate_index=candidate_index,seed=seed.tolist(),accepted=False)
    print(f'PARALLEL IRIS7 semantic={semantic} candidate={candidate_index}',flush=True)
    try:
        ellipse=(Hyperellipsoid.MakeHypersphere(.01,seed) if ellipse_A is None
                 else Hyperellipsoid(np.asarray(ellipse_A),seed))
        record['initial_ellipse_A']=ellipse.A().tolist()
        region=IrisNp2(checker,ellipse,HPolyhedron.MakeBox(lower,upper),options)
        record.update(iris_seconds=time.perf_counter()-began,A=region.A().tolist(),b=region.b().tolist(),
                      seed_contained=bool(region.PointInSet(seed)),inscribed_radius_rad=full_dimensional_radius(region))
        rng=RandomGenerator(10800+candidate_index);point=region.ChebyshevCenter();points=[]
        for _ in range(1000):
            point=region.UniformSample(rng,point,mixing_steps=40);points.append(point.copy())
        record['audit']=audit(original,oracle,points)
        record['accepted']=(record['inscribed_radius_rad']>1e-6 and record['seed_contained']
                            and record['audit']['collisions']==0
                            and (not semantic or record['audit']['semantic_violations']==0))
    except RuntimeError as exc:
        record.update(error=str(exc),elapsed_to_error_seconds=time.perf_counter()-began)
    record['attempt_total_seconds']=time.perf_counter()-began
    print(f'FINISHED IRIS7 semantic={semantic} candidate={candidate_index} accepted={record["accepted"]}',flush=True)
    return record


def grow_parallel(config, scene, start, goal, semantic, args, output):
    checker,parameterize,lower,upper=setup(scene)
    candidates=np.vstack([start,goal,np.random.default_rng(713).uniform(lower,upper,(600,7))])
    records=[];regions=[];cursor=0
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        while len(regions)<args.regions and len(records)<2*args.regions and cursor<len(candidates):
            jobs=[]
            needed=min(args.workers,args.regions-len(regions),2*args.regions-len(records))
            while len(jobs)<needed and cursor<len(candidates):
                index=cursor;seed=candidates[cursor];cursor+=1
                if any(r.PointInSet(seed) for r in regions) or not checker.CheckConfigCollisionFree(parameterize(seed)):
                    continue
                jobs.append(pool.submit(parallel_seed,config,semantic,seed.tolist(),index,args.regions))
            if not jobs:
                break
            # Collect in proposal order, so process completion order cannot
            # change region numbering or GCS rounding behavior.
            for job in jobs:
                record=job.result()
                if record['accepted']:
                    record['region_index']=len(regions)
                    regions.append(HPolyhedron(np.array(record['A']),np.array(record['b'])))
                records.append(record)
                write_json(output,records)
    return regions,records


def grow(scene, original, oracle, start, goal, semantic, args, output):
    checker, parameterize, lower, upper = setup(scene)
    candidates = np.vstack([start, goal, np.random.default_rng(713).uniform(lower,upper,(600,7))])
    regions, records = [], []
    for candidate_index, seed in enumerate(candidates):
        if len(regions) >= args.regions or len(records) >= args.regions*2:
            break
        if any(r.PointInSet(seed) for r in regions) or not checker.CheckConfigCollisionFree(parameterize(seed)):
            continue
        options = IrisNp2Options()
        options.parameterization = IrisParameterizationFunction(parameterize, 7)
        options.solver = IpoptSolver()
        options.add_hyperplane_if_solve_fails = False
        s = options.sampled_iris_options
        s.num_particles = 2000
        s.epsilon = .0025
        s.delta = .01/(2*args.regions)
        s.max_iterations = 6
        s.max_iterations_separating_planes = 30
        s.max_separating_planes_per_iteration = 10
        s.configuration_space_margin = .01
        s.mixing_steps = 50
        s.parallelism = Parallelism(1)
        s.random_seed = 5100+candidate_index
        began = time.perf_counter()
        record = dict(candidate_index=candidate_index,seed=seed.tolist(),accepted=False)
        print(f'IRIS7 semantic={semantic} candidate={candidate_index} accepted={len(regions)}',flush=True)
        try:
            region = IrisNp2(checker,Hyperellipsoid.MakeHypersphere(.01,seed),HPolyhedron.MakeBox(lower,upper),options)
            record.update(iris_seconds=time.perf_counter()-began,A=region.A().tolist(),b=region.b().tolist(),
                          seed_contained=bool(region.PointInSet(seed)),inscribed_radius_rad=full_dimensional_radius(region))
            rng = RandomGenerator(10800+candidate_index)
            point = region.ChebyshevCenter()
            points=[]
            for _ in range(1000):
                point=region.UniformSample(rng,point,mixing_steps=40)
                points.append(point.copy())
            record['audit']=audit(original,oracle,points)
            record['accepted']=(record['inscribed_radius_rad']>1e-6 and record['seed_contained'] and
                                record['audit']['collisions']==0 and
                                (not semantic or record['audit']['semantic_violations']==0))
            if record['accepted']:
                record['region_index']=len(regions)
                regions.append(region)
        except RuntimeError as exc:
            record.update(error=str(exc),elapsed_to_error_seconds=time.perf_counter()-began)
        record['attempt_total_seconds']=time.perf_counter()-began
        records.append(record)
        write_json(output,records)
        print(json.dumps({k:v for k,v in record.items() if k not in ('A','b','seed','audit')})+
              f' audit={record.get("audit",{}).get("collisions")}/{record.get("audit",{}).get("semantic_violations")}',flush=True)
    return regions, records


def solve(regions, start, goal, original, oracle, semantic):
    if not regions:
        return dict(success=False,accepted=False,reason='No audited full-dimensional regions')
    started=time.perf_counter()
    gcs=GcsTrajectoryOptimization(7)
    free=gcs.AddRegions(regions,order=3,h_min=.05,h_max=20.,name='iris_7d')
    source=gcs.AddRegions([Point(start)],order=0,name='start')
    target=gcs.AddRegions([Point(goal)],order=0,name='goal')
    gcs.AddEdges(source,free)
    gcs.AddEdges(free,target)
    gcs.AddPathLengthCost(1.)
    gcs.AddTimeCost(.05)
    gcs.AddVelocityBounds(np.full(7,-.6),np.full(7,.6))
    gcs.AddPathContinuityConstraints(1)
    options=GraphOfConvexSetsOptions()
    options.convex_relaxation=True
    options.preprocessing=True
    options.max_rounded_paths=20
    options.rounding_seed=41
    options.solver=ClarabelSolver()
    options.restriction_solver=ClarabelSolver()
    build_seconds=time.perf_counter()-started
    started=time.perf_counter()
    traj,result=gcs.SolvePath(source,target,options)
    record=dict(success=bool(result.is_success()),accepted=False,build_seconds=build_seconds,
                solve_seconds=time.perf_counter()-started,solver=result.get_solver_id().name(),
                status=str(result.get_solution_result()))
    if not result.is_success():
        return record
    times=np.linspace(traj.start_time(),traj.end_time(),max(1001,int(traj.end_time()/.005)+1))
    points=np.array([traj.value(t).ravel() for t in times])
    checked=audit(original,oracle,points)
    # Preserve actual polynomial coefficients and verify the convex-hull
    # containment condition, rather than inferring it from sampled path points.
    segments=[]
    for i in range(traj.get_number_of_segments()):
        segment=traj.segment(i)
        controls=segment.control_points()
        containing=[j for j,r in enumerate(regions) if np.max(r.A()@controls-r.b()[:,None])<=1e-6]
        segments.append(dict(start_s=segment.start_time(),end_s=segment.end_time(),
                             control_points=controls.tolist(),containing_regions=containing))
    speed=float(max(np.max(np.abs(traj.EvalDerivative(t))) for t in times))
    start_error=float(np.linalg.norm(points[0]-start))
    goal_error=float(np.linalg.norm(points[-1]-goal))
    record.update(objective=float(result.get_optimal_cost()),duration_s=float(traj.end_time()),audit=checked,
                  trajectory_samples=points.tolist(),sample_times=times.tolist(),segments=segments,
                  joint_ranges_rad=np.ptp(points,axis=0).tolist(),
                  start_error_rad=start_error,goal_error_rad=goal_error,max_sampled_speed_rad_s=speed,
                  sampled_path_length_rad=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()))
    record['accepted']=(checked['collisions']==0 and (not semantic or checked['semantic_violations']==0)
                        and start_error<1e-6 and goal_error<1e-6 and speed<.60001
                        and all(s['containing_regions'] for s in segments))
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--regions',type=int,default=6)
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    if args.regions<2:
        parser.error('At least two regions required')
    if not 1<=args.workers<=12:
        parser.error('Workers must be between 1 and 12')
    args.output.mkdir(parents=True,exist_ok=False)
    config=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())['config']
    write_json(args.output/'config.json',config)
    original=make_scene(config)
    oracle=ArmBoundaryOracle(original)
    _,_,lower,upper=setup(original)
    start=original.arm_home_positions.copy();start[0]=-1.6
    goal=original.arm_home_positions.copy();goal[0]=1.6
    write_json(args.output/'protocol.json',dict(
        drake_version=importlib.metadata.version('drake'),dimension=7,active_joints=[f'iiwa_joint_{i}' for i in range(1,8)],
        base=BASE.tolist(),lower=lower.tolist(),upper=upper.tolist(),start=start.tolist(),goal=goal.tolist(),
        algorithms=['Official Drake IrisNp2 (Werner et al. 2024)','Official Drake GcsTrajectoryOptimization'],
        seed_policy='start, goal, 600 fixed-seed uniform proposals over all seven joint limits; only uncovered feasible seeds',
        max_accepted_regions=args.regions,max_attempts_per_condition=2*args.regions,
        workers=args.workers,parallel_seed_policy='Independent batches, uncovered against preceding batches; collection in proposal order',
        region_particles=2000,region_epsilon=.0025,region_delta=.01/(2*args.regions),
        region_iterations=6,separation_iterations=30,configuration_space_margin_rad=.01,holdout_per_region=1000,
        semantic_margin_m=SEMANTIC_MARGIN,gripper='fixed open',cup='stationary, robot contacts forbidden',
        objective='Bezier control polygon length upper bound + 0.05 duration',
        velocity_bound_rad_s=.6,continuity='C1 normalized path, only C0 guaranteed in physical time',
        scope='Seven independent arm joints over their full limit box; base fixed. Not 10D whole-body or dynamics.',
        references=['https://arxiv.org/abs/2205.04422','https://arxiv.org/abs/2303.14737','https://arxiv.org/abs/2410.12649']))
    files=['src/arm_cspace_research.py','src/joint_cspace_research.py','src/iris_region_generator.py',
           'src/arm_boundary_semantics.py','src/collision_checker.py','src/build_scene.py']
    hashes={}
    for filename in files:
        dst=args.output/'source_snapshot'/filename;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(filename,dst);hashes[filename]=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    write_json(args.output/'source_sha256.json',hashes)
    summary={}
    for semantic in (False,True):
        label='semantic' if semantic else 'collision_only'
        began=time.perf_counter()
        planning=make_scene(config,semantic=semantic)
        if args.workers>1:
            regions,records=grow_parallel(config,planning,start,goal,semantic,args,args.output/f'{label}_regions.json')
        else:
            regions,records=grow(planning,original,oracle,start,goal,semantic,args,args.output/f'{label}_regions.json')
        try:
            plan=solve(regions,start,goal,original,oracle,semantic)
        except RuntimeError as exc:
            plan=dict(success=False,accepted=False,error=str(exc))
        write_json(args.output/f'{label}_plan.json',plan)
        summary[label]=dict(accepted_regions=len(regions),attempts=len(records),
                            condition_total_seconds=time.perf_counter()-began,
                            **{k:v for k,v in plan.items() if k not in ('sample_times','trajectory_samples','segments')})
        write_json(args.output/'summary.json',summary)
        print(label,json.dumps(summary[label]),flush=True)


if __name__=='__main__':
    main()
