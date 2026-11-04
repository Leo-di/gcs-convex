"""Guide-local 10D IRIS/GCS for simultaneous empty-hand base and arm motion.

Both base SE(2) and all seven arm joints are variables in the same GCS.
Semantic domain bounds precede official IRIS-NP2. Independent sampled
collision and route checks gate acceptance; no global-safety claim.
"""
import json,time
from pathlib import Path
import numpy as np
from pydrake.common import Parallelism
from pydrake.geometry.optimization import HPolyhedron,Hyperellipsoid,Point,GraphOfConvexSetsOptions
from pydrake.planning import IrisNp2,IrisNp2Options,IrisParameterizationFunction,GcsTrajectoryOptimization
from pydrake.solvers import IpoptSolver,ClarabelSolver
from .build_scene import set_robot_configuration
from .green_room_iris import make_checker
from .green_room_gcs import operation_oracle
from .whole_robot_room import arm_point_reach_bound
from .iris_region_generator import _make_parameterization,_reduced_position_indices
from .room_entry import compress_collinear

def motion_bound(half_width,reach):
 h=np.asarray(half_width,float)
 if h.shape!=(10,) or np.any(h<0) or not np.all(np.isfinite(h)) or not np.isfinite(reach) or reach<=0:raise ValueError('Finite nonnegative 10D half widths and positive reach required')
 return float(np.linalg.norm(h[:2])+reach*np.sum(h[2:]))

def interpolate_q10(points,base_step=.01,joint_step=.01):
 points=np.asarray(points,float)
 if points.ndim!=2 or points.shape[1]!=10 or len(points)<2 or not np.all(np.isfinite(points)):raise ValueError('Finite 10D polyline required')
 out=[points[0]]
 for a,b in zip(points,points[1:]):
  n=max(1,int(np.ceil(max(np.linalg.norm(b[:2]-a[:2])/base_step,np.max(np.abs(b[2:]-a[2:]))/joint_step))))
  out.extend(a+(b-a)*t for t in np.linspace(0,1,n+1)[1:])
 return np.asarray(out)

def plan_joint_approach(config,guide,gripper,out,*,max_seconds=900):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);began=time.perf_counter();guide=np.asarray(guide,float)
 scene,checker,_=make_checker(config,guide[0,:3],guide[0,3:],gripper)
 mapping=_make_parameterization(scene.plant.GetPositions(scene.plant_context).copy(),_reduced_position_indices(scene))
 oracle=operation_oracle(scene);reach=arm_point_reach_bound(scene,oracle)
 rng=np.random.default_rng(20261102);regions=[];records=[];rejected=[]
 def dump(name,value):(out/name).write_text(json.dumps(value,indent=2))
 def query(q):
  if getattr(oracle,'clear',None) is not None:oracle.clear.passage.reset_witness_cache()
  return oracle.query(q,gripper)
 def check(q):return bool(checker.CheckConfigCollisionFree(mapping(q)))
 if any(not check(q) for q in guide):raise RuntimeError('Source guide has a collision under 10D checker')
 joint=config['whole_body_planning'];lower=np.r_[[-1.45,-2.3,-np.pi],joint['joint_lower_limits']];upper=np.r_[[5.25,5.05,np.pi],joint['joint_upper_limits']]
 padding=np.r_[[.03,.03,.003],[.006]*7]
 def grow(a,b,depth=0,pad_scale=1.):
  if time.perf_counter()-began>max_seconds or len(regions)>400:raise RuntimeError('Joint cover budget exceeded')
  lo=np.maximum(np.minimum(a,b)-padding*pad_scale,lower);hi=np.minimum(np.maximum(a,b)+padding*pad_scale,upper);center=(lo+hi)/2
  v=query(center);bound=float(v['residual_m'])-motion_bound((hi-lo)/2,reach)
  if bound<1e-6 or not check(center):
   if depth>=14:raise RuntimeError('Cannot certify a guide-local joint domain')
   m=(a+b)/2;grow(a,m,depth+1,pad_scale*.65);grow(m,b,depth+1,pad_scale*.65);return
  seed=center;domain=HPolyhedron.MakeBox(lo,hi);opts=IrisNp2Options();opts.parameterization=IrisParameterizationFunction(mapping,10);opts.solver=IpoptSolver();opts.add_hyperplane_if_solve_fails=False
  opts.ray_sampler_options.num_particles_to_walk_towards=100;p=opts.sampled_iris_options;p.num_particles=200;p.max_iterations=3;p.max_iterations_separating_planes=15;p.max_separating_planes_per_iteration=10;p.parallelism=Parallelism(1);p.sample_particles_in_parallel=False;p.verbose=False;p.delta=.01/802;p.epsilon=.01;p.configuration_space_margin=1e-6;p.mixing_steps=20;p.random_seed=20261102+len(records)+len(rejected);p.containment_points=np.column_stack([a,seed,b])
  tick=time.perf_counter()
  region=IrisNp2(checker,Hyperellipsoid.MakeHypersphere(min(.0001,float(np.min(hi-lo)/8)),seed),domain,opts)
  seconds=time.perf_counter()-tick
  if not region.PointInSet(a,1e-7) or not region.PointInSet(b,1e-7):raise RuntimeError('10D IRIS excluded a guide endpoint')
  tested=[];bad=[]
  for _ in range(6000):
   q=rng.uniform(lo,hi)
   if not region.PointInSet(q,1e-10):continue
   free=check(q);tested.append(q.tolist())
   if not free:bad.append(q.tolist())
   if len(tested)>=64:break
  if len(tested)<64:raise RuntimeError('Insufficient independent 10D region holdout samples')
  record=dict(lower=lo.tolist(),upper=hi.tolist(),A=region.A().tolist(),b=region.b().tolist(),seed=seed.tolist(),guide_start=a.tolist(),guide_end=b.tolist(),semantic_lower_bound_m=bound,iris_seconds=seconds,holdout_samples=len(tested),holdout_collision_count=len(bad))
  if bad:
   rejected.append(dict(record=record,counterexamples=bad));dump('rejected_regions.json',rejected)
   if depth>=14:raise RuntimeError('Repeated 10D region holdout rejection')
   m=(a+b)/2;grow(a,m,depth+1,pad_scale*.5);grow(m,b,depth+1,pad_scale*.5);return
  records.append(record);regions.append(region);dump('regions.json',records)
  print(f'joint IRIS {len(regions)}: {region.A().shape[0]} faces, {seconds:.2f}s, lower={bound:.4f}',flush=True)
 def grow_guide(points):
  compact=compress_collinear(points,tolerance=1e-8)
  for a,b in zip(compact,compact[1:]):
   n=max(1,int(np.ceil(max(np.linalg.norm(b[:2]-a[:2])/.18,np.max(np.abs(b[2:]-a[2:]))/.12))))
   for j in range(n):grow(a+(b-a)*j/n,a+(b-a)*(j+1)/n)
 grow_guide(guide)
 # Offer genuinely coordinated shortcuts across the staged docking corner.
 # Endpoint candidates come from the same source guide, not hand-chosen arm angles.
 metric=np.diag([1.,1.,.25]+[.35]*7)
 dock=int(np.argmax(np.linalg.norm(guide[:,3:]-guide[0,3:],axis=1)>1e-5))-1
 candidates=[];selected_bridge=None
 if dock>0:
  remaining=np.linalg.norm(guide[:dock+1,:2]-guide[-1,:2],axis=1)
  arm_arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(guide[dock:,3:],axis=0),axis=1))]
  for distance in [.3,.6,1.2]:
   i=int(np.argmin(abs(remaining-distance)))
   for fraction in [.35,.55,.8]:
    j=dock+int(np.argmin(abs(arm_arc/max(arm_arc[-1],1e-12)-fraction)))
    points=interpolate_q10(guide[[i,j]],.01,.01);bad=[]
    for k,q in enumerate(points):
     if not check(q) or not query(q)['satisfied']:bad.append(k)
    before=float(np.linalg.norm(np.diff(guide[i:j+1],axis=0)@metric.T,axis=1).sum());after=float(np.linalg.norm(metric@(guide[j]-guide[i])))
    candidates.append(dict(start_index=i,end_index=j,requested_start_distance_m=distance,arm_arc_fraction=fraction,samples=len(points),rejected_samples=bad,predicted_weighted_length_saving=before-after))
  feasible=[c for c in candidates if not c['rejected_samples'] and c['predicted_weighted_length_saving']>0]
  if feasible:
   selected_bridge=max(feasible,key=lambda c:c['predicted_weighted_length_saving']);print('JOINT_BRIDGE',selected_bridge,flush=True)
   grow_guide(guide[[selected_bridge['start_index'],selected_bridge['end_index']]])
 dump('joint_bridge_candidates.json',dict(candidates=candidates,selected=selected_bridge,scope='Source-guide endpoint pairs ranked by weighted chord saving after sampled collision and semantic checks. The selected bridge adds sets, not a forced graph chain.'))
 edges=[]
 for i,r in enumerate(regions):
  for j in range(i+1,len(regions)):
   if np.any(np.array(records[i]['upper'])<records[j]['lower']) or np.any(np.array(records[j]['upper'])<records[i]['lower']):continue
   if r.IntersectsWith(regions[j]):edges.extend([(i,j),(j,i)])
 src=[i for i,r in enumerate(regions) if r.PointInSet(guide[0],1e-7)];dst=[i for i,r in enumerate(regions) if r.PointInSet(guide[-1],1e-7)]
 planner=GcsTrajectoryOptimization(10);cover=planner.AddRegions(regions,edges,order=1,h_min=.01,h_max=30,name='joint_approach_10D');start=planner.AddRegions([Point(guide[0])],order=0,name='start');goal=planner.AddRegions([Point(guide[-1])],order=0,name='goal');planner.AddEdges(start,cover,edges_between_regions=[(0,i) for i in src]);planner.AddEdges(cover,goal,edges_between_regions=[(i,0) for i in dst])
 planner.AddPathLengthCost(metric);planner.AddTimeCost(.05);speed=np.array([.15,.15,.25]+[float(joint['maximum_joint_speed'])]*7);planner.AddVelocityBounds(-speed,speed)
 options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=5;options.rounding_seed=20261102;options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
 tick=time.perf_counter();trajectory,result=planner.SolvePath(start,goal,options);seconds=time.perf_counter()-tick
 if not result.is_success():raise RuntimeError(f'Joint GCS failed: {result.get_solution_result()}')
 points=[];max_residual=0.
 for i in range(trajectory.get_number_of_segments()):
  seg=trajectory.segment(i);a=seg.value(seg.start_time()).ravel();b=seg.value(seg.end_time()).ravel()
  if points and np.max(np.abs(points[-1]-a))>1e-6:raise RuntimeError('Joint GCS segment discontinuity')
  if not any(r.PointInSet(a,1e-6) and r.PointInSet(b,1e-6) for r in regions):raise RuntimeError('Joint GCS segment leaves region union')
  points.extend([a,b])
 points=np.asarray(points);dense=interpolate_q10(points,base_step=.005,joint_step=.005);audits=[]
 for i,q in enumerate(dense):
  v=query(q);audits.append(dict(sample=i,collision_free=check(q),semantic_satisfied=bool(v['satisfied']),semantic_residual_m=float(v['residual_m'])))
 dump('path_audit.json',audits)
 if any(not a['collision_free'] or not a['semantic_satisfied'] for a in audits):
  dump('rejected_path.json',dict(points=points.tolist(),dense=dense.tolist()));raise RuntimeError('Joint path rejected by independent sampled geometry/access audit')
 def metrics(q):
  dq=np.diff(q,axis=0);db=np.linalg.norm(dq[:,:2],axis=1);da=np.linalg.norm(dq[:,3:],axis=1)
  extended=np.linalg.norm((q[:-1,3:]+q[1:,3:])/2-q[0,3:],axis=1)>.05
  return dict(base_length_m=float(db.sum()),arm_travel_rad=float(da.sum()),weighted_length=float(np.linalg.norm(dq@metric.T,axis=1).sum()),simultaneous_intervals=int(np.sum((db>1e-6)&(da>1e-6))),base_distance_while_arm_departure_exceeds_005rad_m=float(db[extended].sum()))
 report=dict(dimension=10,region_count=len(regions),edge_count=len(edges),solver='Clarabel',solve_seconds=seconds,prepare_seconds=tick-began,total_seconds=time.perf_counter()-began,trajectory_duration_s=float(trajectory.end_time()-trajectory.start_time()),joint=metrics(points),staged=metrics(guide),endpoint_residual=float(max(np.max(np.abs(points[0]-guide[0])),np.max(np.abs(points[-1]-guide[-1])))),path_samples=len(dense),all_sampled_checks_passed=True,discarded_region_count=len(rejected),scope='One combined empty-hand approach and reach query. All 10 variables optimized together within guide-local IRIS sets. Not whole-task joint optimization or continuous-safety proof.')
 dump('result.json',report);dump('path.json',dict(points=points.tolist(),dense=dense.tolist(),source_guide=guide.tolist()));return dense,report
