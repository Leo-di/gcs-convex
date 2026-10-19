"""Fixed-posture SE(2) clearance balls, OMPL guidance, official Drake GCS.

For a rigid robot of horizontal extent R, displacement <= ||delta xy|| +
R |delta yaw| <= sqrt(2) ||diag(1,1,R) delta q||. Thus a ball of scaled
radius (clearance-margin)/sqrt(2) has a model-based clearance lower bound.
Distance computations remain floating point; this is not a hardware proof.
"""
import json,time
from pathlib import Path
import numpy as np
from .build_scene import set_robot_configuration
from .whole_robot_room import shape_support
from .scene_slice_gcs import SliceOracle


def scaled_radius(clearance,margin):
    if not np.isfinite(clearance) or not np.isfinite(margin) or margin<0:
        raise ValueError('Finite clearance and nonnegative margin required')
    return max(0.,float(clearance-margin)/np.sqrt(2))


def rigid_yaw_radius(scene,arm):
    set_robot_configuration(scene,[0.,0.,0.],arm,scene.gripper_open_positions)
    inspector=scene.scene_graph.model_inspector();radii=[]
    for instance in (scene.base_instance,scene.arm_instance,scene.gripper_instance):
        for index in scene.plant.GetBodyIndices(instance):
            body=scene.plant.get_body(index)
            for gid in scene.plant.GetCollisionGeometriesForBody(body):
                shape=inspector.GetShape(gid)
                pose=scene.plant.EvalBodyPoseInWorld(scene.plant_context,body)@inspector.GetPoseInFrame(gid)
                # Norm of the local enclosing box bounds any shape orientation.
                extent=np.linalg.norm([shape_support(shape,e) for e in np.eye(3)])
                radii.append(float(np.linalg.norm(pose.translation()[:2])+extent))
    if not radii:raise ValueError('No collision geometry')
    return max(radii)


def plan_rotation_safe_bridge(scene,start,goal,arm,artifact,seed=20261019):
    from ompl import base as ob,geometric as og,util as ou
    from pydrake.geometry.optimization import Hyperellipsoid,HPolyhedron,Intersection,Point,GraphOfConvexSetsOptions
    from pydrake.planning import GcsTrajectoryOptimization
    from pydrake.solvers import ClarabelSolver
    from .gcs_base_planner import GcsBasePlan,_label_region_sequence
    start=np.asarray(start,float);goal=np.asarray(goal,float);arm=np.asarray(arm,float)
    spec=scene.config['gcs_base_planning'];names=scene.config['green_room']['navigation_regions']
    boxes=[b for b in spec['regions'] if b['name'] in names]
    if len(boxes)!=1:raise ValueError('Current guided bridge requires one convex navigation envelope')
    box=boxes[0];lo=np.maximum([*box['lower'],box['yaw_bounds'][0]],scene.config['robot']['lower_limits']);hi=np.minimum([*box['upper'],box['yaw_bounds'][1]],scene.config['robot']['upper_limits'])
    margin=.0201;guide_margin=.0301;R=rigid_yaw_radius(scene,arm);scale=np.array([1.,1.,R]);oracle=SliceOracle(scene,arm,scene.gripper_open_positions,max_distance=.6701)
    root=Path(artifact).parent;record={'method':'rotation_safe_gcs','accepted':False,'rotation_radius_m':R,'margin_m':margin,'guide_margin_m':guide_margin,'events':[]};began=time.perf_counter()
    def save(event,**kw):
        record['events'].append(dict(event=event,seconds=time.perf_counter()-began,**kw));Path(artifact).write_text(json.dumps(record,indent=2))
    def distance(q):
        oracle.yaw=float(q[2]);d,sd,_=oracle.query(q[:2])
        # A thresholded query with no pair only establishes threshold clearance.
        return min(d,oracle.max_distance),sd
    def valid(z):
        q=np.array([z[i] for i in range(3)])/scale
        if np.any(q<lo) or np.any(q>hi):return False
        d,sd=distance(q);return d>=guide_margin and sd>=0
    save('endpoint_check',distances=[distance(q)[0] for q in [start,goal]])
    if not all(valid(q*scale) for q in [start,goal]):raise RuntimeError('Endpoint violates domain or fixed 3.01 cm guide clearance; no infeasibility claim')
    ou.RNG.setSeed(seed);space=ob.RealVectorStateSpace(3);bounds=ob.RealVectorBounds(3)
    for i in range(3):bounds.setLow(i,float(lo[i]*scale[i]));bounds.setHigh(i,float(hi[i]*scale[i]))
    space.setBounds(bounds);space.setLongestValidSegmentFraction(.002/space.getMaximumExtent());si=ob.SpaceInformation(space);si.setStateValidityChecker(valid);si.setup()
    a=space.allocState();b=space.allocState()
    for i in range(3):a[i]=float(start[i]*scale[i]);b[i]=float(goal[i]*scale[i])
    problem=ob.ProblemDefinition(si);problem.setStartAndGoalStates(a,b);rrt=og.RRTConnect(si);rrt.setRange(.15);rrt.setProblemDefinition(problem);rrt.setup();status=rrt.solve(20.)
    save('ompl_guide_search',status=str(status))
    if not problem.hasExactSolution():raise RuntimeError('Bounded OMPL guide search failed; not infeasibility')
    path=problem.getSolutionPath();og.PathSimplifier(si,problem.getGoal()).simplify(path,.5)
    guide=np.array([[s[i] for i in range(3)] for s in path.getStates()]);samples=[]
    for a,b in zip(guide,guide[1:]):samples.extend((1-t)*a+t*b for t in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(b-a)/.001))+1)))
    regions=[];records=[];domain=HPolyhedron.MakeBox(lo,hi)
    for z in samples:
        if any(np.linalg.norm(z-np.array(r['center'])*scale)<=.7*r['scaled_radius'] for r in records):continue
        q=z/scale;d,sd=distance(q);rho=min(.45,scaled_radius(d,margin))
        if sd<0 or rho<=0:raise RuntimeError('Guide seed has no positive clearance ball')
        regions.append(Intersection(Hyperellipsoid(np.diag(scale/rho),q),domain));records.append({'center':q.tolist(),'scaled_radius':rho,'seed_clearance_m':d,'clearance_lower_bound_m':d-np.sqrt(2)*rho})
    # Endpoint membership must be checked explicitly even when seeds are skipped.
    sources=[i for i,r in enumerate(regions) if r.PointInSet(start,1e-9)];targets=[i for i,r in enumerate(regions) if r.PointInSet(goal,1e-9)]
    if not sources or not targets:raise RuntimeError('Guided cover lacks endpoint')
    edges=[]
    for i,a in enumerate(records):
        for j,b in enumerate(records[:i]):
            if np.linalg.norm((np.array(a['center'])-b['center'])*scale)>a['scaled_radius']+b['scaled_radius']:continue
            if regions[i].IntersectsWith(regions[j]):edges.extend([(i,j),(j,i)])
    data={'scale':scale.tolist(),'regions':records,'edges':edges,'guide_scaled':guide.tolist(),'start':start.tolist(),'goal':goal.tolist(),'domain_lower':lo.tolist(),'domain_upper':hi.tolist()};(root/'rotation_safe_cover.json').write_text(json.dumps(data,indent=2));save('cover_ready',regions=len(regions),directed_edges=len(edges))
    # Independent boundary probes supplement (do not replace) displacement bound.
    rng=np.random.default_rng(seed);minimum=float('inf');tested=0
    for r in records:
        for _ in range(64):
            u=rng.normal(size=3);u/=np.linalg.norm(u);q=np.array(r['center'])+r['scaled_radius']*u/scale
            if np.any(q<lo) or np.any(q>hi):continue
            d,sd=distance(q);tested+=1;minimum=min(minimum,d)
            if d<margin-1e-7 or sd<-1e-7:raise RuntimeError('Independent convex-region boundary audit rejected')
    save('region_boundary_audit',samples=tested,minimum_distance_m=minimum)
    planner=GcsTrajectoryOptimization(3);free=planner.AddRegions(regions,edges,order=1,h_min=.01,h_max=20,name='rotation_safe')
    src=planner.AddRegions([Point(start)],order=0,name='start');dst=planner.AddRegions([Point(goal)],order=0,name='goal')
    planner.AddEdges(src,free,edges_between_regions=[(0,i) for i in sources]);planner.AddEdges(free,dst,edges_between_regions=[(i,0) for i in targets])
    planner.AddPathLengthCost(np.diag([spec['path_length_weight'],spec['path_length_weight'],spec['yaw_path_weight']]));planner.AddTimeCost(spec['time_weight']);speed=spec['maximum_speed'];ang=spec['maximum_angular_speed'];planner.AddVelocityBounds(np.array([-speed,-speed,-ang]),np.array([speed,speed,ang]))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10;options.rounding_seed=0;options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver();tick=time.perf_counter();traj,result=planner.SolvePath(src,dst,options);solve=time.perf_counter()-tick
    if not result.is_success():raise RuntimeError('Official GCS failed on guided convex cover')
    duration=float(traj.end_time());qs=np.array([traj.value(t).ravel() for t in np.linspace(0,duration,max(1001,int(np.ceil(duration*spec['samples_per_second']))+1))]);qs[0]=start;qs[-1]=goal
    measured=[distance(q) for q in qs];minpath=min(d for d,s in measured)
    if minpath<.02-1e-7 or min(s for d,s in measured)<-1e-7:raise RuntimeError('Final GCS path audit rejected')
    report={'algorithm':'OMPL RRTConnect guidance + rotation-displacement-bounded convex ellipsoids + official Drake GCS','region_count':len(regions),'edge_count':len(edges),'rotation_radius_m':R,'solve_seconds':solve,'total_seconds':time.perf_counter()-began,'minimum_distance_m':minpath,'region_boundary_minimum_m':minimum,'region_boundary_samples':tested,'audit_samples':len(qs),'scope':'Fixed arm/gripper, static world, finite yaw chart. Model-based displacement bound plus numerical audits; not IRIS navigation, not hardware or interval-arithmetic proof. Guide restricts searched coverage; no global completeness or optimality claim.'}
    record['accepted']=True;record['path']=qs.tolist();save('gcs_passed',report=report)
    return GcsBasePlan(qs,solve,duration,float(np.linalg.norm(np.diff(qs[:,:2],axis=0),axis=1).sum()),float(np.abs(np.diff(qs[:,2])).sum()),float(result.get_optimal_cost()),result.get_solver_id().name(),traj.get_number_of_segments(),_label_region_sequence(qs,boxes)),report
