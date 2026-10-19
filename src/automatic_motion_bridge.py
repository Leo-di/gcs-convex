"""Bounded coverage repair; keeps official GCS and records unsuccessful stages."""
import json,time
from pathlib import Path
import numpy as np


def classify_bridge_failure(reason):
    if 'endpoint_clearance' in reason:return 'endpoint_clearance'
    if 'disconnected' in reason:return 'coverage_disconnected'
    if 'does not cover' in reason:return 'endpoint_not_covered'
    return 'other_solver_or_audit_failure'


def uncovered_samples(points,records):
    return [p for p in points if not any(np.linalg.norm(p-r['center'])<=r['radius'] and np.all(p>=r['lower']) and np.all(p<=r['upper']) for r in records)]


def plan_bridge(scene,start,goal,arm,method,artifact,seed=20261019):
    if method == 'rotation_safe_gcs':
        from .rotation_safe_bridge import plan_rotation_safe_bridge
        return plan_rotation_safe_bridge(scene,start,goal,arm,artifact,seed)
    if method == 'se2_gcs':
        from .geometry_outbound_gcs import plan_geometry_outbound_se2
        record = {'method': method, 'accepted': False, 'events': []}
        path = Path(artifact)
        record['events'].append({'event': 'se2_gcs_started'})
        path.write_text(json.dumps(record, indent=2))
        try:
            plan, report = plan_geometry_outbound_se2(
                scene, np.asarray(start), np.asarray(goal), arm_q=arm,
                allowed_region_names=scene.config['green_room']['navigation_regions'],
                cover_artifact=str(path.with_name('se2_cover.json')),
                **scene.config['green_room'].get('se2_navigation', {}))
        except Exception as e:
            record['events'].append({'event': 'se2_gcs_failed', 'reason': str(e)})
            path.write_text(json.dumps(record, indent=2))
            raise
        record.update(accepted=True, path=plan.configurations.tolist())
        record['events'].append({'event': 'se2_gcs_passed', 'report': report})
        path.write_text(json.dumps(record, indent=2))
        return plan, report
    from .geometry_outbound_gcs import plan_geometry_outbound,make_cover
    from .scene_slice_gcs import SliceOracle
    from .gcs_base_planner import GcsBasePlan
    from ompl import base as ob,geometric as og,util as ou
    if method not in ('fixed_gcs','repair_gcs','repair_ompl'):raise ValueError(method)
    config=scene.config;spec=config['green_room'];start=np.asarray(start);goal=np.asarray(goal)
    oracle=SliceOracle(scene,arm,scene.gripper_open_positions,yaw=float(start[2]),max_distance=.6701)
    boxes=[r for r in config['gcs_base_planning']['regions'] if r['name'] in spec['navigation_regions'] and r['yaw_bounds'][0]<=start[2]<=r['yaw_bounds'][1]]
    if abs(start[2]-goal[2])>1e-10:raise ValueError('Fixed-posture bridge requires fixed yaw')
    out=Path(artifact);began=time.perf_counter();record={'method':method,'seed':seed,'events':[],'accepted':False}
    def save(event,**kwargs):
        record['events'].append({'event':event,'elapsed_seconds':time.perf_counter()-began,**kwargs});out.write_text(json.dumps(record,indent=2))
    args=dict(allowed_region_names=spec['navigation_regions'],arm_q=arm,spacing=spec['navigation_spacing_m'],prune_contained=spec['navigation_prune_contained'],refinement_spacing=spec['navigation_refinement_spacing_m'],refinement_radius_threshold=spec['navigation_refinement_radius_threshold_m'])
    endpoint=[float(oracle.query(q[:2])[0]) for q in (start,goal)]
    save('endpoint_check',distances_m=endpoint,required_margin_m=.0201)
    if min(endpoint)<.0201:
        save('endpoint_clearance_rejected');raise RuntimeError('endpoint_clearance: start/goal violates unchanged 0.0201 m navigation margin')
    try:
        plan,report=plan_geometry_outbound(scene,start,goal,**args)
        record['accepted']=True;save('fixed_gcs_passed',report=report);return plan,report
    except RuntimeError as e:
        category=classify_bridge_failure(str(e));save('fixed_gcs_failed',category=category,reason=str(e))
        if method=='fixed_gcs' or category not in ('coverage_disconnected','endpoint_not_covered'):raise
    # Repair only a representation failure, never an invalid endpoint or audit.
    def valid(s):
        xy=np.array([s[0],s[1]])
        if not any(np.all(xy>=b['lower']) and np.all(xy<=b['upper']) for b in boxes):return False
        d,sd,_=oracle.query(xy);return d>=.0201 and sd>=0
    ou.RNG.setSeed(seed);space=ob.RealVectorStateSpace(2);bounds=ob.RealVectorBounds(2)
    for i in range(2):bounds.setLow(i,config['robot']['lower_limits'][i]);bounds.setHigh(i,config['robot']['upper_limits'][i])
    space.setBounds(bounds);space.setLongestValidSegmentFraction(.0025/space.getMaximumExtent())
    si=ob.SpaceInformation(space);si.setStateValidityChecker(valid);si.setup();a=space.allocState();b=space.allocState()
    for i in range(2):a[i]=float(start[i]);b[i]=float(goal[i])
    problem=ob.ProblemDefinition(si);problem.setStartAndGoalStates(a,b);planner=og.RRTConnect(si);planner.setRange(.15);planner.setProblemDefinition(problem);planner.setup()
    tick=time.perf_counter();status=planner.solve(20.);save('ompl_search',status=str(status),seconds=time.perf_counter()-tick)
    if not problem.hasExactSolution():raise RuntimeError('OMPL bounded search failed; not an infeasibility certificate')
    path=problem.getSolutionPath();og.PathSimplifier(si,problem.getGoal()).simplify(path,.5)
    points=np.array([[s[0],s[1]] for s in path.getStates()]);samples=[]
    for a,b in zip(points,points[1:]):
        samples.extend((1-t)*a+t*b for t in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(b-a)/.0025))+1)))
    if not all(valid(q) for q in samples):raise RuntimeError('OMPL independent dense audit rejected')
    record['guide']=points.tolist();save('ompl_guide_audited',samples=len(samples),minimum_distance_m=min(oracle.query(q)[0] for q in samples))
    if method=='repair_ompl':
        xy=np.array(samples);qs=np.c_[xy,np.full(len(xy),start[2])];length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());seconds=time.perf_counter()-began
        plan=GcsBasePlan(qs,seconds,length/config['gcs_base_planning']['maximum_speed'],length,0.,length,'OMPL RRTConnect + PathSimplifier',len(points)-1,[])
        report={'algorithm':'fixed GCS with OMPL fallback; final fallback trajectory has no GCS optimization','minimum_distance_m':min(oracle.query(q)[0] for q in samples),'audit_samples':len(samples)}
    else:
        _,regions=make_cover(oracle.query,boxes,start[:2],goal[:2],np.array(config['robot']['lower_limits'][:2]),np.array(config['robot']['upper_limits'][:2]),spacing=spec['navigation_spacing_m'],refinement_spacing=spec['navigation_refinement_spacing_m'],refinement_radius_threshold=spec['navigation_refinement_radius_threshold_m'])
        seeds=uncovered_samples(samples[::8]+[samples[-1]],regions);save('coverage_gap_seeds',count=len(seeds),seeds=[p.tolist() for p in seeds])
        plan,report=plan_geometry_outbound(scene,start,goal,extra_seeds=seeds,**args)
    record['accepted']=True;record['path']=plan.configurations.tolist();save('repaired_bridge_passed',report=report);return plan,report
