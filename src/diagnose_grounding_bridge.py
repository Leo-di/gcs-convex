"""Diagnose frozen coverage disconnection using official OMPL, then GCS."""
import argparse,json,time
from pathlib import Path
import numpy as np
from ompl import base as ob,geometric as og,util as ou
from .build_scene import build_scene
from .scene_spec import load_scene_config
from .scene_slice_gcs import SliceOracle
from .geometry_outbound_gcs import plan_geometry_outbound,make_cover


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--uncovered-only',action='store_true');args=parser.parse_args()
    root=Path('outputs/results/semantic_grounding_choices_20261019')/('bridge_repair_606_uncovered' if args.uncovered_only else 'bridge_repair_606');root.mkdir(exist_ok=False)
    config=load_scene_config('configs/scene_semantic_choice_v6.yaml');scene=build_scene(config=config,visualize=False)
    start=np.array(config['robot']['start']);goal=np.array([1.35,.55,0.]);arm=np.array(config['pick_task']['navigation_arm_q']);spec=config['green_room']
    oracle=SliceOracle(scene,arm,scene.gripper_open_positions,max_distance=.6701)
    boxes=[r for r in config['gcs_base_planning']['regions'] if r['name'] in spec['navigation_regions']]
    def valid(s):
        q=np.array([s[0],s[1]])
        if not any(np.all(q>=b['lower']) and np.all(q<=b['upper']) for b in boxes):return False
        d,self_d,_=oracle.query(q);return d>=.05 and self_d>=0
    ou.RNG.setSeed(20261019);space=ob.RealVectorStateSpace(2);bounds=ob.RealVectorBounds(2)
    for i in range(2):bounds.setLow(i,config['robot']['lower_limits'][i]);bounds.setHigh(i,config['robot']['upper_limits'][i])
    space.setBounds(bounds);space.setLongestValidSegmentFraction(.0025/space.getMaximumExtent())
    si=ob.SpaceInformation(space);si.setStateValidityChecker(valid);si.setup();a=space.allocState();b=space.allocState()
    for i in range(2):a[i]=float(start[i]);b[i]=float(goal[i])
    problem=ob.ProblemDefinition(si);problem.setStartAndGoalStates(a,b);planner=og.RRTConnect(si);planner.setRange(.15);planner.setProblemDefinition(problem);planner.setup()
    tick=time.perf_counter();status=planner.solve(20.);result={'source_candidate':'candidate_606','ompl_status':str(status),'ompl_seconds':time.perf_counter()-tick,'ompl_margin_m':.05,'original_gcs_margin_m':.0201,'scope':'Frozen scene, posture and goal. Extra motion seeds supplied by OMPL; no global infeasibility or GCS-only discovery claim.'}
    (root/'result.json').write_text(json.dumps(result,indent=2))
    if not problem.hasExactSolution():return
    path=problem.getSolutionPath();og.PathSimplifier(si,problem.getGoal()).simplify(path,.5)
    points=np.array([[s[0],s[1]] for s in path.getStates()]);seeds=[]
    for a,b in zip(points,points[1:]):
        n=max(2,int(np.ceil(np.linalg.norm(b-a)/.02))+1);seeds.extend((1-t)*a+t*b for t in np.linspace(0,1,n))
    result['guide']=points.tolist();result['guide_samples']=len(seeds)
    if args.uncovered_only:
        _,records=make_cover(oracle.query,boxes,start[:2],goal[:2],np.array(config['robot']['lower_limits'][:2]),np.array(config['robot']['upper_limits'][:2]),spacing=spec['navigation_spacing_m'],refinement_spacing=spec['navigation_refinement_spacing_m'],refinement_radius_threshold=spec['navigation_refinement_radius_threshold_m'])
        seeds=[p for p in seeds if not any(np.linalg.norm(p-r['center'])<=r['radius'] and np.all(p>=r['lower']) and np.all(p<=r['upper']) for r in records)]
    result['seed_policy']='only guide samples outside original cover' if args.uncovered_only else 'all guide samples'
    result['extra_seeds']=len(seeds)
    (root/'result.json').write_text(json.dumps(result,indent=2))
    plan,report=plan_geometry_outbound(scene,start,goal,allowed_region_names=spec['navigation_regions'],arm_q=arm,spacing=spec['navigation_spacing_m'],prune_contained=True,refinement_spacing=spec['navigation_refinement_spacing_m'],refinement_radius_threshold=spec['navigation_refinement_radius_threshold_m'],extra_seeds=seeds)
    result.update(gcs_bridge_accepted=True,gcs=report,path=plan.configurations.tolist())
    (root/'result.json').write_text(json.dumps(result,indent=2));print(report,flush=True)

if __name__=='__main__':main()
