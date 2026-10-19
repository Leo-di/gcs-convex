"""Official OMPL RRTConnect on actual 7D geometry and whole-room semantics.

All frozen five queries and three seeds are retained. The first seed of the
full reach query is prescribed for contact execution, not selected by outcome.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np


def worker(root, query, seed):
    from ompl import base as ob, geometric as og, util as ou
    from .build_scene import build_scene
    from .green_room_gcs import room_oracle
    from .green_room_iris import make_checker
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text())
    source=Path(protocol['source']);qfolder=Path(protocol['queries'])/f'query_{query:02d}'
    out=root/f'query_{query:02d}_seed_{seed}';out.mkdir(exist_ok=True)
    began=time.perf_counter();config=json.loads((source/'scene_config_snapshot.json').read_text())
    result=json.loads((source/'result.json').read_text());cover=json.loads((qfolder/'reach_cover.json').read_text())
    start=np.array(cover['regions'][0]['guide_start']);goal=np.array(cover['regions'][-1]['guide_end']);dock=np.array(result['dock_q'])
    scene=build_scene(config=config,visualize=False);oracle=room_oracle(scene)
    mode,checker,mapping=make_checker(config,dock,start,scene.gripper_open_positions)
    checks=0
    def valid(state):
        nonlocal checks
        checks+=1;q=np.array([state[i] for i in range(7)])
        return bool(checker.CheckConfigCollisionFree(mapping(q)) and oracle.query(np.r_[dock,q],scene.gripper_open_positions)['satisfied'])
    ou.RNG.setSeed(seed)
    space=ob.RealVectorStateSpace(7);bounds=ob.RealVectorBounds(7)
    for i,(lo,hi) in enumerate(zip(config['whole_body_planning']['joint_lower_limits'],config['whole_body_planning']['joint_upper_limits'])):
        bounds.setLow(i,lo);bounds.setHigh(i,hi)
    space.setBounds(bounds)
    # OMPL's built-in discrete motion validator limits Euclidean segment
    # length to 0.005 rad, hence every coordinate step is at most 0.005 rad.
    space.setLongestValidSegmentFraction(protocol['motion_check_step_rad']/space.getMaximumExtent())
    si=ob.SpaceInformation(space);si.setStateValidityChecker(valid);si.setup()
    qs=space.allocState();qg=space.allocState()
    for i in range(7):qs[i]=float(start[i]);qg[i]=float(goal[i])
    problem=ob.ProblemDefinition(si);problem.setStartAndGoalStates(qs,qg)
    planner=og.RRTConnect(si);planner.setRange(.25);planner.setProblemDefinition(problem);planner.setup()
    report={'method':'ompl_rrtconnect','query_id':query,'seed':seed,'setup_seconds':time.perf_counter()-began,
            'ompl_version':importlib.metadata.version('ompl'),'exact_solution':False,'numerically_valid':False}
    tick=time.perf_counter();status=planner.solve(79.);report['solve_seconds']=time.perf_counter()-tick
    report['status']=str(status);report['exact_solution']=bool(problem.hasExactSolution())
    if report['exact_solution']:
        path=problem.getSolutionPath();report['raw_path_length_rad']=float(path.length())
        tick=time.perf_counter();og.PathSimplifier(si,problem.getGoal()).simplify(path,1.);report['simplify_seconds']=time.perf_counter()-tick
        qs=np.array([[s[i] for i in range(7)] for s in path.getStates()])
        speed=float(config['whole_body_planning']['maximum_joint_speed']);segments=[]
        for a,b in zip(qs,qs[1:]):
            duration=max(.01,float(np.max(abs(b-a)))/speed)
            segments.append({'start':a.tolist(),'end':b.tolist(),'duration':duration})
        endpoint=max(float(np.max(abs(qs[0]-start))),float(np.max(abs(qs[-1]-goal))))
        # Finer independent edge audit on the final simplified polyline.
        tick=time.perf_counter();audited=0;violations=0
        for a,b in zip(qs,qs[1:]):
            count=max(2,int(np.ceil(np.max(abs(b-a))/.0025))+1)
            for t in np.linspace(0,1,count):
                q=(1-t)*a+t*b;audited+=1
                if not valid(q):violations+=1
        length=float(np.linalg.norm(np.diff(qs,axis=0),axis=1).sum())
        report.update(segments=segments,path_length_rad=length,duration_s=sum(s['duration'] for s in segments),
                      endpoint_error_rad=endpoint,audit_samples=audited,audit_violations=violations,
                      audit_seconds=time.perf_counter()-tick,numerically_valid=bool(endpoint<1e-6 and violations==0))
    report['validity_calls']=checks;report['total_seconds']=time.perf_counter()-began
    (out/'result.json').write_text(json.dumps(report,indent=2));print({k:v for k,v in report.items() if k!='segments'},flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--worker',action='store_true');p.add_argument('--query',type=int);p.add_argument('--seed',type=int);p.add_argument('--motion-step',type=float,default=.005)
    a=p.parse_args()
    if a.worker:return worker(a.output,a.query,a.seed)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    protocol={'source':'outputs/results/semantic_repair_final_20261019/full_motion/candidate_296',
              'queries':'outputs/results/gcs_multiroute_20261019','query_ids':list(range(5)),'seeds':[100,101,102],
              'planner':'official OMPL RRTConnect 2.0.1','range_rad':.25,'motion_check_step_rad':a.motion_step,
              'final_audit_step_rad':.0025,'planning_budget_s':79,'simplification_budget_s':1,'simplification_time_limit_kind':'soft termination; one shortcut operation may exceed the requested time, actual wall time reported',
              'physical_execution_prespecified':{'query':0,'seed':100},
              'scope':'Same physical scene/semantics and endpoints, full joint bounds; RRT not restricted to GCS guide corridors. Five correlated queries, three seeds, not fifteen independent scenes. Cached GCS results compared separately from cover construction.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def run_case(case):
        q,seed=case;out=root/f'query_{q:02d}_seed_{seed}';out.mkdir()
        with (out/'run.log').open('w') as log:
            try:
                r=subprocess.run([sys.executable,'-m','src.benchmark_ompl_room','--output',str(root),'--worker','--query',str(q),'--seed',str(seed)],stdout=log,stderr=subprocess.STDOUT,timeout=150)
                code=r.returncode
            except subprocess.TimeoutExpired:code='timeout'
        f=out/'result.json'
        result=json.loads(f.read_text()) if f.exists() else {'method':'ompl_rrtconnect','query_id':q,'seed':seed,'numerically_valid':False,'failure':code}
        result['process_exit']=code
        if code!=0:result['numerically_valid']=False
        return result
    cases=[(q,s) for q in protocol['query_ids'] for s in protocol['seeds']]
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=[]
        for result in pool.map(run_case,cases):
            rows.append(result);(root/'comparison.json').write_text(json.dumps(rows,indent=2))
            print(result['query_id'],result['seed'],result['numerically_valid'],result.get('solve_seconds'),flush=True)

if __name__=='__main__':main()
