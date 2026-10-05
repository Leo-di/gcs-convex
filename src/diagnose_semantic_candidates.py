"""Replay every rejected proposal without relaxing feasibility thresholds."""
import copy,json,time,argparse
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json
from .build_scene import build_scene
from .animate_delivery_demo import plan_delivery_candidate
from .grasp_planner import plan_pick

def main():
 p=argparse.ArgumentParser();p.add_argument('scene');a=p.parse_args()
 source=Path('outputs/results/semantic_multiscene_v1/libraries')/a.scene
 out=Path('outputs/results/semantic_candidate_diagnosis')/a.scene;out.mkdir(parents=True,exist_ok=True)
 config=json.loads((source/'config.json').read_text());prior=json.loads((source/'report.json').read_text());rows=[]
 for old in prior['attempts']:
  if old['status']=='accepted':continue
  row={'name':old['name'],'base_q':old['base_q'],'original_stage':old['stage']};tick=time.perf_counter()
  try:
   data=plan_delivery_candidate(build_scene(config=copy.deepcopy(config),visualize=False),np.array(old['base_q']),row)
   row['status']='accepted_on_replay'
  except Exception as e:row.update(status='rejected',reason=str(e),details=getattr(e,'details',{}))
  # Controlled diagnostic: only initial guesses change, not constraints. For
  # proximity rejects, isolated IK is a reachability probe, not task acceptance.
  if old['stage'] in ['pickup_ik','pickup_proximity']:
   probe=copy.deepcopy(config);probe.setdefault('ik_solver',{})['max_restarts']=12
   try:
    plan=plan_pick(build_scene(config=probe,visualize=False),np.array(old['base_q']))
    row['ik_probe']={'status':'success','pregrasp_q':plan.pregrasp_q.tolist(),'grasp_q':plan.grasp_q.tolist(),'lift_q':plan.lift_q.tolist()}
   except Exception as e:row['ik_probe']={'status':'failed','reason':str(e),'details':getattr(e,'details',{})}
  row['seconds']=time.perf_counter()-tick;rows.append(row);atomic_json(out/'report.json',{'scene':a.scene,'attempts':rows});print(a.scene,row['name'],row['status'],flush=True)
if __name__=='__main__':main()
