"""Independent fresh connected-components audit for every measured state and route."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from pydrake.math import RigidTransform
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from src.build_scene import build_scene,set_robot_configuration
from src.human_passage import HumanPassageEvaluator,capsule_box_distances

def verify(root,output):
 cfg=json.loads((root/'config.json').read_text());physics=json.loads((root/'physics.json').read_text());result=json.loads((root/'result.json').read_text())
 s=build_scene(config=cfg,visualize=False);h=HumanPassageEvaluator(s,force=True);routes=[h,*h.connections]
 assert all(np.array_equal(q.points,h.points) and np.array_equal(q.static_free,h.static_free) for q in routes)
 counts={q.spec['id']:0 for q in routes};bad=[];witnesses=[];began=time.monotonic()
 for i,row in enumerate(physics['trace']):
  set_robot_configuration(s,row['base_q'],row['arm_q'],row['gripper_q'])
  for key,item in row['objects'].items():s.plant.SetFreeBodyPose(s.plant_context,s.object_bodies[key],RigidTransform(np.asarray(item['pose'])))
  lo,hi=h.bounds(h.dynamic);free=h.static_free & ((capsule_box_distances(h.points,lo,hi,h.zlo,h.zhi)-h.required).min(axis=1)>=0)
  components,_=label(free.reshape(h.shape),structure=np.array([[0,1,0],[1,1,1],[0,1,0]]));components=components.ravel()
  access={q.spec['id']:bool(components[q.start] and components[q.start]==components[q.goal]) for q in routes}
  for key,connected in access.items():counts[key]+=int(not connected)
  if not all(access.values()):bad.append(dict(index=i,time_s=row['time_s'],phase=row['phase'],failed_connections=[k for k,v in access.items() if not v]))
  if i%10==0 or i==len(physics['trace'])-1:
   recorded={}
   for q in routes:
    r=q._current_single();key=q.spec['id'];assert r['satisfied']==access[key],(i,key)
    if r['satisfied']:
     path=np.asarray(r['path_xy']);indices=np.array([np.argmin(np.sum((h.points-p)**2,axis=1)) for p in path]);assert np.all(free[indices])
     assert np.all(np.abs(np.diff(np.column_stack(np.divmod(indices,h.shape[1])),axis=0)).sum(axis=1)==1)
    recorded[key]=dict(connected=access[key],path_xy=r['path_xy'])
   witnesses.append(dict(time_s=row['time_s'],phase=row['phase'],connected=all(access.values()),connections=recorded,path_xy=recorded[h.spec['id']]['path_xy'],base_q=row['base_q']))
  if i%10000==0:print(root.name,'independent state',i,flush=True)
 ranges=[]
 for b in bad:
  if ranges and b['index']==ranges[-1]['last_index']+1:ranges[-1].update(last_index=b['index'],end_s=b['time_s'])
  else:ranges.append(dict(first_index=b['index'],last_index=b['index'],start_s=b['time_s'],end_s=b['time_s'],phase=b['phase']))
 report=dict(task_satisfied=result['satisfied'],human_access_satisfied=not bad,measured_samples=len(physics['trace']),violations=len(bad),connection_violations=counts,outage_ranges=ranges,
  unexpected_contacts=result['unexpected_contacts'],actions=result['actions'],model=cfg['human_passage'],full_cell_guard_m=h.guard,
  trace_sha256=hashlib.sha256((root/'physics.json').read_bytes()).hexdigest(),audit_wall_seconds=time.monotonic()-began,
  audit='Fresh SciPy four-connected components at every saved physical state; each route checked; path witnesses independently validated',scope='Static people and furniture, conservative capsule/body-AABBs, sampled time; not a walking or continuous-time guarantee')
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2));output.with_name(output.stem+'_witnesses.json').write_text(json.dumps(witnesses))
 if cfg['human_passage']['enforce']:assert not bad and result['satisfied']
 print(root.name,'measured violations',counts,flush=True);return report

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 import scripts.verify_auto_posture as original
 original.verify_passage=verify  # Adapter: replace the old single-route audit, not its acceptance criteria.
 from scripts.service_cafe_artifacts import resolve_pickup
 original.verify(a.run,a.output,pickup_resolver=resolve_pickup)
