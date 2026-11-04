"""Fresh connected-component audit of measured states, independent of route cache."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.ndimage import label
from pydrake.math import RigidTransform
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from src.build_scene import build_scene,set_robot_configuration
from src.human_passage import HumanPassageEvaluator,capsule_box_distances


def verify(root,output):
    cfg=json.loads((root/'config.json').read_text());physics=json.loads((root/'physics.json').read_text())
    result=json.loads((root/'result.json').read_text());scene=build_scene(config=cfg,visualize=False)
    h=HumanPassageEvaluator(scene,force=True);bad=[];samples=[];switches=[];last_branch=None;began=time.perf_counter()
    for i,row in enumerate(physics['trace']):
        set_robot_configuration(scene,row['base_q'],row['arm_q'],row['gripper_q'])
        for key,obj in row['objects'].items():scene.plant.SetFreeBodyPose(scene.plant_context,scene.object_bodies[key],RigidTransform(np.asarray(obj['pose'])))
        lo,hi=h.bounds(h.dynamic)
        d=capsule_box_distances(h.points,lo,hi,h.zlo,h.zhi)-h.required
        free=h.static_free & (d.min(axis=1)>=0)
        labels,_=label(free.reshape(h.shape),structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
        labels=labels.ravel();connected=bool(labels[h.start] and labels[h.start]==labels[h.goal])
        if not connected:bad.append(dict(index=i,time_s=row['time_s'],phase=row['phase'],base_q=row['base_q']))
        # Witness output supports visual inspection; acceptance above uses fresh components.
        if i%10==0 or i==len(physics['trace'])-1:
            witness=h.current();assert witness['satisfied']==connected
            branch=None
            if connected:
                path=np.asarray(witness['path_xy']);cross=path[abs(path[:,0]-1.9)<.05]
                branch='north' if len(cross) and np.mean(cross[:,1])>-.75 else 'south'
                indices=np.array([np.argmin(np.sum((h.points-p)**2,axis=1)) for p in path])
                assert np.all(free[indices])
                assert np.all(np.abs(np.diff(np.column_stack(np.divmod(indices,h.shape[1])),axis=0)).sum(axis=1)==1)
            if branch!=last_branch:
                switches.append(dict(time_s=row['time_s'],phase=row['phase'],route=branch));last_branch=branch
            samples.append(dict(time_s=row['time_s'],phase=row['phase'],connected=connected,branch=branch,path_xy=witness['path_xy'],base_q=row['base_q']))
        if i%10000==0:print(root.name,i,flush=True)
    ranges=[]
    for b in bad:
        if ranges and b['index']==ranges[-1]['last_index']+1:ranges[-1].update(last_index=b['index'],end_s=b['time_s'])
        else:ranges.append(dict(first_index=b['index'],last_index=b['index'],start_s=b['time_s'],end_s=b['time_s'],phase=b['phase']))
    report=dict(task_satisfied=result['satisfied'],human_access_satisfied=not bad,measured_samples=len(physics['trace']),violations=len(bad),outage_ranges=ranges,route_changes=switches,
        unexpected_contacts=result.get('unexpected_contacts'),actions=result.get('actions'),model=cfg['human_passage'],full_cell_guard_m=h.guard,
        trace_sha256=hashlib.sha256((root/'physics.json').read_bytes()).hexdigest(),audit_wall_seconds=time.perf_counter()-began,
        audit='Fresh SciPy 4-connected components at every saved measured sample; same conservative capsule/body-AABB model; cached witnesses checked separately',
        scope='One development layout, fixed furniture, upright capsule, sampled time; no pedestrian prediction, continuous-time guarantee, completeness, or general benchmark conclusion')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2))
    output.with_name(output.stem+'_witnesses.json').write_text(json.dumps(samples))
    print(json.dumps({k:v for k,v in report.items() if k not in ('actions','model','outage_ranges')},indent=2))
    if cfg['human_passage']['enforce']:assert not bad and result['satisfied']
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.run,a.output)
