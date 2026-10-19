"""Diagnostic hybrid: official OMPL guide -> official IRIS-NP2 -> GCS.

Does not attribute RRT's route discovery to GCS. The scene, task and grasp
endpoint are unchanged; every polyline corner is retained by corridor stride.
"""
import argparse,json,time,subprocess,sys
from pathlib import Path
import numpy as np
from .build_scene import build_scene
from .green_room_iris import grow_cover
from .green_room_gcs import region_holdout
from .compare_frozen_operation_graph import compare


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--guide',required=True);p.add_argument('--padding',type=float);a=p.parse_args()
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    source=Path('outputs/results/semantic_repair_final_20261019/full_motion/candidate_296')
    config=json.loads((source/'scene_config_snapshot.json').read_text());report=json.loads((source/'result.json').read_text());guide=json.loads(Path(a.guide).read_text())
    if a.padding is not None:config['green_room']['corridor_padding_rad']=a.padding
    if not guide['numerically_valid']:raise RuntimeError('Rejected OMPL guide')
    stride=int(config['green_room']['corridor_stride']);points=[]
    for seg in guide['segments']:
        start=np.array(seg['start']);goal=np.array(seg['end'])
        n=max(stride,int(np.ceil(np.max(abs(goal-start))/.01/stride))*stride)
        points.extend((1-t)*start+t*goal for t in np.linspace(0,1,n+1)[:-1])
    points.append(np.array(guide['segments'][-1]['end']));reference=np.array(points)
    scene=build_scene(config=config,visualize=False);base=np.array(report['dock_q']);began=time.perf_counter()
    regions,records,_=grow_cover(scene,base,reference,scene.gripper_open_positions,semantic=True)
    audit=region_holdout(scene,base,records,scene.gripper_open_positions)
    (root/'region_audit.json').write_text(json.dumps(audit,indent=2))
    if any(audit[k] for k in ('collisions','raw_grasp_violations','semantic_violations')):raise RuntimeError('IRIS holdout rejected hybrid cover')
    lo=np.array([r['lower'] for r in records]);hi=np.array([r['upper'] for r in records]);edges=[]
    for i,r in enumerate(regions):
        candidates=np.flatnonzero(np.all(lo<=hi[i]+1e-9,axis=1)&np.all(hi>=lo[i]-1e-9,axis=1))
        for j in candidates:
            if j>i and r.IntersectsWith(regions[j]):edges.extend([(i,int(j)),(int(j),i)])
    data={'region_method':'iris_np2','regions':records,'edges':edges,
          'sources':[i for i,r in enumerate(regions) if r.PointInSet(reference[0],1e-7)],
          'targets':[i for i,r in enumerate(regions) if r.PointInSet(reference[-1],1e-7)]}
    (root/'reach_cover.json').write_text(json.dumps(data));(root/'scene_config_snapshot.json').write_text(json.dumps(config))
    cover_time=time.perf_counter()-began
    result=compare(root,'reach',methods=('gcs',));result['query_id']=0
    (root/'comparison.json').write_text(json.dumps([result],indent=2))
    subprocess.run([sys.executable,'-m','src.audit_sampling_comparison','--ompl','outputs/results/ompl_room_fine_20261019','--hybrid',str(root)],check=True)
    strict=json.loads(Path('outputs/results/ompl_room_fine_20261019/common_hybrid_audit.json').read_text())
    (root/'strict_path_audit.json').write_text(json.dumps(strict,indent=2))
    if not all(x['accepted'] for x in strict):raise RuntimeError('Strict collision/semantic path audit rejected hybrid; no physics execution authorized by this result')
    (root/'summary.json').write_text(json.dumps({'guide':a.guide,'region_count':len(regions),'cover_seconds':cover_time,'scope':'RRT supplies route discovery; IRIS and GCS refine the corridor; not a GCS-only success'},indent=2))

if __name__=='__main__':main()
