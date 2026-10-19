"""Apply the same fine actual-geometry audit to cached GCS and OMPL paths."""
import argparse,json
from pathlib import Path
import numpy as np
from .build_scene import build_scene
from .green_room_gcs import room_oracle
from .green_room_iris import make_checker


def main():
    p=argparse.ArgumentParser();p.add_argument('--ompl',required=True);p.add_argument('--hybrid');a=p.parse_args();root=Path(a.ompl)
    protocol=json.loads((root/'protocol.json').read_text());source=Path(protocol['source'])
    config=json.loads((source/'scene_config_snapshot.json').read_text());r=json.loads((source/'result.json').read_text());dock=np.array(r['dock_q'])
    scene=build_scene(config=config,visualize=False);oracle=room_oracle(scene)
    mode,checker,mapping=make_checker(config,dock,np.array(r['fixed_tucked_arm_q']),scene.gripper_open_positions)
    cases=json.loads((root/'comparison.json').read_text())
    for row in json.loads((Path(protocol['queries'])/'comparison.json').read_text()):
        cases.append(dict(next(r for r in row['results'] if r['method']=='gcs'),query_id=row['query_id'],seed=None))
    if a.hybrid:
        item=json.loads((Path(a.hybrid)/'comparison.json').read_text())[0]['results'][0]
        cases=[dict(item,method='rrt_guided_iris_gcs',query_id=0,seed=None)]
    results=[]
    for r in cases:
        bad=[];count=0
        for j,segment in enumerate(r.get('segments',[])):
            x=np.array(segment['start']);y=np.array(segment['end']);n=max(2,int(np.ceil(np.max(abs(y-x))/.00125))+1)
            for t in np.linspace(0,1,n):
                q=(1-t)*x+t*y;count+=1
                free=checker.CheckConfigCollisionFree(mapping(q));inside=oracle.query(np.r_[dock,q],scene.gripper_open_positions)
                if not free or not inside['satisfied']:
                    clearance=checker.CalcRobotClearance(mapping(q),.005)
                    pairs=[{'a':mode.plant.get_body(i).name(),'b':mode.plant.get_body(j).name(),'distance_m':float(d)}
                           for i,j,d in zip(clearance.robot_indices(),clearance.other_indices(),clearance.distances()) if d<0.]
                    bad.append({'segment':j,'alpha':float(t),'q':q.tolist(),'collision_free':bool(free),'room_residual_m':inside['residual_m'],'negative_pairs':pairs})
        results.append({'method':r['method'],'query_id':r['query_id'],'seed':r.get('seed'),'samples':count,'violations':len(bad),'counterexamples':bad[:10],'accepted':bool(count and not bad)})
        (root/('common_hybrid_audit.json' if a.hybrid else 'common_fine_audit.json')).write_text(json.dumps(results,indent=2))
        print(r['method'],r['query_id'],r.get('seed'),len(bad),flush=True)

if __name__=='__main__':main()
