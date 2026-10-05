"""Fixed finite placement configurations, shared by every solver baseline."""
import argparse,copy,json,time,shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .semantic_candidate_gcs import read_library,q,PICK_PHASES,PLACE_PHASES
from .prepare_full_task_development import change_place
from .online_carry_iris import attach_carry_endpoints
from .online_carry_bridge import bridge
from .benchmark_semantic_baselines import atomic_json,file_hash


def prepare_grasp(job):
    root,name,index=job;root=Path(root);config,library=read_library(Path('outputs/results/semantic_candidates/library'));original=library[name]
    out=root/name;out.mkdir();begin=time.perf_counter();candidate=copy.deepcopy(original)
    candidate['cover']['regions']=[r for r in candidate['cover']['regions'] if not r['independent_holdout_audit']['collisions']]
    regions=candidate['cover']['regions'];anchor=[r for r in original['rows'] if r['phase'] in PICK_PHASES][-1]
    goals={};records=[]
    for i,(x,y,h) in enumerate(( (x,y,h) for x in [-2.55,-2.35,-2.15] for y in [1.10,1.25,1.40] for h in [.22,.14,.30])):
        key=f'g{i:02d}';c=copy.deepcopy(config);base=[x,y,float(np.pi/2)];c['delivery_task']['preplace_height']=h;c['delivery_task']['transfer_path'][-1]=base
        r={'goal':key,'base':base,'height':h,'primitive_valid':False}
        try:goal=change_place(c,candidate,placement_base=base)
        except RuntimeError as e:r['failure']=str(e)
        else:
            r['primitive_valid']=True;goal['base']=base;goal['height']=h
            # Store the entire independently checked primitive; do not decide
            # which goal is best during preparation.
            endpoint=q([p for p in goal['rows'] if p['phase'] in PLACE_PHASES][0])
            regions,online=attach_carry_endpoints(c,regions,anchor,q(anchor),endpoint,seed_base=320000+1000*index+10*i)
            regions,bridging=bridge(c,regions,anchor,q(anchor),endpoint,seed_base=420000+1000*index+10*i,max_calls=2)
            r.update(online=online,bridge=bridging);goal.pop('cover');goals[key]=goal
        records.append(r);atomic_json(out/'progress.json',{'attempts':records,'seconds':time.perf_counter()-begin});print('GOAL',name,key,r['primitive_valid'],flush=True)
    candidate['cover']['regions']=regions
    result={'candidate':candidate,'goals':goals,'attempts':records,'seconds':time.perf_counter()-begin}
    atomic_json(out/'library.json',result);return name,result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output
    if root.exists():raise ValueError('New output directory required')
    root.mkdir(parents=True);config,library=read_library(Path('outputs/results/semantic_candidates/library'))
    sources=list(Path('src').glob('*.py'));resources=list(Path('outputs/results/semantic_candidates/library').rglob('*.json'))+[Path('outputs/results/semantic_joint_gcs/shared_model.json')]
    protocol={'bases':{'x':[-2.55,-2.35,-2.15],'y':[1.10,1.25,1.40],'yaw':'pi/2'},'heights':[.22,.14,.30],
        'scope':'New finite goal-library task: original cup target unchanged, docking base selected within task-allowed regions. All methods share every accepted primitive and all regions; not compared to fixed-base old tasks.',
        'preparation':'All 108 grasp/base/height combinations attempted; endpoint IRIS then at most2 bridge calls per primitive; deterministic order; no outcome-based goal pruning.',
        'hashes':{str(f):file_hash(f) for f in sources+resources}}
    atomic_json(root/'protocol.json',protocol);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    results={}
    with ProcessPoolExecutor(4) as pool:
        for name,r in pool.map(prepare_grasp,[(str(root),n,i) for i,n in enumerate(library)]):results[name]=r
    disks=json.loads(Path('outputs/results/semantic_joint_gcs/shared_model.json').read_text())['empty_cover']
    atomic_json(root/'shared.json',{'config':config,'disks':disks,'candidates':{n:r['candidate'] for n,r in results.items()},'goals':{n:r['goals'] for n,r in results.items()}})
    atomic_json(root/'complete.json',{'grasps':len(results),'primitive_attempts':sum(len(r['attempts']) for r in results.values()),'valid_goals':{n:len(r['goals']) for n,r in results.items()},'changed_inputs':[f for f,h in protocol['hashes'].items() if file_hash(f)!=h]})


if __name__=='__main__':main()
