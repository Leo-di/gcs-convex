"""Preregister real placement-base changes; keep all primitive failures."""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .semantic_candidate_gcs import read_library, q, PLACE_PHASES, PICK_PHASES
from .prepare_full_task_development import change_place
from .online_carry_iris import sets


def prepare_case(job):
    root,cell=job;root=Path(root);began=time.perf_counter()
    config,library=read_library(Path('outputs/results/semantic_candidates/library'))
    config['delivery_task']['transfer_path'][-1]=cell['placement_base']
    config['delivery_task']['place_cup_center']=(np.asarray(config['delivery_task']['place_cup_center'])+cell['target_shift']).tolist()
    if 'sites' in cell:config['inspection_sites']=cell['sites']
    candidates={};rejected={};removed={};coverage={}
    for name,original in library.items():
        candidate=copy.deepcopy(original)
        removed[name]=[i for i,r in enumerate(candidate['cover']['regions']) if r['independent_holdout_audit']['collisions']]
        candidate['cover']['regions']=[r for r in candidate['cover']['regions'] if not r['independent_holdout_audit']['collisions']]
        try:candidate=change_place(config,candidate,placement_base=cell['placement_base'])
        except RuntimeError as error:rejected[name]=str(error);continue
        candidates[name]=candidate
        goal=q([r for r in candidate['rows'] if r['phase'] in PLACE_PHASES][0])
        start=q([r for r in candidate['rows'] if r['phase'] in PICK_PHASES][-1])
        regions=sets(candidate['cover']['regions'])
        coverage[name]={'start_covered':any(r.PointInSet(start) for r in regions),
                        'goal_covered':any(r.PointInSet(goal) for r in regions),
                        'goal':goal.tolist()}
    disks=json.loads(Path('outputs/results/semantic_joint_gcs/shared_model.json').read_text())['empty_cover']
    bundle={'cell':cell,'config':config,'candidates':candidates,'disks':disks,
            'primitive_rejections':rejected,'known_colliding_cached_regions_removed':removed,
            'preparation_seconds':time.perf_counter()-began,'endpoint_coverage':coverage}
    (root/'inputs'/(cell['id']+'.json')).write_text(json.dumps(bundle))
    return {'cell':cell,'valid_candidates':list(candidates),'primitive_rejections':rejected,
            'coverage':coverage,'preparation_seconds':bundle['preparation_seconds']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluation',action='store_true');p.add_argument('--design',type=Path)
    args=p.parse_args();root=args.output
    if root.exists():raise ValueError('Use a new output directory')
    (root/'inputs').mkdir(parents=True)
    names=['x110','x115','x120','x125']
    if args.evaluation:
        design=json.loads(args.design.read_text());cells=design['evaluation_cases']
    else:
        cells=[{'id':f'grid_{i:02d}','allowed':names,'placement_base':[float(x),float(y),float(np.pi/2)],'target_shift':[0,0,0]}
               for i,(x,y) in enumerate(( (x,y) for x in [-2.65,-2.45,-2.25,-2.05] for y in [1.,1.12,1.24,1.36]))]
        rng=np.random.default_rng(20261001);evaluation=[]
        for i in range(40):
            base=[float(rng.uniform(-2.65,-2.05)),float(rng.uniform(1.,1.36)),float(np.pi/2)]
            shift=[float(rng.uniform(-.04,.04)),float(rng.uniform(-.02,.02)),0.]
            c={'id':f'random_{i:02d}','allowed':names.copy(),'placement_base':base,'target_shift':shift}
            # Balanced task policies are fixed before either diagnostic or evaluation.
            if i%4==1:c['allowed']=names[1:]
            if i%4==2:c['sites']=[{'name':'south','xy':[-2.5,-1.]}]
            if i%4==3:c['sites']=[{'name':'south','xy':[-2.5,-1.]},{'name':'north','xy':[-2.5,1.]}]
            evaluation.append(c)
        design={'diagnostic_cases':cells,'evaluation_cases':evaluation,'rng_seed':20261001,
                'distribution':'Uniform placement base x[-2.65,-2.05], y[1.0,1.36], yaw pi/2; cup target offsets x +/-4cm,y +/-2cm. Four balanced task policies.',
                'scope':'One unchanged scene and four fixed grasps; no withheld regions, no selecting cases by planner outcome. Cases drawn before diagnostic results.',
                'rules':'All 40 evaluation tasks retained including no-primitive cases; same methods/budgets as prior development; no tuning after evaluation begins.'}
    design['preparation_source_sha256']={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [Path(__file__),Path('src/prepare_full_task_development.py'),Path('src/grasp_planner.py')]}
    (root/'design.json').write_text(json.dumps(design,indent=2));(root/'development_cases.json').write_text(json.dumps(cells,indent=2))
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        for row in pool.map(prepare_case,[(str(root),c) for c in cells]):
            rows.append(row);(root/'preparation_summary.json').write_text(json.dumps(rows,indent=2))
            print(row['cell']['id'],'valid',len(row['valid_candidates']),'uncovered',sum(not c['goal_covered'] for c in row['coverage'].values()),flush=True)


if __name__=='__main__':main()
