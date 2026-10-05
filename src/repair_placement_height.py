"""Bounded alternative preplace-height primitives on all 27 preparation failures.

This is a post-evaluation action-library diagnostic, not held-out evaluation.
Final cup target, base pose, grasp, and collision/pose tolerances stay fixed.
"""
import argparse
import copy
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
import time
from .semantic_candidate_gcs import read_library,q,PICK_PHASES,PLACE_PHASES,build_and_solve,expand
from .prepare_full_task_development import change_place
from .online_carry_iris import attach_carry_endpoints
from .benchmark_full_task_development import contact_worker


def worker(job):
    source,root,cell=job;source=Path(source);root=Path(root)
    original=json.loads((source/'inputs'/(cell+'.json')).read_text())
    config=copy.deepcopy(original['config']);_,library=read_library(Path('outputs/results/semantic_candidates/library'))
    out=root/cell/'height_online';out.mkdir(parents=True);started=time.perf_counter();attempts=[];selected={}
    for height in [.14,.30]:
        c=copy.deepcopy(config);c['delivery_task']['preplace_height']=height;candidates={};rejected={}
        for name in original['cell']['allowed']:
            candidate=copy.deepcopy(library[name])
            candidate['cover']['regions']=[r for r in candidate['cover']['regions'] if not r['independent_holdout_audit']['collisions']]
            try:candidates[name]=change_place(c,candidate,placement_base=original['cell']['placement_base'])
            except RuntimeError as e:rejected[name]=str(e)
        attempts.append({'preplace_height':height,'valid_candidates':list(candidates),'rejections':rejected})
        if candidates:selected=candidates;config=c;break
    record={'cell':cell,'method':'height_online','planning_success':False,'end_to_end_success':False,
            'height_attempts':attempts,'status':'preparing','online':{},'preplace_height':config['delivery_task']['preplace_height']}
    def save():
        record['planning_wall_seconds']=time.perf_counter()-started
        (out/'record.json').write_text(json.dumps(record,indent=2))
    if not selected:
        record.update(status='failed',failure='alternative_height_primitives_failed');save();return record
    # Save the exact alternative library; no changed base/target or constraint relaxation.
    (out/'input.json').write_text(json.dumps({'config':config,'candidates':selected,'cell':original['cell']}))
    for j,(name,candidate) in enumerate(selected.items()):
        rows=candidate['rows'];pick=[r for r in rows if r['phase'] in PICK_PHASES];place=[r for r in rows if r['phase'] in PLACE_PHASES]
        regions,report=attach_carry_endpoints(config,candidate['cover']['regions'],pick[-1],q(pick[-1]),q(place[0]),seed_base=180000+100*int(cell.split('_')[-1])+j*2)
        candidate['cover']['regions']=regions;record['online'][name]=report;save()
    result,path=build_and_solve(original['disks'],selected,list(selected),config,rounding_seed=0,max_rounded_paths=10)
    (out/'gcs.json').write_text(json.dumps(result))
    if path is None:
        record.update(status='failed',failure='online_cover_or_gcs_failed');save();return record
    audit=expand(path,selected,original['disks'],config,out/'reference',visualize=False)
    record.update(audit=audit,planning_success=audit['accepted'],reference=str(out/'reference'),selected_candidate=result['selected_candidate'],status='planned' if audit['accepted'] else 'failed')
    if not audit['accepted']:record['failure']='trajectory_audit_failed'
    save()
    if record['planning_success']:
        contact_worker(SimpleNamespace(output=root,cell=cell,method='height_online'))
        record=json.loads((out/'record.json').read_text())
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('outputs/results/full_task_evaluation_40_v1'));p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    cells=[f.stem for f in sorted((args.source/'inputs').glob('*.json')) if not json.loads(f.read_text())['candidates']]
    (args.output/'protocol.json').write_text(json.dumps({'cases':cells,'heights_m':[.14,.30],
        'scope':'Post-evaluation repair on all 27 no-primitive cases. First height with an admissible candidate; original base, final cup target, grasp, tolerances, audits unchanged. Same official IK/IRIS/GCS, no collision relaxation; no bridge extension in this experiment.'},indent=2))
    import shutil
    (args.output/'source_snapshot').mkdir()
    for f in [Path(__file__),Path('src/prepare_full_task_development.py'),Path('src/grasp_planner.py')]:shutil.copy2(f,args.output/'source_snapshot'/f.name)
    rows=[]
    with ProcessPoolExecutor(4) as pool:
        for r in pool.map(worker,[(args.source,args.output,c) for c in cells]):
            rows.append(r);(args.output/'summary.json').write_text(json.dumps(rows,indent=2))
            print(r['cell'],'height',r['preplace_height'],'plan',r['planning_success'],'task',r['end_to_end_success'],r.get('failure'),flush=True)


if __name__=='__main__':main()
