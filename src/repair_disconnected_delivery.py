"""Post-evaluation repair of the two disconnected cases; not a new benchmark."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace
from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES,build_and_solve,expand
from .online_carry_bridge import bridge
from .benchmark_full_task_development import contact_worker


def run_case(job):
    source,root,cell=map(str,job);source=Path(source);root=Path(root)
    bundle=json.loads((source/'inputs'/(cell+'.json')).read_text())
    old=json.loads((source/cell/'product_online_iris/record.json').read_text())
    config=bundle['config'];candidates=copy.deepcopy(bundle['candidates']);reports={}
    out=root/cell/'product_online_bridge';out.mkdir(parents=True);begin=time.perf_counter()
    record={'cell':cell,'method':'product_online_bridge','planning_success':False,'end_to_end_success':False,'status':'bridging'}
    def save():
        record['bridge']=reports;(out/'record.json').write_text(json.dumps(record,indent=2))
    save()
    for j,name in enumerate(bundle['cell']['allowed']):
        if name not in candidates:continue
        candidate=candidates[name]
        records=candidate['cover']['regions']+[a for a in old['online'][name]['attempts'] if a['accepted']]
        rows=candidate['rows'];pick=[r for r in rows if r['phase'] in PICK_PHASES];place=[r for r in rows if r['phase'] in PLACE_PHASES]
        grown,report=bridge(config,records,pick[-1],q(pick[-1]),q(place[0]),
                            seed_base=150000+100*['random_02','random_22'].index(cell)+10*j,max_calls=4)
        candidate['cover']['regions']=grown;reports[name]=report;save()
        print(cell,name,'calls',len(report['attempts']),'connected',report['after']['region_graph_connected'],flush=True)
    (out/'regions.json').write_text(json.dumps({n:c['cover']['regions'] for n,c in candidates.items()}))
    result,path=build_and_solve(bundle['disks'],candidates,bundle['cell']['allowed'],config,rounding_seed=0,max_rounded_paths=10)
    (out/'gcs.json').write_text(json.dumps(result))
    if path is None:
        record.update(status='failed',failure='bridge_or_gcs_failed',planning_wall_seconds=time.perf_counter()-begin);save();return record
    audit=expand(path,candidates,bundle['disks'],config,out/'reference',visualize=False)
    used=[v['region'] for v in path if v['kind']=='carry' and candidates[v['candidate']]['cover']['regions'][v['region']].get('online_bridge')]
    record.update(audit=audit,planning_success=audit['accepted'],selected_candidate=result['selected_candidate'],
                  used_bridge_regions=used,reference=str(out/'reference'),planning_wall_seconds=time.perf_counter()-begin,
                  status='planned' if audit['accepted'] else 'failed')
    if not audit['accepted']:record['failure']='trajectory_audit_failed'
    save()
    if record['planning_success']:
        contact_worker(SimpleNamespace(output=root,cell=cell,method='product_online_bridge'))
        record=json.loads((out/'record.json').read_text())
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=Path('outputs/results/full_task_evaluation_40_v1'))
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise ValueError('Use a new output directory')
    args.output.mkdir(parents=True)
    files=[Path(__file__),Path('src/online_carry_bridge.py'),Path('src/online_carry_iris.py')]
    protocol={'cases':['random_02','random_22'],'max_calls_per_candidate':4,
        'scope':'Post-evaluation diagnostic repair, not held-out evidence. Reuses prior endpoint results; all original task constraints and audits unchanged. QP seed heuristic only; final path solved by actual semantic GCS.',
        'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    import shutil
    (args.output/'source_snapshot').mkdir()
    for f in files:shutil.copy2(f,args.output/'source_snapshot'/f.name)
    with ProcessPoolExecutor(2) as pool:records=list(pool.map(run_case,[(args.source,args.output,c) for c in protocol['cases']]))
    (args.output/'summary.json').write_text(json.dumps(records,indent=2))
    print([(r['cell'],r['planning_success'],r['end_to_end_success']) for r in records],flush=True)


if __name__=='__main__':main()
