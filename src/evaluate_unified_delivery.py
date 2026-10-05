"""Frozen regression plus 40 fresh paired tasks, same budgets and acceptance."""
import argparse
from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np
from .benchmark_semantic_baselines import atomic_json,file_hash


def worker(args):
    from .benchmark_full_task_development import plan_worker,contact_worker
    root=args.output
    if args.stage=='contact':contact_worker(args);return
    design=json.loads((root/'protocol.json').read_text())
    for f,h in design['hashes'].items():
        if file_hash(f)!=h:raise RuntimeError('Frozen input changed: '+f)
    if args.method=='product_online_iris':plan_worker(args)
    else:
        from .unified_delivery import run
        bundle=json.loads((root/'inputs'/(args.cell+'.json')).read_text())
        run(bundle,root/args.cell/args.method,design['cell_ids'].index(args.cell))


def execute(root,cell,method,cpu,stage):
    out=root/cell/method;out.mkdir(parents=True,exist_ok=True)
    command=['taskset','-c',str(cpu),sys.executable,'-m','src.evaluate_unified_delivery','--output',str(root),
             '--worker','--cell',cell,'--method',method,'--stage',stage]
    with (out/(stage+'.log')).open('w') as log:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timeout=False
        try:p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    file=out/'record.json';r=json.loads(file.read_text()) if file.exists() else {'cell':cell,'method':method,'planning_success':False,'end_to_end_success':False}
    if timeout or p.returncode:
        r.update(status='timeout' if timeout else 'error',failure=stage+'_timeout' if timeout else stage+'_error',end_to_end_success=False)
        if stage=='plan':r['planning_success']=False
    r[stage+'_return_code']=p.returncode;atomic_json(file,r);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--regression',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method');p.add_argument('--stage',choices=['plan','contact'])
    args=p.parse_args()
    if args.worker:worker(args);return
    root=args.output
    if root.exists():raise ValueError('Use a new output directory')
    root.mkdir(parents=True)
    old=Path('outputs/results/full_task_evaluation_40_v1')
    # Fresh queries fixed before any unified result is seen.
    rng=np.random.default_rng(20261002);names=['x110','x115','x120','x125'];fresh=[]
    for i in range(40):
        c={'id':f'random_{i:02d}','allowed':names.copy(),'placement_base':[float(rng.uniform(-2.65,-2.05)),float(rng.uniform(1.,1.36)),float(np.pi/2)],
           'target_shift':[float(rng.uniform(-.04,.04)),float(rng.uniform(-.02,.02)),0.]}
        if i%4==1:c['allowed']=names[1:]
        if i%4==2:c['sites']=[{'name':'south','xy':[-2.5,-1.]}]
        if i%4==3:c['sites']=[{'name':'south','xy':[-2.5,-1.]},{'name':'north','xy':[-2.5,1.]}]
        fresh.append(c)
    atomic_json(root/'fresh_design.json',{'seed':20261002,'cases':fresh,'fixed_n':40,
        'primary_comparison':'unified vs original endpoint-online Product-GCS, paired full-task success; exact two-sided McNemar alpha=.05; no optional stopping or tuning on new cases',
        'rules':'Same scene/distribution/policies as previous evaluation. Old40 regression kept separate; full fixed-base target constraints unchanged.'})
    (root/'inputs').mkdir()
    if args.regression:
        cells=json.loads((old/'development_cases.json').read_text());methods=['unified']
        for f in (old/'inputs').glob('*.json'):shutil.copy2(f,root/'inputs'/f.name)
    else:
        from .diagnose_full_task_coverage import prepare_case
        cells=fresh;methods=['product_online_iris','unified'];preparation=[]
        with ProcessPoolExecutor(4) as pool:
            for r in pool.map(prepare_case,[(str(root),c) for c in cells]):
                preparation.append(r);atomic_json(root/'preparation.json',preparation);print('PREPARED',r['cell']['id'],flush=True)
    atomic_json(root/'cases.json',cells)
    sources=list(Path('src').glob('*.py'));inputs=list((root/'inputs').glob('*.json'))
    library=list(Path('outputs/results/semantic_candidates/library').rglob('*.json'))
    resources=library+[Path('outputs/results/semantic_joint_gcs/shared_model.json')]
    atomic_json(root/'resource_hashes.json',{str(f):file_hash(f) for f in resources})
    cpus=sorted(os.sched_getaffinity(0));cpus=[cpus[i] for i in [4,12,20,28]]
    design={'cell_ids':[c['id'] for c in cells],'methods':methods,'regression':args.regression,'budget_per_stage':300,
            'heights':[.22,.14,.30],'bridge_calls_per_branch_per_height':4,'no_retry_after_physics_failure':True,
            'hashes':{str(f):file_hash(f) for f in sources+inputs+resources},'fresh_design':json.loads((root/'fresh_design.json').read_text()),
            'timing':'Shared nominal primitive preparation separate; fallback preparation and repair included in unified planning budget. Original baseline has no fallback preparation.'}
    atomic_json(root/'protocol.json',design);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    began=time.perf_counter()
    def slot(i):
        for j in range(i,len(cells),4):
            cell=cells[j]['id'];order=methods if j%2==0 else list(reversed(methods))
            for m in order:
                print('START',cell,m,flush=True);r=execute(root,cell,m,cpus[i],'plan')
                if r['planning_success']:r=execute(root,cell,m,cpus[i],'contact')
                print('DONE',cell,m,r['planning_success'],r['end_to_end_success'],r.get('failure',''),flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
    records=[json.loads(f.read_text()) for f in sorted(root.glob('*/*/record.json'))]
    atomic_json(root/'summary.json',{'records':records,'regression':args.regression})
    changed=[f for f,h in design['hashes'].items() if file_hash(f)!=h]
    atomic_json(root/'complete.json',{'seconds':time.perf_counter()-began,'runs':len(records),'changed_inputs':changed})
    print('COMPLETE',[(m,sum(r['end_to_end_success'] for r in records if r['method']==m)) for m in methods],flush=True)


if __name__=='__main__':main()
