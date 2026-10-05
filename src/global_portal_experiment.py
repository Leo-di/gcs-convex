"""Paired rerun: one global portal search versus unchanged joint GCS."""
import argparse,copy,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
from .coupled_goal_experiment import worker as original_worker,options,materialize,path_cost,dock_ok
from .semantic_candidate_gcs import expand
PREVIOUS=Path('outputs/results/coupled_goal_evaluation_v1')
METHODS=['joint_gcs','global_portal']


def worker(args):
    if args.method=='joint_gcs' or args.stage=='contact':return original_worker(args)
    root=args.output;out=root/args.cell/args.method;protocol=json.loads((root/'protocol.json').read_text())
    for f,h in protocol['hashes'].items():
        if file_hash(f)!=h:raise RuntimeError('Frozen file changed: '+f)
    began=time.perf_counter();shared=json.loads((root/'shared.json').read_text());cell=next(c for c in protocol['cases'] if c['id']==args.cell)
    config=copy.deepcopy(shared['config'])
    if cell['sites'] is not None:config['inspection_sites']=cell['sites']
    goals=options(shared,cell);r={'cell':args.cell,'method':args.method,'planning_success':False,'end_to_end_success':False,'contact_selected':cell['contact'],'available_pairs':sum(map(len,goals.values())),'attempts':[],'status':'planning'}
    atomic_json(out/'record.json',r)
    from .full_task_portal_socp import prepare
    from .global_portal_socp import solve
    t=time.perf_counter();maps=prepare({n:c for n,c in shared['candidates'].items() if n in cell['allowed']},shared['disks'],5);r['roadmap_seconds']=time.perf_counter()-t
    t=time.perf_counter();result,path=solve(shared['disks'],shared['candidates'],cell['allowed'],config,maps,cell['start'],goals);r['solver_seconds']=time.perf_counter()-t
    attempt=out/'attempt_000';attempt.mkdir(exist_ok=True);atomic_json(attempt/'plan.json',result);entry={'status':result['status'],'solve_wall_seconds':r['solver_seconds']}
    if path is not None:
        name=result['selected_candidate'];key=result['selected_goal'];cs,c=materialize(shared,name,key,config)
        cost=path_cost(path,cs);audit=expand(path,cs,shared['disks'],c,attempt,visualize=False)
        audit['docking_satisfied']=dock_ok(cs[name]['rows'][-1]['base_q'],cell['docking'],1e-6);audit['accepted']=bool(audit['accepted'] and audit['docking_satisfied']);atomic_json(attempt/'audit.json',audit)
        entry.update(cost=cost,audited=True,accepted=audit['accepted'])
        r.update(planning_success=audit['accepted'],reference=str(attempt),cost=cost,selected_candidate=name,selected_goal=key,selected_base=cs[name]['base'],selected_height=cs[name]['height'])
    r.update(attempts=[entry],enumeration_complete=True,planning_wall_seconds=time.perf_counter()-began,status='planned' if r['planning_success'] else 'failed')
    if not r['planning_success']:r['failure']='global_route_or_audit_failed'
    atomic_json(out/'record.json',r)


def execute(root,cell,method,cpu,stage):
    out=root/cell/method;out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter();timeout=False
    with (out/(stage+'.log')).open('w') as log:
        p=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-m','src.global_portal_experiment','--output',str(root),'--worker','--cell',cell,'--method',method,'--stage',stage],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    f=out/'record.json';r=json.loads(f.read_text()) if f.exists() else {'cell':cell,'method':method,'contact_selected':False,'planning_success':False,'end_to_end_success':False}
    r[stage+'_return_code']=p.returncode;r[stage+'_parent_wall_seconds']=time.perf_counter()-begin
    if timeout or p.returncode:
        r.update(status='timeout' if timeout else 'error',failure=stage+'_failed',end_to_end_success=False)
        if stage=='plan':r['planning_success']=False
    atomic_json(f,r);return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method');p.add_argument('--stage');args=p.parse_args()
    if args.worker:return worker(args)
    root=args.output;root.mkdir();shutil.copy2(PREVIOUS/'shared.json',root/'shared.json');old=json.loads((PREVIOUS/'protocol.json').read_text());cases=old['cases']
    sources=list(Path('src').glob('*.py'));hashes={str(f):file_hash(f) for f in sources+[root/'shared.json',PREVIOUS/'protocol.json']}
    protocol={'cases':cases,'methods':METHODS,'hashes':hashes,'budget_per_stage_s':300,'scope':'Same 24 structured tasks and same 6 preselected contact cases as report48. Paired rerun with one global portal Dijkstra search and fixed-route SOCP; joint GCS unchanged, 10 rounded paths, seed0. No tuning on this rerun. Shared library preprocessing excluded from query times; per-query portal construction included.'}
    atomic_json(root/'protocol.json',protocol);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    cpu=sorted(os.sched_getaffinity(0));slots=[cpu[i] for i in [4,12,20,28]];begin=time.perf_counter()
    def slot(i):
        for j in range(i,len(cases),4):
            c=cases[j]
            for m in METHODS[j%2:]+METHODS[:j%2]:
                print('START',c['id'],m,flush=True);r=execute(root,c['id'],m,slots[i],'plan')
                if c['contact'] and r['planning_success']:r=execute(root,c['id'],m,slots[i],'contact')
                print('DONE',c['id'],m,r['planning_success'],r['end_to_end_success'],flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
    records=[json.loads(f.read_text()) for f in sorted(root.glob('task_*/*/record.json'))];atomic_json(root/'summary.json',{'records':records});atomic_json(root/'complete.json',{'seconds':time.perf_counter()-begin,'runs':len(records),'changed_inputs':[f for f,h in hashes.items() if file_hash(f)!=h]});print('COMPLETE',flush=True)


if __name__=='__main__':main()
