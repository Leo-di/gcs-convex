"""Shared finite grasp/place library; joint GCS versus exhaustive baselines."""
import argparse,copy,json,os,shutil,signal,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json,file_hash
from .semantic_candidate_gcs import build_and_solve,expand,q,WEIGHTS,PICK_PHASES,PLACE_PHASES,primitive_cost
LIBRARY=Path('outputs/results/coupled_goal_library_v1/shared.json')
METHODS=['joint_gcs','enumerate_gcs','portal_socp']


def dock_ok(base,region,tolerance=0.):
    xy=np.asarray(base[:2]);return bool(np.all(xy>=np.array(region['lower'])-tolerance) and np.all(xy<=np.array(region['upper'])+tolerance))


def options(shared,cell):
    return {name:{key:g for key,g in goals.items() if dock_ok(g['base'],cell['docking'])} for name,goals in shared['goals'].items() if name in cell['allowed']}


def materialize(shared,name,key,config):
    cs=dict(shared['candidates']);goal=shared['goals'][name][key];cs[name]={**goal,'cover':cs[name]['cover']}
    c=copy.deepcopy(config);c['delivery_task']['transfer_path'][-1]=goal['base'];c['delivery_task']['preplace_height']=goal['height']
    return cs,c


def path_cost(path,cs):
    total=0.
    for v in path:
        if v['kind'] in ['empty','carry']:total+=float(np.linalg.norm(WEIGHTS*(np.array(v['x'][10:])-v['x'][:10])))
        elif v['kind'] in ['pick','place']:
            phases=PICK_PHASES if v['kind']=='pick' else PLACE_PHASES
            total+=primitive_cost([r for r in cs[v['candidate']]['rows'] if r['phase'] in phases])
    return total


def worker(args):
    root=args.output;out=root/args.cell/args.method;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    for f,h in protocol['hashes'].items():
        if file_hash(f)!=h:raise RuntimeError('Frozen file changed: '+f)
    cell=next(c for c in protocol['cases'] if c['id']==args.cell)
    if args.stage=='contact':
        from .benchmark_full_task_development import contact_worker
        contact_worker(args);r=json.loads((out/'record.json').read_text());raw=json.loads((out/'physics.json').read_text())
        final=[x for x in raw['trace'] if x['time_s']>=raw['duration_s']-1]
        dock=all(dock_ok(x['base_q'],cell['docking'],.02) for x in final)
        r['actual_docking_satisfied']=dock;r['end_to_end_success']=bool(r['end_to_end_success'] and r['planning_success'] and dock)
        if not dock:r['failure']='actual_docking_region_violated'
        atomic_json(out/'record.json',r);return
    begin=time.perf_counter();shared=json.loads((root/'shared.json').read_text());config=copy.deepcopy(shared['config'])
    if cell['sites'] is not None:config['inspection_sites']=cell['sites']
    goals=options(shared,cell);pairs=[(n,k) for n,g in goals.items() for k in g];cs=shared['candidates'];disks=shared['disks']
    r={'cell':args.cell,'method':args.method,'planning_success':False,'end_to_end_success':False,'status':'planning',
       'contact_selected':cell['contact'],'available_pairs':len(pairs),'attempts':[],'enumeration_complete':False,'solver_seconds':0.}
    def save():r['planning_wall_seconds']=time.perf_counter()-begin;atomic_json(out/'record.json',r)
    save()
    if not pairs:r.update(status='failed',failure='no_goal_in_shared_finite_library',enumeration_complete=True);save();return
    maps=None;navigation_cache={}
    if args.method=='portal_socp':
        from .full_task_portal_socp import prepare
        t=time.perf_counter();maps=prepare({n:c for n,c in cs.items() if n in cell['allowed']},disks,5);r['roadmap_seconds']=time.perf_counter()-t
    jobs=[None] if args.method=='joint_gcs' else pairs
    for index,pair in enumerate(jobs):
        if time.perf_counter()-begin>=280:break
        attempt=out/f'attempt_{index:03d}';attempt.mkdir(exist_ok=True);t=time.perf_counter()
        if args.method=='joint_gcs':
            result,path=build_and_solve(disks,cs,cell['allowed'],config,start_base_q=cell['start'],goal_options=goals)
        elif args.method=='enumerate_gcs':
            n,k=pair;result,path=build_and_solve(disks,cs,[n],config,start_base_q=cell['start'],goal_options={n:{k:goals[n][k]}})
        else:
            from .full_task_portal_socp import solve
            n,k=pair;local,c=materialize(shared,n,k,config)
            result,path=solve(disks,local,[n],c,maps,cell['start'],navigation_cache=navigation_cache)
            if path is not None:
                result['selected_goal']=k
                next(v for v in path if v['kind']=='place')['goal']=k
        elapsed=time.perf_counter()-t;r['solver_seconds']+=elapsed;atomic_json(attempt/'plan.json',result)
        entry={'pair':pair,'status':result['status'],'solve_wall_seconds':elapsed}
        if path is not None:
            name=result['selected_candidate'];key=result['selected_goal'];local,c=materialize(shared,name,key,config)
            cost=path_cost(path,local);entry['cost']=cost
            if not r['planning_success'] or cost<r['cost']-1e-8:
                audit=expand(path,local,disks,c,attempt,visualize=False)
                dock=dock_ok(local[name]['rows'][-1]['base_q'],cell['docking'],1e-6);audit['docking_satisfied']=dock;audit['accepted']=bool(audit['accepted'] and dock);atomic_json(attempt/'audit.json',audit)
                entry['audited']=True;entry['accepted']=audit['accepted']
                if audit['accepted']:
                    r.update(planning_success=True,reference=str(attempt),selected_candidate=name,selected_goal=key,
                             selected_base=shared['goals'][name][key]['base'],selected_height=shared['goals'][name][key]['height'],
                             cost=audit['weighted_path_cost'],audit=audit,incumbent_found_seconds=time.perf_counter()-begin)
        r['attempts'].append(entry);save()
    r['enumeration_complete']=len(r['attempts'])==len(jobs)
    r['status']='planned' if r['planning_success'] else 'failed'
    if not r['planning_success']:r['failure']='no_audited_solution_within_budget'
    save()


def execute(root,cell,method,cpu,stage):
    out=root/cell/method;out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    with (out/(stage+'.log')).open('w') as log:
        p=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-m','src.coupled_goal_experiment','--output',str(root),'--worker','--cell',cell,'--method',method,'--stage',stage],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timeout=False
        try:p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    f=out/'record.json';r=json.loads(f.read_text()) if f.exists() else {'cell':cell,'method':method,'planning_success':False,'end_to_end_success':False}
    r[stage+'_return_code']=p.returncode;r[stage+'_parent_wall_seconds']=time.perf_counter()-begin
    if timeout:
        r['enumeration_complete']=False;r['status']='timeout_with_incumbent' if stage=='plan' and r['planning_success'] else 'timeout_without_solution'
        if stage=='contact':r['end_to_end_success']=False;r['failure']='contact_timeout'
    elif p.returncode:
        r.update(status='error',failure=stage+'_error',end_to_end_success=False)
        if stage=='plan':r['planning_success']=False
    atomic_json(f,r);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method');p.add_argument('--stage');a=p.parse_args()
    if a.worker:worker(a);return
    root=a.output
    if root.exists():raise ValueError('New output directory required')
    root.mkdir(parents=True);shutil.copy2(LIBRARY,root/'shared.json');shared=json.loads(LIBRARY.read_text());names=list(shared['candidates'])
    policies=[names,[names[0]],names[1:],[names[-1]]];docks=[{'name':'all','lower':[-2.6,1.0],'upper':[-2.1,1.5]},
        {'name':'west','lower':[-2.6,1.0],'upper':[-2.35,1.5]},{'name':'east','lower':[-2.35,1.0],'upper':[-2.1,1.5]}]
    rng=np.random.default_rng(20261004);cases=[]
    for pi,allowed in enumerate(policies):
        for docking in docks:
            for repeat in range(2):
                while True:
                    start=rng.uniform([-3.2,-1.2],[-2.6,1.2])
                    if any(np.linalg.norm(start-d['center'])<d['radius']-1e-5 for d in shared['disks']):break
                i=len(cases);cases.append({'id':f'task_{i:02d}','policy':pi,'allowed':allowed,'docking':docking,'start':[*start.tolist(),0.],
                    'sites':None if repeat==0 else [{'name':'south','xy':[-2.5,-1.]},{'name':'north','xy':[-2.5,1.]}],
                    'contact':i in [0,3,7,10,17,22]})
    sources=list(Path('src').glob('*.py'));protocol={'cases':cases,'methods':METHODS,'seed':20261004,'budget_per_stage_s':300,'start_no_new_solve_after_s':280,
        'scope':'24 structured mechanism tasks, not 24 independent scenes; finite shared grasp/place catalog, free choice within docking region. All failures retained; no tuning on evaluation outcomes.',
        'baselines':'Both enumerators try all legal grasp/goal pairs, retaining best audited incumbent. Non-GCS roadmap rebuilt once per query; navigation optimizations reused within query only. Enumerate-GCS still uses GCS.',
        'metrics':'Audited task success, cost and wall time on all24; actual contact and docking on six cases fixed in advance. Timeout preserves only already-audited incumbent.',
        'hashes':{str(f):file_hash(f) for f in sources+[root/'shared.json']}}
    atomic_json(root/'protocol.json',protocol);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    cpu=sorted(os.sched_getaffinity(0));cpu=[cpu[i] for i in [4,12,20,28]];begin=time.perf_counter()
    def slot(i):
        for j in range(i,len(cases),4):
            c=cases[j];order=METHODS[j%3:]+METHODS[:j%3]
            for m in order:
                print('START',c['id'],m,flush=True);r=execute(root,c['id'],m,cpu[i],'plan')
                if c['contact'] and r['planning_success']:r=execute(root,c['id'],m,cpu[i],'contact')
                print('DONE',c['id'],m,r['planning_success'],r['end_to_end_success'],r.get('failure',''),flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
    records=[json.loads(f.read_text()) for f in sorted(root.glob('*/*/record.json'))];atomic_json(root/'summary.json',{'records':records})
    atomic_json(root/'complete.json',{'seconds':time.perf_counter()-begin,'runs':len(records),'changed_inputs':[f for f,h in protocol['hashes'].items() if file_hash(f)!=h]});print('COMPLETE',flush=True)


if __name__=='__main__':main()
