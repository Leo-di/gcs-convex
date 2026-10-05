"""Frozen new physical conditions; equal-input query solver comparison."""
import argparse,copy,json,os,shutil,signal,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
from .prepare_transfer_scenes import design,prepare
from .coupled_goal_experiment import options,materialize,path_cost,dock_ok,worker as old_worker
from .semantic_candidate_gcs import build_and_solve,expand
METHODS=['joint_10','joint_50','global_portal']


def worker(args):
    if args.stage=='contact':return old_worker(args)
    root=args.output;out=root/args.cell/args.method;p=json.loads((root/'protocol.json').read_text())
    for f,h in p['hashes'].items():
        if file_hash(f)!=h:raise RuntimeError('Frozen input changed: '+f)
    cell=next(c for c in p['cases'] if c['id']==args.cell);directory=root/'libraries'/cell['scene'];shared=json.loads((directory/'shared.json').read_text());prep=json.loads((directory/'preparation.json').read_text());begin=time.perf_counter()
    c=copy.deepcopy(shared['config'])
    if cell['sites'] is not None:c['inspection_sites']=cell['sites']
    goals=options(shared,cell);pairs=sum(map(len,goals.values()));r={'cell':args.cell,'scene':cell['scene'],'method':args.method,'contact_selected':cell['contact'],'planning_success':False,'end_to_end_success':False,'status':'planning','available_pairs':pairs,'attempts':[],'preparation_seconds_shared':prep.get('seconds')}
    def save():r['planning_wall_seconds']=time.perf_counter()-begin;atomic_json(out/'record.json',r)
    if not shared['candidates']:
        r.update(status='failed',failure='shared_preparation_no_candidate');save();return
    if not pairs:
        r.update(status='failed',failure='place_action_library_empty');save();return
    linked=[x for x in prep['connectivity'] if x['candidate'] in cell['allowed'] and x['goal'] in goals.get(x['candidate'],{})];r['carry_connected_pairs']=sum(x['region_graph_connected'] for x in linked)
    if args.method=='global_portal':
        from .full_task_portal_socp import prepare as make_maps
        from .global_portal_socp import solve
        t=time.perf_counter();maps=make_maps({n:v for n,v in shared['candidates'].items() if n in cell['allowed']},shared['disks'],5);r['roadmap_seconds']=time.perf_counter()-t
        t=time.perf_counter();result,path=solve(shared['disks'],shared['candidates'],cell['allowed'],c,maps,cell['start'],goals)
    else:
        t=time.perf_counter();result,path=build_and_solve(shared['disks'],shared['candidates'],cell['allowed'],c,start_base_q=cell['start'],goal_options=goals,max_rounded_paths=10 if args.method=='joint_10' else 50,rounding_seed=0)
    r['solver_seconds']=time.perf_counter()-t;attempt=out/'attempt_000';attempt.mkdir(exist_ok=True);atomic_json(attempt/'plan.json',result);entry={'status':result['status'],'solve_wall_seconds':r['solver_seconds']}
    if path is not None:
        name=result['selected_candidate'];key=result['selected_goal'];cs,config=materialize(shared,name,key,c);audit=expand(path,cs,shared['disks'],config,attempt,visualize=False)
        audit['docking_satisfied']=dock_ok(cs[name]['rows'][-1]['base_q'],cell['docking'],1e-6);audit['accepted']=bool(audit['accepted'] and audit['docking_satisfied']);atomic_json(attempt/'audit.json',audit)
        r.update(planning_success=audit['accepted'],reference=str(attempt),cost=path_cost(path,cs),selected_candidate=name,selected_goal=key,selected_base=cs[name]['base'],selected_height=cs[name]['height']);entry.update(audited=True,accepted=audit['accepted'])
        if not audit['accepted']:r['failure']='reference_geometry_or_semantic_audit'
    elif not r['carry_connected_pairs']:r['failure']='carrying_region_coverage_or_connectivity'
    elif result['status'] in ['unreachable_in_portal_graph','unreachable_in_compiled_graph']:r['failure']='navigation_or_semantic_graph_disconnected'
    else:r['failure']='query_solver_failed'
    r.update(attempts=[entry],enumeration_complete=True,status='planned' if r['planning_success'] else 'failed');save()


def run_process(cmd,log_path,timeout):
    with log_path.open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);expired=False
        try:p.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            expired=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    return p.returncode,expired


def execute(root,cell,method,cpu,stage):
    out=root/cell/method;out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    code,timeout=run_process(['taskset','-c',str(cpu),sys.executable,'-m','src.transfer_scene_experiment','--output',str(root),'--worker','--cell',cell,'--method',method,'--stage',stage],out/(stage+'.log'),300)
    f=out/'record.json';r=json.loads(f.read_text()) if f.exists() else {'cell':cell,'method':method,'planning_success':False,'end_to_end_success':False}
    r[stage+'_return_code']=code;r[stage+'_parent_wall_seconds']=time.perf_counter()-begin
    if code or timeout:
        r.update(status='timeout' if timeout else 'error',failure=stage+'_timeout' if timeout else stage+'_program_error',end_to_end_success=False)
        if stage=='plan':r['planning_success']=False
    atomic_json(f,r);return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare');p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method');p.add_argument('--stage');a=p.parse_args();root=a.output
    if a.prepare:return prepare(root,a.prepare)
    if a.worker:return worker(a)
    root.mkdir();spec=design();sources=list(Path('src').glob('*.py'));source_hashes={str(f):file_hash(f) for f in sources};spec['source_hashes']=source_hashes;atomic_json(root/'design.json',spec);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[i] for i in [4,12,20,28]];names=list(spec['scenes']);begin=time.perf_counter()
    def prep_slot(i):
        for si in range(i,len(names),4):
            name=names[si];directory=root/'libraries'/name;directory.mkdir(parents=True);print('START_PREP',name,flush=True);t=time.perf_counter()
            code,timeout=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.transfer_scene_experiment','--output',str(root),'--prepare',name],directory/'prepare.log',900)
            if code or timeout:
                atomic_json(directory/'preparation.json',{'seconds':time.perf_counter()-t,'failure':'preparation_timeout' if timeout else 'preparation_program_error','return_code':code,'attempts':[],'goal_attempts':[],'connectivity':[],'candidate_count':0,'goal_count':0})
                atomic_json(directory/'shared.json',{'config':spec['scenes'][name],'candidates':{},'goals':{},'disks':[]})
            print('DONE_PREP',name,code,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(prep_slot,range(4)))
    prep_seconds=time.perf_counter()-begin;changed=[f for f,h in source_hashes.items() if file_hash(f)!=h]
    if changed:raise RuntimeError('Sources changed during preparation: '+str(changed))
    resources=[root/'design.json']+list((root/'libraries').glob('*/shared.json'))+list((root/'libraries').glob('*/preparation.json'))
    protocol={'cases':spec['cases'],'methods':METHODS,'budget_per_stage_s':300,'preparation_budget_per_scene_s':900,'preparation_wall_seconds':prep_seconds,'scope':spec['scope'],'hashes':{**source_hashes,**{str(f):file_hash(f) for f in resources}},'method_scope':'GCS10 and GCS50 both seed0; global Dijkstra+SOCP same finite library. All query graph/roadmap construction timed, no cross-query cache. 8 preselected contact tasks, 24 scheduled method executions, unavailable plans remain failures.'};atomic_json(root/'protocol.json',protocol);query_begin=time.perf_counter()
    def slot(i):
        for j in range(i,len(spec['cases']),4):
            c=spec['cases'][j]
            for m in METHODS[j%3:]+METHODS[:j%3]:
                print('START',c['id'],m,flush=True);r=execute(root,c['id'],m,slots[i],'plan')
                if c['contact'] and r['planning_success']:r=execute(root,c['id'],m,slots[i],'contact')
                # Include every prespecified contact task, even without a plan.
                r['contact_selected']=c['contact'];r['scene']=c['scene'];atomic_json(root/c['id']/m/'record.json',r)
                print('DONE',c['id'],m,r['planning_success'],r['end_to_end_success'],r.get('failure',''),flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
    records=[json.loads(f.read_text()) for f in sorted(root.glob('*__q*/*/record.json'))];atomic_json(root/'summary.json',{'records':records});atomic_json(root/'complete.json',{'runs':len(records),'preparation_wall_seconds':prep_seconds,'query_and_contact_wall_seconds':time.perf_counter()-query_begin,'total_wall_seconds':time.perf_counter()-begin,'changed_inputs':[f for f,h in protocol['hashes'].items() if file_hash(f)!=h]});print('COMPLETE',flush=True)


if __name__=='__main__':main()
