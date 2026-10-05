"""Forty frozen same-library planning queries; first 12 also contact-executed."""
import argparse,copy,json,os,shutil,signal,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json,file_hash
METHODS=['product_gcs','portal_socp','ignore_inspection']


def worker(args):
    root=args.output;out=root/args.cell/args.method;out.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((root/'protocol.json').read_text())
    for f,h in protocol['hashes'].items():
        if file_hash(f)!=h:raise RuntimeError('Frozen source or input changed: '+f)
    if args.stage=='contact':
        from .benchmark_full_task_development import contact_worker
        contact_worker(args)
        r=json.loads((out/'record.json').read_text());r['end_to_end_success']=bool(r['end_to_end_success'] and r['planning_success'])
        atomic_json(out/'record.json',r);return
    from .semantic_candidate_gcs import build_and_solve,expand
    from .full_task_portal_socp import prepare,solve
    shared=json.loads((root/'shared.json').read_text());cs=shared['candidates'];disks=shared['disks'];config=shared['config']
    cell=next(c for c in protocol['cases'] if c['id']==args.cell)
    if cell['sites'] is not None:config['inspection_sites']=cell['sites']
    r={'cell':args.cell,'method':args.method,'status':'planning','planning_success':False,'end_to_end_success':False,'contact_selected':cell['contact']}
    atomic_json(out/'record.json',r);began=time.perf_counter()
    if args.method=='portal_socp':
        maps=prepare(cs,disks,density=5);atomic_json(out/'roadmap.json',maps)
        result,path=solve(disks,cs,cell['allowed'],config,maps,cell['start'])
    else:
        planner_config=copy.deepcopy(config)
        if args.method=='ignore_inspection':planner_config.pop('inspection_sites',None)
        result,path=build_and_solve(disks,cs,cell['allowed'],planner_config,start_base_q=cell['start'],rounding_seed=0,max_rounded_paths=10)
    r['planner_seconds']=time.perf_counter()-began;atomic_json(out/'plan.json',result)
    if path is None:r.update(status='failed',failure='no_planner_path',geometric_success=False)
    else:
        ref=out/'reference';audit=expand(path,cs,disks,config,ref,visualize=False)
        geometry=bool(audit['semantic']['satisfied'] and audit['grasp']['satisfied'] and audit['dense']['satisfied'])
        r.update(reference=str(ref),planning_success=bool(audit['accepted']),geometric_success=geometry,
                 selected_candidate=result['selected_candidate'],cost=audit['weighted_path_cost'],audit=audit,status='planned')
        if not r['planning_success']:r['failure']='inspection_semantics_violated' if geometry else 'geometry_or_grasp_audit_failed'
    r['planning_wall_seconds']=time.perf_counter()-began;atomic_json(out/'record.json',r)


def execute(root,cell,method,cpu,stage):
    out=root/cell/method;out.mkdir(parents=True,exist_ok=True)
    with (out/(stage+'.log')).open('w') as log:
        p=subprocess.Popen(['taskset','-c',str(cpu),sys.executable,'-m','src.benchmark_gcs_mechanism','--output',str(root),'--worker','--cell',cell,'--method',method,'--stage',stage],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        timeout=False
        try:p.wait(timeout=300)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    f=out/'record.json';r=json.loads(f.read_text()) if f.exists() else {'cell':cell,'method':method,'planning_success':False,'end_to_end_success':False,'geometric_success':False}
    r[stage+'_return_code']=p.returncode
    if timeout or p.returncode:
        r.update(status='timeout' if timeout else 'error',failure=stage+'_timeout' if timeout else stage+'_error',end_to_end_success=False)
        if stage=='plan':r.update(planning_success=False,geometric_success=False)
    atomic_json(f,r);return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method');p.add_argument('--stage');a=p.parse_args()
    if a.worker:worker(a);return
    root=a.output
    if root.exists():raise ValueError('Use a fresh output directory')
    root.mkdir(parents=True)
    from .semantic_candidate_gcs import read_library
    config,cs=read_library(Path('outputs/results/semantic_candidates/library'))
    removed={}
    for name,c in cs.items():
        removed[name]=[i for i,r in enumerate(c['cover']['regions']) if r['independent_holdout_audit']['collisions']]
        c['cover']['regions']=[r for r in c['cover']['regions'] if not r['independent_holdout_audit']['collisions']]
    disks=json.loads(Path('outputs/results/semantic_joint_gcs/shared_model.json').read_text())['empty_cover']
    atomic_json(root/'shared.json',{'config':config,'candidates':cs,'disks':disks,'removed_known_colliding_regions':removed})
    rng=np.random.default_rng(20261003);cases=[];draws=0
    for i in range(40):
        while True:
            xy=rng.uniform([-3.2,-1.2],[-2.6,1.2]);draws+=1
            if any(np.linalg.norm(xy-d['center'])<d['radius']-1e-5 for d in disks):break
        sites=[None,[{'name':'south','xy':[-2.5,-1.]}],[{'name':'north','xy':[-2.5,1.]}],[{'name':'south','xy':[-2.5,-1.]},{'name':'north','xy':[-2.5,1.]}]][i%4]
        cases.append({'id':f'query_{i:02d}','start':[*xy.tolist(),0.],'sites':sites,'allowed':list(cs)[i%2:],'contact':i<12})
    sources=list(Path('src').glob('*.py'));inputs=[root/'shared.json']
    protocol={'cases':cases,'methods':METHODS,'seed':20261003,'start_draws':draws,'budget_per_stage_s':300,
        'scope':'Conditional same-scene fixed-action-library mechanism ablation; not the random placement-base benchmark. Contact cases first 12 fixed before outcomes.',
        'portal':'Five projected overlap points, same legal region adjacencies, Dijkstra then fixed-route SOCP; exhaustive legal grasp/site branches. Roadmap rebuilt inside each query timing. Project implementation, not published PRM reproduction.',
        'semantics':'Ignore only checkpoint requirement in planner; all methods audited against full task and actual pre-grasp visitation. Ignore group is an intentional incomplete ablation.',
        'metrics':'All40 audited task success and weighted path cost; first12 full contact task success; planner time includes graph/roadmap build. No optional stopping or result-driven case removal.',
        'statistics':'Descriptive paired comparison; no presumption that GCS wins. No novel global-optimality/safety claim.',
        'hashes':{str(f):file_hash(f) for f in sources+inputs}}
    atomic_json(root/'protocol.json',protocol);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    cpus=sorted(os.sched_getaffinity(0));cpus=[cpus[i] for i in [4,12,20,28]];begin=time.perf_counter()
    def slot(i):
        for j in range(i,40,4):
            cell=cases[j];order=METHODS[j%3:]+METHODS[:j%3]
            for m in order:
                print('START',cell['id'],m,flush=True);r=execute(root,cell['id'],m,cpus[i],'plan')
                if cell['contact'] and r.get('geometric_success'):r=execute(root,cell['id'],m,cpus[i],'contact')
                print('DONE',cell['id'],m,r['planning_success'],r['end_to_end_success'],r.get('failure',''),flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
    records=[json.loads(f.read_text()) for f in sorted(root.glob('*/*/record.json'))]
    atomic_json(root/'summary.json',{'records':records});atomic_json(root/'complete.json',{'seconds':time.perf_counter()-begin,'runs':len(records),'changed_inputs':[f for f,h in protocol['hashes'].items() if file_hash(f)!=h]})
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
