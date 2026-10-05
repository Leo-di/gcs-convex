"""Preregistered, paired finite-library baselines with hard wall-time budgets.

Workers run on separate single CPU cores. B1 and B2 are independently executed;
neither reuses another method's solve or audit. Every incumbent is checkpointed
only after the same complete geometric and semantic audit.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

from .semantic_candidate_gcs import read_library, build_and_solve, expand


METHODS=['B1_first_valid','B2_enumerate','joint_gcs']
STARTS={'original':[-3.,0.,0.], 'south':[-2.5,-1.,0.], 'north':[-3.,1.,0.]}
POLICIES={'all':['x110','x115','x120','x125'],
          'exclude_110':['x115','x120','x125'],
          'exclude_110_115':['x120','x125'], 'none':[]}
SEEDS=[0,7,19]


def atomic_json(path,data):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(data,indent=2));temporary.replace(path)


def file_hash(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol(root,budget):
    library=Path('outputs/results/semantic_candidates/library')
    paths=[Path(__file__),Path('src/semantic_candidate_gcs.py'),Path('src/compare_carry_planners.py'),
           Path('src/audit_replay.py'),Path('src/semantic_monitor.py'),Path('src/grasp_audit.py'),
           Path('outputs/results/semantic_joint_gcs/shared_model.json')]
    paths+=sorted(library.rglob('*.json'))
    cells=[]
    for si,(s,xy) in enumerate(STARTS.items()):
        for pi,(p,allowed) in enumerate(POLICIES.items()):
            for ri,seed in enumerate(SEEDS):
                shift=(si+pi+ri)%3
                cells.append({'id':f'{s}__{p}__seed{seed}','start':xy,'start_name':s,'policy':p,
                              'allowed':allowed,'seed':seed,'methods':METHODS[shift:]+METHODS[:shift]})
    spec={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'budget_seconds_per_method':budget,
          'version':2,'pilot_excluded':'outputs/results/semantic_baselines_long/pilot_stopped.json',
          'representation':'Eliminate the eight fixed empty-posture coordinates algebraically for ALL methods. Carrying remains 10D; edge continuity and output audits use lifted full states. This changes numerical representation, not the feasible paths or objective.',
          'library':str(library),'model':'outputs/results/semantic_joint_gcs/shared_model.json',
          'methods':METHODS,'cells':cells,'rounding_paths':10,'seeds':SEEDS,'workers':4,'cpus':[4,12,20,28],
          'drake':version('drake'),'input_sha256':{str(p):file_hash(p) for p in paths},
          'protocol':'36 paired cells = 3 starts x 4 action policies x 3 rounding seeds; 108 independent method runs. No obstacle or object geometry variations. Negative none policy analyzed separately.',
          'budget_scope':'Wall time includes Python startup, input loading, graph construction, solving and complete trace audit. Contact dynamics excluded and evaluated separately. Incumbents preserved on timeout.',
          'primary_metric':'Audited weighted configuration path length; graph objective includes tiny vertex tie costs and is secondary.',
          'candidate_order':'Ascending x, fixed before results; B1 stops at first audited success; B2 tries every allowed candidate and keeps best audited incumbent.',
          'baseline_equivalence':'A fixed-candidate branch has fixed action endpoints; outbound and carrying path objectives are separable. It is solved as one single-branch GCS, using the same transcription as the joint method.',
          'library_cost_scope':'Primary comparison reuses the identical frozen finite library. Historical measured library preparation is reported separately, not as a newly measured end-to-end cold-start time.',
          'contact_selection':'After all planning: seed0 for all-start/all-policy-positive cells (3 starts x 3 policies x 3 methods = 27 runs). Execute every accepted selected incumbent, including failures; no best-looking filtering.',
          'failure_scope':'Graph infeasibility and solver failures are not proofs of physical task infeasibility; current cover contains known nonzero IRIS holdout collision fractions.'}
    atomic_json(root/'protocol.json',spec);return spec


def worker(args):
    root=args.output;spec=json.loads((root/'protocol.json').read_text())
    cell=next(c for c in spec['cells'] if c['id']==args.cell)
    directory=root/cell['id']/args.method;directory.mkdir(parents=True,exist_ok=True)
    record={'cell':cell['id'],'method':args.method,'seed':cell['seed'],'start':cell['start'],
            'policy':cell['policy'],'allowed':cell['allowed'],'status':'running','attempts':[],
            'best':None,'incumbent_history':[],'cpu_affinity':sorted(os.sched_getaffinity(0))}
    def elapsed():return time.perf_counter()-args.started
    def save():
        record['elapsed_seconds']=elapsed();atomic_json(directory/'record.json',record)
    save()
    try:
        for f,expected in spec['input_sha256'].items():
            if file_hash(f)!=expected:raise ValueError('Frozen benchmark input changed: '+f)
        config,candidates=read_library(Path(cell.get('library',spec['library'])))
        disks=json.loads(Path(cell.get('model',spec['model'])).read_text())['empty_cover']
        subsets=[cell['allowed']] if args.method=='joint_gcs' else [[n] for n in cell['allowed']]
        if not cell['allowed']:
            record['status']='library_exhausted' if cell.get('physical_scene') else 'unreachable_in_compiled_graph';record['enumeration_complete']=True;save();return
        for allowed in subsets:
            attempt_name='joint' if args.method=='joint_gcs' else allowed[0]
            record['current_attempt']=attempt_name;save()
            started=elapsed()
            attempt_dir=directory/attempt_name;attempt_dir.mkdir(exist_ok=True)
            from .online_candidate_repair import solve_audited
            result,audit,attempt_dir,repair_history=solve_audited(disks,candidates,allowed,config,attempt_dir,
                cell['start'],cell['seed'],spec['rounding_paths'],spec.get('repair_mode','none'),spec.get('repair_rounds',0))
            entry={'name':attempt_name,'allowed':allowed,'started_seconds':started,
                   'solver_status':result['status'],'build_seconds':result['build_seconds'],
                   'solve_seconds':result['solve_seconds'],'accepted':False,'repair_history':repair_history}
            if audit is not None:
                entry.update(accepted=audit['accepted'],candidate=result['selected_candidate'],
                             objective=result['objective'],cost=audit['weighted_path_cost'],
                             base_path_length_m=audit['base_path_length_m'],
                             semantic=audit['semantic']['satisfied'],grasp=audit['grasp']['satisfied'],
                             dense=audit['dense']['satisfied'],dense_samples=audit['dense']['samples'])
                if entry['accepted'] and (record['best'] is None or entry['cost']<record['best']['cost']):
                    record['best']={**entry,'reference':str(attempt_dir),'discovered_seconds':elapsed()}
                    record['incumbent_history'].append({'seconds':elapsed(),'cost':entry['cost'],'candidate':entry['candidate']})
            entry['finished_seconds']=elapsed();record['attempts'].append(entry);save()
            if args.method=='B1_first_valid' and entry['accepted']:break
        record['status']='completed' if record['best'] else 'no_valid_solution'
        record['enumeration_complete']=(args.method!='B2_enumerate' or len(record['attempts'])==len(cell['allowed']))
        record.pop('current_attempt',None);save()
    except Exception as e:
        record.update(status='error',error=str(e),traceback=traceback.format_exc());save();raise


def run(root,spec):
    lock=threading.Lock();finished=[];active={};began=time.perf_counter()
    def progress():
        atomic_json(root/'progress.json',{'elapsed_seconds':time.perf_counter()-began,
                    'completed':len(finished),'total':len(spec['cells'])*3,'active':active,'runs':finished})
    def cell_run(index,cell,cpu):
        for method in cell['methods']:
            directory=root/cell['id']/method;directory.mkdir(parents=True,exist_ok=True)
            started=time.perf_counter();key=cell['id']+'/'+method
            command=['taskset','-c',str(cpu),sys.executable,'-m','src.benchmark_semantic_baselines',
                     '--worker','--output',str(root),'--cell',cell['id'],'--method',method,'--started',str(started)]
            env={**os.environ,'LCM_DEFAULT_URL':'memq://','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1',
                 'MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1'}
            with lock:active[key]={'cpu':cpu,'started_monotonic':started};progress()
            with (directory/'run.log').open('w') as log:
                process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                timed_out=False
                try:process.wait(timeout=spec['budget_seconds_per_method'])
                except subprocess.TimeoutExpired:
                    timed_out=True;os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=3)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            duration=time.perf_counter()-started
            file=directory/'record.json'
            record=json.loads(file.read_text()) if file.exists() else {'cell':cell['id'],'method':method,'status':'error','best':None,'attempts':[]}
            record.update(parent_wall_seconds=duration,return_code=process.returncode,timed_out=timed_out)
            if timed_out:record.update(status='timeout_with_incumbent' if record['best'] else 'timeout_without_solution',enumeration_complete=False)
            elif process.returncode!=0:record['status']='error'
            atomic_json(file,record)
            with lock:
                active.pop(key);finished.append(record);progress()
                print(json.dumps({'done':len(finished),'of':len(spec['cells'])*3,'cell':cell['id'],'method':method,
                                  'status':record['status'],'wall':duration,'best':record['best']['cost'] if record['best'] else None}),flush=True)
    # Each slot owns a fixed core, and all three methods in a cell share it.
    def slot_run(slot):
        for index in range(slot,len(spec['cells']),spec['workers']):cell_run(index,spec['cells'][index],spec['cpus'][slot])
    with ThreadPoolExecutor(max_workers=spec['workers']) as pool:
        list(pool.map(slot_run,range(spec['workers'])))
    atomic_json(root/'complete.json',{'elapsed_seconds':time.perf_counter()-began,'runs':len(finished)})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/results/semantic_baselines_long'))
    p.add_argument('--budget',type=float,default=300.)
    p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method',choices=METHODS);p.add_argument('--started',type=float)
    args=p.parse_args()
    if args.worker:worker(args);return
    if args.output.exists() and any(args.output.iterdir()):raise ValueError('Use an empty output directory; do not mix benchmark runs')
    args.output.mkdir(parents=True,exist_ok=True)
    spec=protocol(args.output,args.budget);run(args.output,spec)


if __name__=='__main__':main()
