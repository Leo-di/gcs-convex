"""Checkpoint task ablation with independent raw-state task monitoring."""
import argparse,copy,json,os,sys,subprocess,time,signal,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
from .semantic_candidate_gcs import read_library,inspection_monitor
from .online_candidate_repair import solve_audited
ROOT=Path('outputs/results/inspection_semantics_v1')
METHODS=['ignore_task','first_site','enumerate_sites','product_gcs']

def worker(args):
 root=args.root;design=json.loads((root/'design.json').read_text());cell=next(c for c in design['cells'] if c['id']==args.cell)
 out=root/cell['id']/args.method;out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
 for p,h in design['hashes'].items():
  if file_hash(p)!=h:raise RuntimeError('Frozen input changed: '+p)
 config,candidates=read_library(Path(design['library']));disks=json.loads(Path(design['model']).read_text())['empty_cover']
 sites=cell['sites'];subsets=([sites] if args.method in ['ignore_task','product_gcs'] else [[s] for s in sites]) or [[]]
 record={'cell':cell['id'],'method':args.method,'attempts':[],'best':None,'status':'running'}
 for i,subset in enumerate(subsets):
  c=copy.deepcopy(config)
  if args.method!='ignore_task':c['inspection_sites']=subset
  result,audit,dest,history=solve_audited(disks,candidates,list(candidates),c,out/f'site{i}',[-3.,0.,0.],cell['seed'],10,'cuts',3)
  entry={'sites':subset,'solver_status':result['status'],'history':history,'physical_valid':bool(audit and audit['accepted']),'task_valid':False}
  if audit is not None:
   rows=json.loads((dest/'delivery_trace.json').read_text());task=inspection_monitor(rows,sites)
   entry.update(task_valid=task['satisfied'],task=task,cost=audit['weighted_path_cost'],reference=str(dest),selected_site=result.get('inspection_site'),candidate=result['selected_candidate'])
   if entry['physical_valid'] and entry['task_valid'] and (record['best'] is None or entry['cost']<record['best']['cost']):record['best']=copy.deepcopy(entry)
  record['attempts'].append(entry);record['elapsed_seconds']=time.perf_counter()-begin;atomic_json(out/'record.json',record)
  if args.method=='first_site' and record['best']:break
 record.update(status='accepted' if record['best'] else 'no_task_valid_solution',elapsed_seconds=time.perf_counter()-begin);atomic_json(out/'record.json',record)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method');args=p.parse_args()
 if args.worker:worker(args);return
 root=args.root
 if root.exists():raise RuntimeError('Output exists')
 root.mkdir();library=Path('outputs/results/semantic_multiscene_v2/libraries/nominal');south={'name':'south','xy':[-2.5,-1.]};north={'name':'north','xy':[-2.5,1.]}
 tasks={'south_required':[south],'north_required':[north],'either':[south,north],'unavailable':[]};cells=[]
 for i,(task,sites) in enumerate(tasks.items()):
  for j,seed in enumerate([0,7,19]):
   shift=(i+j)%4;cells.append({'id':f'{task}__seed{seed}','task':task,'sites':sites,'seed':seed,'methods':METHODS[shift:]+METHODS[:shift]})
 paths=sorted(Path('src').glob('*.py'))+sorted(library.rglob('*.json'))
 design={'cells':cells,'library':str(library),'model':str(library/'model.json'),'budget':300,'hashes':{str(p):file_hash(p) for p in paths},
 'scope':'One original 3D scene, frozen repaired library. Visit one allowed checkpoint before grasp then deliver. Checkpoint is geometric visit, not sensor simulation. ignore_task is deliberately task-incomplete ablation, not a valid baseline. First-site tries south then north; enumeration tries each; product jointly chooses sites in layered GCS. All share candidate-choice GCS and up to 3 cut/retry rounds per site subset.',
 'contacts':'Positive required/either tasks at seed0 for three task-aware methods; 9 executions. All selected before outcomes; unavailable must reject.'}
 atomic_json(root/'design.json',design);snapshot=root/'source_snapshot';snapshot.mkdir()
 for p in Path('src').glob('*.py'):shutil.copyfile(p,snapshot/p.name)
 def slot(slot):
  for cell in cells[slot::4]:
   for method in cell['methods']:
    out=root/cell['id']/method;out.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    env={**os.environ,'LCM_DEFAULT_URL':'memq://','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with (out/'run.log').open('w') as log:
     process=subprocess.Popen(['taskset','-c',str([4,12,20,28][slot]),sys.executable,'-m','src.benchmark_inspection_semantics','--root',str(root),'--worker','--cell',cell['id'],'--method',method],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     timeout=False
     try:process.wait(timeout=300)
     except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait();timeout=True
    path=out/'record.json';record=json.loads(path.read_text()) if path.exists() else {'cell':cell['id'],'method':method,'best':None,'attempts':[]}
    record.update(parent_wall_seconds=time.perf_counter()-t,return_code=process.returncode,timed_out=timeout)
    if timeout:record['status']='timeout'
    elif process.returncode:record['status']='error'
    atomic_json(path,record);print(cell['id'],method,record['status'],flush=True)
 with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
 atomic_json(root/'complete.json',{'runs':48})
if __name__=='__main__':main()
