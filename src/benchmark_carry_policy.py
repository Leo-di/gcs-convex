"""State-dependent geofence ablation, same 3D scene and frozen library."""
import argparse,copy,json,os,sys,subprocess,time,signal,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
from .semantic_candidate_gcs import read_library
from .online_candidate_repair import solve_audited
from .carry_zone_policy import clip_library,monitor
ROOT=Path('outputs/results/carry_policy_v1')
METHODS=['ignore_rule','first_valid','enumerate','product']

def worker(args):
 root=args.root;design=json.loads((root/'design.json').read_text());cell=next(c for c in design['cells'] if c['id']==args.cell)
 out=root/cell['id']/args.method;out.mkdir(parents=True,exist_ok=True)
 for p,h in design['hashes'].items():
  if file_hash(p)!=h:raise RuntimeError('Frozen input changed: '+p)
 config,candidates=read_library(Path(cell.get('library',design['library'])));disks=json.loads(Path(cell.get('model',design['model'])).read_text())['empty_cover']
 started=args.started if args.started is not None else time.perf_counter()
 if args.method!='ignore_rule':candidates=clip_library(candidates,cell['box'])
 atomic_json(out/'policy_library_sizes.json',{n:len(c['cover']['regions']) for n,c in candidates.items()})
 subsets=[list(candidates)] if args.method in ['ignore_rule','product'] else [[n] for n in candidates]
 record={'cell':cell['id'],'method':args.method,'attempts':[],'best':None,'status':'running'}
 for i,allowed in enumerate(subsets):
  result,audit,dest,history=solve_audited(disks,candidates,allowed,config,out/f'candidate{i}',[-3.,0.,0.],cell['seed'],10,'retry',3)
  entry={'allowed':allowed,'solver_status':result['status'],'history':history,'physical_valid':bool(audit and audit['accepted']),'policy_valid':False}
  if audit is not None:
   rows=json.loads((dest/'delivery_trace.json').read_text());task=monitor(rows,cell['box'])
   entry.update(policy_valid=task['satisfied'],policy=task,cost=audit['weighted_path_cost'],reference=str(dest),candidate=result['selected_candidate'],discovered_seconds=time.perf_counter()-started)
   if entry['physical_valid'] and entry['policy_valid'] and (record['best'] is None or entry['cost']<record['best']['cost']):record['best']=copy.deepcopy(entry)
  record['attempts'].append(entry);atomic_json(out/'record.json',record)
  if args.method=='first_valid' and record['best']:break
 record['status']='accepted' if record['best'] else 'no_policy_valid_solution';atomic_json(out/'record.json',record)

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--worker',action='store_true');p.add_argument('--cell');p.add_argument('--method');p.add_argument('--started',type=float);args=p.parse_args()
 if args.worker:worker(args);return
 root=args.root
 if root.exists():raise RuntimeError('Output exists')
 root.mkdir();library=Path('outputs/results/semantic_multiscene_v2/libraries/nominal')
 policies={'control':None,'central_strip':{'lower':[-.2,-.08],'upper':[.2,.08]},'upper_door':{'lower':[-.2,0.],'upper':[.2,.6]},'door_closed_to_carry':{'lower':[-.2,-.6],'upper':[.2,.6]}}
 cells=[]
 for i,(policy,box) in enumerate(policies.items()):
  for j,seed in enumerate([0,7,19]):
   shift=(i+j)%4;cells.append({'id':f'{policy}__seed{seed}','policy':policy,'box':box,'seed':seed,'methods':METHODS[shift:]+METHODS[:shift]})
 paths=sorted(Path('src').glob('*.py'))+sorted(library.rglob('*.json'))
 design={'cells':cells,'library':str(library),'model':str(library/'model.json'),'budget':300,'hashes':{str(p):file_hash(p) for p in paths},
 'scope':'Carrying base-center geofence; empty transit unchanged. Same frozen library for all methods, policy-aware carry sets intersect complement halfspaces with 1mm margin. No regeneration. Same retry-only up to 4 solves per subset, no cuts because fragment seeds are inherited and not anchors.',
 'contacts':'All four policies, seed0, three policy-aware methods, 12 planned slots; missing valid plans retained as skipped. Actual mode conservatively latched at first bilateral grasp contact until release phase.'}
 atomic_json(root/'design.json',design);snapshot=root/'source_snapshot';snapshot.mkdir()
 for p in Path('src').glob('*.py'):shutil.copyfile(p,snapshot/p.name)
 def slot(slot):
  for cell in cells[slot::4]:
   for method in cell['methods']:
    out=root/cell['id']/method;out.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    env={**os.environ,'LCM_DEFAULT_URL':'memq://','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with (out/'run.log').open('w') as log:
     process=subprocess.Popen(['taskset','-c',str([4,12,20,28][slot]),sys.executable,'-m','src.benchmark_carry_policy','--root',str(root),'--worker','--cell',cell['id'],'--method',method],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
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
