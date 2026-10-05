"""Success-first experiment: 8 physical-scene/policy conditions, 3 methods."""
import json,os,sys,subprocess,time,signal,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash


def main():
 root=Path('outputs/results/success_priority_v1')
 if root.exists():raise RuntimeError('Output exists')
 root.mkdir();methods=['first_valid','enumerate','product'];cells=[]
 policies={'upper_door':{'lower':[-.2,0.],'upper':[.2,.6]},'stricter_upper':{'lower':[-.2,-.08],'upper':[.2,.6]}}
 for i,scene in enumerate(['nominal','cup_left','cup_right','cart_near_route']):
  library=Path('outputs/results/semantic_multiscene_v2/libraries')/scene
  for j,(policy,box) in enumerate(policies.items()):
   shift=(i+j)%3;cells.append({'id':scene+'__'+policy,'scene':scene,'policy':policy,'box':box,'seed':0,'methods':methods[shift:]+methods[:shift],'library':str(library),'model':str(library/'model.json')})
 paths=sorted(Path('src').glob('*.py'))+sorted(Path('outputs/results/semantic_multiscene_v2/libraries').rglob('*.json'))
 design={'cells':cells,'library':cells[0]['library'],'model':cells[0]['model'],'budget':300,'hashes':{str(p):file_hash(p) for p in paths},
 'primary_metric':'Predeclared task-attempt success: valid audited incumbent found within 300 seconds AND policy-valid physical execution. Missing plan counts as end-to-end failure, no filtering. Some conditions may be infeasible in current finite cover; no physical feasibility claim.',
 'secondary':'Audited planning success, execution success conditional on having a plan, first valid time; negative correct-rejection rates reported separately from earlier experiment, not pooled.',
 'scope':'8 predeclared physical scene/policy pairs; one fixed seed per condition, three fair task-aware methods. Includes 1mm policy margin, retries only, same frozen v2 candidate libraries. No selecting easy scenes or regenerating based on results.',
 'contacts':'All 24 planning runs scheduled for contact validation; missing plans retained as failed end-to-end attempts. No seed or best-looking filtering.',
 'frozen_library_limit':'End-to-end here covers planner through contact simulation using a shared existing candidate library; it is not cold-start from an unprepared scene.'}
 atomic_json(root/'design.json',design);snapshot=root/'source_snapshot';snapshot.mkdir()
 for p in Path('src').glob('*.py'):shutil.copyfile(p,snapshot/p.name)
 def slot(slot):
  for cell in cells[slot::4]:
   for method in cell['methods']:
    out=root/cell['id']/method;out.mkdir(parents=True,exist_ok=True);t=time.perf_counter()
    env={**os.environ,'LCM_DEFAULT_URL':'memq://','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    with (out/'run.log').open('w') as log:
     proc=subprocess.Popen(['taskset','-c',str([4,12,20,28][slot]),sys.executable,'-m','src.benchmark_carry_policy','--root',str(root),'--worker','--cell',cell['id'],'--method',method,'--started',str(t)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
     timeout=False
     try:proc.wait(timeout=300)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();timeout=True
    p=out/'record.json';record=json.loads(p.read_text()) if p.exists() else {'cell':cell['id'],'method':method,'best':None,'attempts':[]}
    record.update(parent_wall_seconds=time.perf_counter()-t,return_code=proc.returncode,timed_out=timeout)
    if timeout:record['status']='timeout'
    elif proc.returncode:record['status']='error'
    atomic_json(p,record);print(cell['id'],method,record['status'],flush=True)
 with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
 atomic_json(root/'complete.json',{'runs':24})
if __name__=='__main__':main()
