"""Freeze the functional-cafe task and run each method exactly once."""
import argparse,concurrent.futures,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(d,ensure_ascii=False,indent=2));t.replace(p)
def freeze(root):
 root.mkdir(parents=True,exist_ok=False)
 source=PROJECT/'outputs/reports/cafe_pipeline_20261031/service_prototype'
 scene=json.loads((source/'scene_config.json').read_text());checks=json.loads((source/'initial_check.json').read_text())
 scene['human_passage']['connections']=[{k:r[k] for k in ['id']}|dict(start_xy=r['start'],goal_xy=r['goal']) for r in checks['checks'][1:]]
 scene['human_passage']['enforce']=True
 task=json.loads((PROJECT/'outputs/results/posture_layout_study_20261031_v2/cases/shifted_island_chairs/automatic_task.json').read_text())
 task.update(scene=str((root/'scene.json').relative_to(PROJECT)),human_passage=scene['human_passage'],time_budget_s=5400.,max_expansions=24)
 task['transport_posture_search'].update(max_full_plans=10,time_budget_s=1200.)
 write(root/'scene.json',scene);write(root/'task.json',task)
 sources={}
 for p in [*sorted((PROJECT/'src').glob('*.py')),*sorted((PROJECT/'tests').glob('*.py')),Path(__file__).resolve()]:
  rel=p.relative_to(PROJECT);dst=root/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);sources[str(rel)]=sha(p)
 assets={}
 for f in scene['asset_fixtures']:
  p=PROJECT/f['asset_file'];d=json.loads(p.read_text())
  for q in [p,Path(d['visual']),*[Path(c['path']) for c in d['collisions'] if c['type']=='convex']]:assets[str(q.relative_to(PROJECT))]=sha(q)
 jobs=[dict(method=m,output=m,hard_timeout_s=9000) for m in ['early','geometry','posthoc']]
 protocol=dict(jobs=jobs,input_sha256={str(p.relative_to(PROJECT)):sha(p) for p in [root/'scene.json',root/'task.json']},source_sha256=sources,asset_sha256=assets,
  primary_outcome='Complete three-cup contact execution with both functional access relations; geometry-only audited with the same rules',
  matched='One task, scene, candidate family, cost order, numerical settings, full-trial budget, and backtracking. Only semantic check timing/enforcement differs.',
  failure_policy='No hidden retries; preserve every rejected candidate and failed process. Any repair uses a new version.',
  scope='One functional cafe development task; no performance/generalization claim from one run per method')
 write(root/'protocol.json',protocol)
 (root/'experiment_plan.md').write_text('## Material Passport\n\n- Origin Skill: experiment-agent\n- Origin Mode: run\n- Origin Date: 2026-11-01\n- Verification Status: UNVERIFIED (no independent same-code reproduction)\n- Version Label: service_cafe_v1\n\nThe user explicitly authorized implementation and complete execution of this new scene with early, geometry-only and matched post-hoc-retry methods. Necessary scripts and concrete commands are authorized without an additional confirmation. Freeze all inputs and shared sources before execution; preserve failures. Independent measured-state checks are separate from same-code reproducibility.\n')
 return protocol
def execute(root,job):
 method=job['method'];log=root/(method+'.log');cmd=[sys.executable,'-m','src.service_cafe_study','--config',str(root/'task.json'),'--output',str(root/method),'--method',method];began=time.monotonic()
 with log.open('w') as f:
  p=subprocess.Popen(cmd,cwd=PROJECT,env=dict(os.environ,LCM_DEFAULT_URL='memq://',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT)
  write(root/(method+'.process.json'),dict(pid=p.pid,command=cmd,hard_timeout_s=job['hard_timeout_s']))
  try:code=p.wait(timeout=job['hard_timeout_s'])
  except subprocess.TimeoutExpired:p.kill();p.wait();code='timeout'
 r=root/method/'result.json'
 return dict(method=method,exit=code,wall_seconds=time.monotonic()-began,result=json.loads(r.read_text()) if r.exists() else None)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--freeze-only',action='store_true');p.add_argument('--run-frozen',action='store_true');a=p.parse_args();root=a.output.resolve();d=json.loads((root/'protocol.json').read_text()) if a.run_frozen else freeze(root)
 for group in ['input_sha256','source_sha256','asset_sha256']:
  for name,value in d[group].items():assert sha(PROJECT/name)==value,name
 if not a.freeze_only:
  assert not (root/'completion.json').exists(),'Do not overwrite or silently retry a cohort'
  rows=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
   for f in concurrent.futures.as_completed([pool.submit(execute,root,j) for j in d['jobs']]):rows.append(f.result());write(root/'completion.json',dict(expected=3,completed=len(rows),runs=rows));print(rows[-1]['method'],rows[-1]['exit'],flush=True)
  write(root/'complete.json',dict(expected=3,completed=len(rows),all_processes_finished=True))
