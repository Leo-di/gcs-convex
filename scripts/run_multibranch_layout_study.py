"""Three frozen non-rigid cafe layouts x single-chain / semantic GCS / posthoc.

Layout recipes are set before geometry checks; every cell is retained.
"""
import argparse,concurrent.futures,copy,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from scripts.run_service_cafe_study import sha,write
RECIPES=[
 dict(id='staggered_tables',label='桌位错列与杯子前后换位',tables={'table':[.50,.80],'customer_b':[3.25,-1.0],'customer_c':[.45,3.65],'dropoff_table':[2.10,2.40]},island=[1.70,-.90],cups=[[-.08,.15],[.10,.16],[-.10,.13]],chairs='translated'),
 dict(id='asymmetric_chairs',label='非对称椅位与横向杯位',tables={'table':[.30,.95],'customer_b':[3.45,-.85],'customer_c':[.30,3.85],'dropoff_table':[1.90,2.45]},island=[1.55,-.95],cups=[[.16,.12],[-.10,.16],[.05,.16]],chairs='asymmetric'),
 dict(id='offset_return',label='回收桌右前移与错位清理桌',tables={'table':[.45,.75],'customer_b':[3.50,-1.0],'customer_c':[.55,3.70],'dropoff_table':[2.20,2.25]},island=[1.80,-.80],cups=[[-.14,.10],[.16,.15],[.15,.12]],chairs='offset'),
]
METHODS=['branch_posthoc','branch_early','early_suffix']

def make_case(scene,task,recipe,scene_path):
 scene=copy.deepcopy(scene);task=copy.deepcopy(task);task['carry'].pop('graph_mode',None)
 old={k:np.array((scene['objects'] if k in ('table','dropoff_table') else scene['support_surfaces'])[k]['center']) for k in recipe['tables']}
 shifts={k:np.r_[np.array(v)-old[k][:2],0.] for k,v in recipe['tables'].items()}
 for k,xy in recipe['tables'].items():(scene['objects'] if k in ('table','dropoff_table') else scene['support_surfaces'])[k]['center']=[*xy,.375]
 for fixture in scene['asset_fixtures']:
  name=fixture['name']
  if name in shifts:fixture['center']=(np.array(fixture['center'])+shifts[name]).tolist()
  elif name=='service_island':fixture['center']=[*recipe['island'],.45]
  elif name in ['cafe_chair_0','cafe_chair_1','cafe_chair_4','cafe_chair_5']:
   support='table' if name in ['cafe_chair_0','cafe_chair_1'] else 'customer_c';fixture['center']=(np.array(fixture['center'])+shifts[support]).tolist()
   if recipe['chairs']=='asymmetric':
    side=-1 if name.endswith(('0','4')) else 1;fixture['center'][0]+=side*.05;fixture['center'][1]+=.08 if side<0 else -.08;fixture['rpy'][2]+=side*.1
   elif recipe['chairs']=='offset':fixture['center'][1]+=.06 if name.endswith(('0','5')) else -.06
  elif recipe['chairs']=='asymmetric' and name in ['cafe_chair_2','cafe_chair_3']:
   fixture['center'][0]+=-.05 if name.endswith('2') else .05;fixture['center'][1]+=-.05 if name.endswith('2') else .05
 for item in scene['objects']['distractors']:
  if item.get('name','').startswith('staff_'):item['center']=(np.array(item['center'])+shifts['dropoff_table']).tolist()
 task['scene']=str(scene_path);task['destination_table']=copy.deepcopy(scene['objects']['dropoff_table']);task['support_surfaces']=copy.deepcopy(scene['support_surfaces'])
 for key,support,offset,action in zip(['cup','cup_b','cup_c'],['table','customer_b','customer_c'],recipe['cups'],task['actions']):
  center=(scene['objects'] if support=='table' else scene['support_surfaces'])[support]['center'];point=[center[0]+offset[0],center[1]+offset[1],.84];task['objects'][key]=point
  cup=scene['objects']['cup'] if key=='cup' else scene['objects']['additional_cups'][key];cup['center']=point.copy()
  base=[point[0]-.03,center[1]-.70,0.];action['pickup_base']=base;action['pickup_candidates']=[dict(base_q=base.copy(),grasp_rpy=task['grasp_rpy'].copy())]
  action['placements']=[(np.array(p)+shifts['dropoff_table']).tolist() for p in action['placements']]
  action.pop('transport_offset_m',None)
 human=copy.deepcopy(scene['human_passage']);human['connections'][0]['goal_xy']=(np.array(human['connections'][0]['goal_xy'])+shifts['dropoff_table'][:2]).tolist();scene['human_passage']=copy.deepcopy(human);task['human_passage']=human
 scene['scenario']['name']='service_cafe_layout_'+recipe['id'];scene['name']=scene['scenario']['name']
 fixtures={f['name']:f for f in scene['asset_fixtures']}
 for k in recipe['tables']:
  support=(scene['objects'] if k in ('table','dropoff_table') else scene['support_surfaces'])[k];assert np.allclose(support['center'],fixtures[k]['center'])
 for action in task['actions']:
  k=action['source_support'];support=scene['objects'][k] if k=='table' else scene['support_surfaces'][k];point=np.array(task['objects'][action['object_id']]);assert np.all(np.abs(point[:2]-np.array(support['center'][:2]))+.05<np.array(support['size'][:2])/2)
 assert task['human_passage']==scene['human_passage'] and 'graph_mode' not in task['carry']
 return scene,task

def freeze(root):
 root.mkdir(parents=True,exist_ok=False);old=PROJECT/'outputs/results/service_cafe_multibranch_20261101_v1';scene=json.loads((old/'scene.json').read_text());task=json.loads((old/'task.json').read_text());inputs={}
 for recipe in RECIPES:
  folder=root/'cases'/recipe['id'];folder.mkdir(parents=True);cfg,spec=make_case(scene,task,recipe,(folder/'scene.json').relative_to(PROJECT));write(folder/'scene.json',cfg);write(folder/'task.json',spec)
  for p in [folder/'scene.json',folder/'task.json']:inputs[str(p.relative_to(PROJECT))]=sha(p)
 jobs=[dict(case=recipe['id'],method=method,task=f"cases/{recipe['id']}/task.json",output=f"cases/{recipe['id']}/{method}",hard_timeout_s=9000) for method in METHODS for recipe in RECIPES]
 sources={}
 for p in [*sorted((PROJECT/'src').glob('*.py')),*sorted((PROJECT/'tests').glob('*.py')),Path(__file__).resolve(),PROJECT/'scripts/run_service_cafe_study.py']:
  rel=p.relative_to(PROJECT);dst=root/'source_snapshot'/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);sources[str(rel)]=sha(p)
 protocol=dict(recipes=RECIPES,jobs=jobs,source_sha256=sources,input_sha256=inputs,asset_sha256=json.loads((old/'protocol.json').read_text())['asset_sha256'],parallelism=6,
  method_definitions={'branch_early':'Multi-branch semantic carrying regions + fixed-suffix precheck','early_suffix':'Prior single-chain semantic carrying planner + same fixed-suffix precheck','branch_posthoc':'Geometry multi-branch carrying graph, complete action then same-rule check/reject/retry'},
  anchors='Entrance and occupied customer table fixed. Service endpoint and static staff translate by return-table displacement. Two functional relations and capsule/grid/margins unchanged.',
  docking='Same rule in all layouts: base x=cup x-.03, base y=source table center y-.70, yaw0. No manual transport offsets.',
  failure_policy='Freeze all three cases before geometry checks. Execute all nine cells once. Preserve every failure; no outcome-based case deletion or hidden retry.',
  scope='Three constructed development variations, one run/cell, not random/held-out/statistical evidence; same candidate/budget policy, not necessarily bitwise geometric covers. Pure posthoc lacks suffix precheck, so do not isolate graph effects from total timings.')
 write(root/'protocol.json',protocol);(root/'experiment_plan.md').write_text('# Multi-branch layout generalization\n\nThree frozen non-rigid table/chair/cup recipes, three matched methods per case, all nine retained. Initial/docking checks are diagnostics and never filter cases. Full planning from scratch and continuous three-cup contact execution required for task success; independently audit both human connections and admitted early graph region witnesses. All candidates, numerical settings, margins and time budgets are unchanged; no manual offset. Methods differ only as disclosed in protocol. Run long posthoc jobs first, at most six concurrent single-thread processes on the same host; wall clocks are development records, not statistical comparisons. No hidden rerun.\n\nMaterial Passport: academic-research-suite / experiment-agent / run / 2026-11-01 / UNVERIFIED.\n')
 return protocol

def execute(root,job):
 out=root/job['output'];log=root/'cases'/job['case']/(job['method']+'.log');began=time.monotonic();cmd=[sys.executable,'-u','-m','src.service_cafe_study','--config',str(root/job['task']),'--output',str(out),'--method',job['method']]
 with log.open('w') as f:
  proc=subprocess.Popen(cmd,cwd=PROJECT,env=dict(os.environ,LCM_DEFAULT_URL='memq://',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT);write(log.with_suffix('.process.json'),dict(pid=proc.pid,command=cmd,hard_timeout_s=job['hard_timeout_s']))
  try:code=proc.wait(timeout=job['hard_timeout_s'])
  except subprocess.TimeoutExpired:proc.kill();proc.wait();code='timeout'
 result=out/'result.json';return dict(case=job['case'],method=job['method'],exit=code,wall_seconds=time.monotonic()-began,result=json.loads(result.read_text()) if result.exists() else None)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--freeze-only',action='store_true');p.add_argument('--run-frozen',action='store_true');a=p.parse_args();root=a.output.resolve();protocol=json.loads((root/'protocol.json').read_text()) if a.run_frozen else freeze(root)
 for group in ['source_sha256','input_sha256','asset_sha256']:
  for path,h in protocol[group].items():assert sha(PROJECT/path)==h,path
 if not a.freeze_only:
  assert not (root/'completion.json').exists();results=[]
  with concurrent.futures.ThreadPoolExecutor(max_workers=protocol['parallelism']) as pool:
   for future in concurrent.futures.as_completed([pool.submit(execute,root,j) for j in protocol['jobs']]):
    row=future.result();results.append(row);write(root/'completion.json',dict(expected=9,completed=len(results),runs=results));print(row['case'],row['method'],row['exit'],flush=True)
  write(root/'complete.json',dict(completed=9,all_processes_finished=True))
