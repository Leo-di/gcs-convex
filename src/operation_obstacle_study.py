"""Prespecified tabletop screen widths, compared with identical final validation."""
import argparse,copy,hashlib,json,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False);(root/'configs').mkdir()
    from .scene_spec import load_scene_config
    original=load_scene_config('configs/scene_semantic_choice_v6.yaml');prior=json.loads(Path('outputs/results/joint_grounding_study_20261019/protocol.json').read_text());tasks=[]
    for name,width in [('screen_narrow',.15),('screen_medium',.30),('screen_wide',.45)]:
        config=copy.deepcopy(original);obstacle={'name':'operation_table_screen','center':[.72,.55,1.05],'size':[width,.04,.60]};config['world']['walls'].append(obstacle)
        (root/'configs'/f'{name}.json').write_text(json.dumps(config,indent=2));tasks.append({'id':name,'obstacle':obstacle})
    protocol=dict(prior,tasks=tasks,methods=['goal_only','endpoint_first','operation_first'],scope='Three prespecified widths of a fixed tabletop screen; no assumption that endpoints pass or paths fail. Local development cohort, not a generalization or infeasibility proof. All outcomes retained.',comparison='Same final semantic and physical requirements. Only order differs: navigation first, IK endpoints first, or full operation planning and audit first. Official IRIS/GCS unchanged.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest={}
    for f in Path('src').glob('*.py'):
        dst=root/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes());manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2));rows=[];begin=time.monotonic()
    def run(task,method):
        with (root/f'{task}_{method}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.joint_grounding_study','--output',str(root),'--worker',task,method],stdout=log,stderr=subprocess.STDOUT);code=proc.wait()
        path=root/task/method/'search_result.json';row=json.loads(path.read_text()) if path.exists() else {'task':task,'method':method,'success':False,'selected':None,'attempts':[],'total_wall_seconds':0,'reason':'worker failed'};row['search_exit']=code;return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,t['id'],m) for t in tasks for m in protocol['methods']]
        for future in as_completed(futures):
            row=future.result();rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2));print(row['task'],row['method'],row['success'],len(row['attempts']),flush=True)
    (root/'run_summary.json').write_text(json.dumps({'runs':len(rows),'parallel_wall_seconds':time.monotonic()-begin},indent=2))

if __name__=='__main__':main()
