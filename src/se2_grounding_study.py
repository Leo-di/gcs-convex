"""Frozen SE(2) grounding extension, with unchanged obstacle scenes and solvers."""
import argparse, copy, hashlib, itertools, json, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--require-rotation',action='store_true');parser.add_argument('--task');parser.add_argument('--validated-yaw-domain',action='store_true',default=True);parser.add_argument('--legacy-yaw-domain',dest='validated_yaw_domain',action='store_false');parser.add_argument('--bridge-method',choices=['rotation_safe_gcs','se2_gcs'],default='rotation_safe_gcs');args=parser.parse_args()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False);(root/'configs').mkdir()
    prior=Path('outputs/results/operation_obstacles_20261019')
    protocol=json.loads((prior/'protocol.json').read_text())
    if args.task:protocol['tasks']=[t for t in protocol['tasks'] if t['id']==args.task]
    candidates=[]
    for old,yaw in itertools.product(protocol['candidates'],[-.2,0.,.2]):
        if args.require_rotation and yaw==0.:continue
        c=copy.deepcopy(old);c['base_q'][2]=yaw;candidates.append(c)
    candidates.sort(key=lambda c:(round(c['ranking_score'],10),c['base_q'][2],c['orientation_index'],c['base_q']))
    for i,c in enumerate(candidates):c['id']=f'se2_{i:03d}'
    protocol.update(validated_yaw_domain=args.validated_yaw_domain,require_nonzero_goal_yaw=args.require_rotation,candidates=candidates,methods=['operation_first'],bridge_method=args.bridge_method,
        total_search_seconds=600,per_candidate_seconds=300,
        candidate_order='Navigation distance, base yaw (-0.2,0,0.2), grasp orientation index, base coordinates; fixed before execution.',
        scope='Prespecified existing development scenes (see tasks list). SE(2) capability study, not a timing or success-rate comparison with the earlier fixed-yaw study. No scene retuning. Bounded failure is not infeasibility.')
    for task in protocol['tasks']:
        config=json.loads((prior/'configs'/f"{task['id']}.json").read_text())
        # Numerical cover settings, fixed once for all tasks; physical scene unchanged.
        config['green_room']['se2_navigation']={'yaw_step':.2,'yaw_window':.2,'spacing':.3}
        if args.validated_yaw_domain:
            config['green_room']['se2_navigation'].update(yaw_window=0.,prune_contained=True)
            for box in config['gcs_base_planning']['regions']:
                if box['name'] in config['green_room']['navigation_regions']:box['yaw_bounds']=[-.4,.4]
        (root/'configs'/f"{task['id']}.json").write_text(json.dumps(config,indent=2))
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest={}
    for f in Path('src').glob('*.py'):
        dst=root/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(f.read_bytes());manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2));rows=[];tick=time.monotonic()
    def run(task):
        with (root/f'{task}.log').open('w') as log:
            code=subprocess.call([sys.executable,'-m','src.joint_grounding_study','--output',str(root),'--worker',task,'operation_first'],stdout=log,stderr=subprocess.STDOUT)
        p=root/task/'operation_first/search_result.json'
        row=json.loads(p.read_text()) if p.exists() else dict(task=task,method='operation_first',success=False,selected=None,attempts=[],total_wall_seconds=0)
        row['search_exit']=code;return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        for future in as_completed([pool.submit(run,t['id']) for t in protocol['tasks']]):
            row=future.result();rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2));print(row['task'],row['success'],row['selected'],flush=True)
    (root/'run_summary.json').write_text(json.dumps({'runs':len(rows),'parallel_wall_seconds':time.monotonic()-tick},indent=2))

if __name__=='__main__':main()
