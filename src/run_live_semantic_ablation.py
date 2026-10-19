"""Matched policy ablation using the rotation-safe GCS / IRIS execution pipeline."""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'configs').mkdir()
    source=Path('outputs/results/rotation_safe_search_20261019/configs/screen_narrow.json')
    config=json.loads(source.read_text())
    candidates=[]
    for x,y,yaw in itertools.product([-.45,-.15,.15,.45,.65],[0.,.25,.5],[0.,-.2,.2]):
        candidates.append({'base_q':[x,y,yaw],'grasp_rpy':[0.,0.,0.],
                           'ranking_score':((x-config['robot']['start'][0])**2+(y-config['robot']['start'][1])**2)**.5})
    candidates.sort(key=lambda c:(round(c['ranking_score'],10),abs(c['base_q'][2]),c['base_q']))
    for i,c in enumerate(candidates):c['id']=f'ablation_{i:03d}'
    policies=['geometry_only','goal_only','full_motion']
    protocol={'tasks':[{'id':v} for v in policies],'methods':['operation_first'],
              'candidates':candidates,'seed':20261019,'bridge_method':'rotation_safe_gcs',
              'total_search_seconds':600,'per_candidate_seconds':300,
              'scene_source':str(source),'scene_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
              'scope':'One frozen prior narrow-screen scene; same candidate pool, ordering and wall-time budget. Only semantic gates differ. No statistical superiority claim. Full semantics audited for every executed policy. Task ends after lift/hold.',
              'candidate_order':'Navigation distance, absolute base yaw, base coordinates. Fixed before outcomes.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for policy in policies:
        cfg=copy.deepcopy(config);cfg['semantic_policy']=policy
        (root/'configs'/f'{policy}.json').write_text(json.dumps(cfg,indent=2))
    manifest={}
    for f in Path('src').glob('*.py'):
        dest=root/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
        manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LCM_DEFAULT_URL='memq://')
    processes=[];began=time.monotonic()
    for policy in policies:
        log=(root/f'{policy}.log').open('w')
        proc=subprocess.Popen([sys.executable,'-m','src.joint_grounding_study','--output',str(root),'--worker',policy,'operation_first'],stdout=log,stderr=subprocess.STDOUT,env=env)
        processes.append((policy,proc,log))
    rows=[]
    for policy,proc,log in processes:
        code=proc.wait();log.close()
        f=root/policy/'operation_first/search_result.json'
        row=json.loads(f.read_text()) if f.exists() else {'success':False}
        row.update(semantic_policy=policy,process_exit=code)
        if row.get('selected'):
            c=json.loads((f.parent/row['selected']/'candidate_result.json').read_text())
            row['semantic_task_satisfied']=c['physical']['semantic_task_satisfied']
            row['physical']=c['physical'];row['selected_base']=c['base_q']
        rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2))
        print(policy,row.get('success'),row.get('semantic_task_satisfied'),flush=True)
    (root/'run_summary.json').write_text(json.dumps({'wall_seconds':time.monotonic()-began,'runs':len(rows)},indent=2))

if __name__=='__main__':main()
