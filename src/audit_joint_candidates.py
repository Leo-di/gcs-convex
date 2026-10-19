"""Exhaustive endpoint diagnostic after freeze; excluded from search metrics."""
import argparse,json,time
from pathlib import Path
import numpy as np
from .scene_spec import load_scene_config
from .semantic_refinement import make_candidate_config
from .build_scene import build_scene
from .green_room_gcs import room_oracle
from .joint_grounding_checks import check_pick_endpoints


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);protocol=json.loads((root/'protocol.json').read_text());rows=[];started=time.monotonic()
    for task in protocol['tasks']:
        config=load_scene_config(root/'configs'/f"{task['id']}.json")
        screened=json.loads((root/task['id']/'goal_only/candidate_screen.json').read_text())
        for candidate in screened:
            if candidate['status']!='eligible':continue
            tick=time.monotonic();row={'task':task['id'],'candidate':candidate['id'],'base_q':candidate['base_q'],'grasp_rpy':candidate['grasp_rpy'],'accepted':False}
            try:
                scene=build_scene(config=make_candidate_config(config,candidate),visualize=False);pick,goals=check_pick_endpoints(scene,np.array(candidate['base_q']),room_oracle(scene));row.update(accepted=True,goals=goals,arm_q={k:getattr(pick,k).tolist() for k in ['pregrasp_q','grasp_q','lift_q']})
            except Exception as e:row['reason']=str(e)
            row['seconds']=time.monotonic()-tick;rows.append(row)
            (root/'endpoint_diagnostic.json').write_text(json.dumps({'scope':'Exhaustive endpoint-only diagnostic, not included in search budget or success rates. Local IK failure is not an infeasibility certificate. No task changes or candidate replacement.','rows':rows,'elapsed_seconds':time.monotonic()-started},indent=2))
        print(task['id'],sum(r['accepted'] for r in rows if r['task']==task['id']),flush=True)

if __name__=='__main__':main()
