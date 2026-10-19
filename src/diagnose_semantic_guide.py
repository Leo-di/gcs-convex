"""Reproduce guide/IRIS-seed failures without solving GCS or changing a scene."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .scene_spec import load_scene_config
from .build_scene import build_scene
from .green_room_gcs import room_oracle, make_operation_cover
from .grasp_planner import plan_pick, cartesian_arm_path
from .safe_pregrasp import plan_pregrasp
from .green_room_iris import make_checker
from .collision_checker import evaluate_collision_status
from .semantic_policy import policy_from_config


def diagnose(config_path, output):
    config=load_scene_config(config_path)
    scene=build_scene(config=config,visualize=False)
    base=np.array(config['pick_task']['candidates'][0]['base_q'])
    arm=np.array(config['pick_task']['navigation_arm_q'])
    oracle=room_oracle(scene)
    semantic=policy_from_config(config)=='full_motion'
    pick=plan_pick(scene,base)
    transition=plan_pregrasp(scene,base,arm,pick.pregrasp_q,
        state_constraint=(lambda q:oracle.query(np.r_[base,q])['satisfied']) if semantic else None)
    approach=cartesian_arm_path(scene,base,pick.pregrasp_q,pick.grasp_q,25,stage='green_room_approach')
    reference=np.vstack([transition,approach[1:]])
    domains,records,_=make_operation_cover(scene,base,reference,scene.gripper_open_positions,semantic=semantic)
    mode_scene,checker,mapping=make_checker(config,base,reference[0],scene.gripper_open_positions)
    failures=[]
    for i,r in enumerate(records):
        seed=(np.array(r['lower'])+r['upper'])/2
        if not checker.CheckConfigCollisionFree(mapping(seed)):
            failures.append({'region':i,'seed':seed.tolist(),
                'distance':asdict(evaluate_collision_status(mode_scene,base,arm_q=seed,gripper_q=scene.gripper_open_positions)),
                'guide_endpoints_free':[bool(checker.CheckConfigCollisionFree(mapping(np.array(r[k])))) for k in ['guide_start','guide_end']]})
    steps=np.max(np.abs(np.diff(reference,axis=0)),axis=1)
    index=int(np.argmax(steps))
    summary={'domain_count':len(domains),'invalid_seed_count':len(failures),
        'penetrating_seed_count':sum(r['distance']['minimum_signed_distance']<0 for r in failures),
        'minimum_failed_seed_distance_m':min((r['distance']['minimum_signed_distance'] for r in failures),default=None),
        'maximum_reference_joint_step_rad':float(steps[index]),'maximum_step_edge_index':index,
        'approach_starts_at_reference_index':len(transition)-1,
        'scope':'Guide and domain-seed diagnostics only; no GCS solution or task infeasibility claim'}
    Path(output).write_text(json.dumps(dict(summary,invalid_seeds=failures,
        reference=reference.tolist()),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();diagnose(a.config,a.output)
