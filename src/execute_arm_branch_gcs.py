"""Execute all declared positive-case plans and audit full-arm halfspaces."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Rgba
from pydrake.math import RigidTransform

from .arm_branch_gcs import ROOT
from .arm_boundary_semantics import ArmBoundaryOracle
from .build_scene import build_scene
from .collision_checker import evaluate_collision_status
from .execute_arm_boundary_gcs import reference_from_path
from .execute_delivery_dynamics import execute


def audit_execution(raw, scene, case, goal):
    x_oracle=ArmBoundaryOracle(scene)
    lateral={side:ArmBoundaryOracle(scene,[0,1 if side=='positive_y' else -1,0],bound)
             for side,bound in case['lateral_limits'].items()}
    minima={'room_x':float('inf'),**{side:float('inf') for side in lateral}}
    crossings=0;active=0;collision_min=float('inf');gate=case['gate_x']
    for i,row in enumerate(raw['trace']):
        q=np.r_[row['base_q'],row['arm_q']];g=np.array(row['gripper_q'])
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(row['cup_pose'])))
        col=evaluate_collision_status(scene,q[:3],arm_q=q[3:],gripper_q=g)
        collision_min=min(collision_min,col.minimum_signed_distance)
        samples=[(q,g)]
        if i:
            prev=raw['trace'][i-1];a=np.r_[prev['base_q'],prev['arm_q']]
            if a[0]<gate<=q[0]:
                t=(gate-a[0])/(q[0]-a[0]);crossings+=1
                samples.append(((1-t)*a+t*q,(1-t)*np.array(prev['gripper_q'])+t*g))
        for state,gripper in samples:
            if state[0]<=gate+1e-9:
                active+=1;minima['room_x']=min(minima['room_x'],x_oracle.query(state,gripper)['residual_m'])
            for side,oracle in lateral.items():
                minima[side]=min(minima[side],oracle.query(state,gripper)['residual_m'])
    last=raw['trace'][-1];error=np.r_[last['base_q'],last['arm_q']]-goal
    report=dict(minimum_semantic_residuals_m=minima,active_room_samples=active,gate_crossings=crossings,
                minimum_sampled_collision_distance_m=collision_min,
                final_configuration_error=float(np.max(np.abs(error))),
                unexpected_contact_count=raw['unexpected_contact_count'],
                actuator_effort_limits_satisfied=raw['actuator_effort_limits_satisfied'],
                base_workspace_bounds_satisfied=raw['base_workspace_bounds_satisfied'],
                maximum_arm_tracking_error_rad=raw['maximum_arm_tracking_error_rad'],
                maximum_base_position_error_m=raw['maximum_base_position_error_m'],
                duration_s=raw['duration_s'],observed_states=len(raw['trace']),
                scope='Force-driven doorway subtask; ideal planar base; observed-state halfspace and collision checks, not continuous-time proof')
    report['accepted']=bool(active>0 and crossings==1 and min(minima.values())>=-1e-6 and
                            collision_min>=-1e-6 and max(abs(error))<=.02 and
                            not raw['unexpected_contact_count'] and raw['actuator_effort_limits_satisfied']
                            and raw['base_workspace_bounds_satisfied'])
    return report


def run(case_name, method):
    directory=ROOT/case_name/method;record=json.loads((directory/'result.json').read_text())
    fixture=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())
    config=fixture['config'];config['execution_dynamics']={'hydroelastic_support':True}
    oracle_scene=build_scene(config=config,visualize=False)
    reference=reference_from_path(record['result']['path'],oracle_scene.gripper_open_positions.tolist())
    output=directory/'execution';output.mkdir(exist_ok=True)
    (output/'reference.json').write_text(json.dumps(reference))
    raw,scene=execute(config,reference,time_step=.001,time_scale=3.,visualize=True,grip=False)
    (output/'controller_raw.json').write_text(json.dumps(raw))
    result=audit_execution(raw,oracle_scene,record['case'],np.array(record['result']['path'][-1]['end']))
    result.update(case=case_name,method=method,selected_branch=record['result']['selected_branch'])
    (output/'result.json').write_text(json.dumps(result,indent=2))
    scene.meshcat.SetObject('semantic_boundary',Box(.008,1.2,2.2),Rgba(1,.15,.15,.22))
    scene.meshcat.SetTransform('semantic_boundary',RigidTransform([0,0,1.1]))
    for side,bound in record['case']['lateral_limits'].items():
        name='semantic_'+side
        scene.meshcat.SetObject(name,Box(2.8,.008,1.8),Rgba(.1,.45,1.,.18))
        scene.meshcat.SetTransform(name,RigidTransform([-.35,bound if side=='positive_y' else -bound,1.2]))
    scene.meshcat.SetCameraPose([.4,-1.1,5.],[-.4,0,.9])
    (output/'scene.html').write_text(scene.meshcat.StaticHtml())
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case');p.add_argument('--method');p.add_argument('--freeze-protocol',action='store_true')
    args=p.parse_args()
    if args.freeze_protocol:
        records=json.loads((ROOT/'summary.json').read_text())
        cells=[r for r in records if r['case']['expected']=='positive']
        attempts=[dict(case=r['case']['name'],method=r['method'],planning_status=r['result']['status'],
                       run=bool(r['result']['path'])) for r in cells]
        (ROOT/'execution_protocol.json').write_text(json.dumps(dict(attempts=attempts,
            execution_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            physics_step_s=.001,control_period_s=.01,observation_period_s=.05,
            reference_interval_s=.15,semantic_tolerance_m=1e-6,collision_tolerance_m=1e-6,
            final_configuration_tolerance=.02,
            policy='Every returned positive-case plan is executed, irrespective of reference audit outcome. Unreachable plans remain denominator failures.'),indent=2))
    else:
        if not args.case or not args.method:p.error('--case and --method required')
        run(args.case,args.method)


if __name__=='__main__':main()
