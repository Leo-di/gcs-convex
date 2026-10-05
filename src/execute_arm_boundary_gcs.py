"""Force-driven doorway subtask validation, with its own acceptance criteria."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Rgba
from pydrake.math import RigidTransform

from .arm_boundary_semantics import ArmBoundaryOracle
from .build_scene import build_scene
from .collision_checker import evaluate_collision_status
from .execute_delivery_dynamics import execute


def reference_from_path(path, gripper):
    rows=[]
    for segment in path:
        a,b=np.array(segment['start']),np.array(segment['end'])
        duration=float(np.max(np.abs(b-a)/np.array([.2,.2,.3]+[.3]*7)))
        intervals=max(1,int(np.ceil(duration/.15)))
        for t in np.linspace(0,1,intervals+1):
            if rows and t==0:continue
            q=(1-t)*a+t*b
            rows.append(dict(base_q=q[:3].tolist(),arm_q=q[3:].tolist(),gripper_q=gripper,
                             phase='navigate_to_pickup',semantic_mode=segment['mode']))
    return rows


def run(case_name, method):
    root=Path('outputs/results/arm_boundary_gcs_v1')
    directory=root/case_name/method
    record=json.loads((directory/'result.json').read_text())
    fixture=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())
    config=fixture['config'];config['execution_dynamics']={'hydroelastic_support':True}
    oracle_scene=build_scene(config=config,visualize=False)
    reference=reference_from_path(record['result']['path'],oracle_scene.gripper_open_positions.tolist())
    output=directory/'execution';output.mkdir(exist_ok=True)
    (output/'reference.json').write_text(json.dumps(reference,indent=2))
    raw,scene=execute(config,reference,time_step=.001,time_scale=3.,visualize=True,grip=False)
    # Preserve the controller's complete original output. Its delivery success
    # flag tests picking/placing a cup, which is NOT the task in this experiment.
    (output/'controller_raw.json').write_text(json.dumps(raw))
    oracle=ArmBoundaryOracle(oracle_scene)
    gate=record['case']['gate_x'];semantic_min=float('inf');collision_min=float('inf')
    active_samples=0;guard_crossings=0;worst=None
    for i,row in enumerate(raw['trace']):
        q=np.r_[row['base_q'],row['arm_q']]
        oracle_scene.plant.SetFreeBodyPose(oracle_scene.plant_context,oracle_scene.cup_body,RigidTransform(np.array(row['cup_pose'])))
        col=evaluate_collision_status(oracle_scene,q[:3],arm_q=q[3:],gripper_q=row['gripper_q'])
        collision_min=min(collision_min,col.minimum_signed_distance)
        # Also inspect interpolation to the exact gate in the recorded trace.
        samples=[(q,np.array(row['gripper_q']))]
        if i:
            prev=raw['trace'][i-1];a=np.r_[prev['base_q'],prev['arm_q']]
            if a[0]<gate<=q[0]:
                t=(gate-a[0])/(q[0]-a[0]);guard_crossings+=1
                samples.append(((1-t)*a+t*q,(1-t)*np.array(prev['gripper_q'])+t*np.array(row['gripper_q'])))
        for state,gripper in samples:
            if state[0]>gate+1e-9:continue
            active_samples+=1;value=oracle.query(state,gripper)
            if value['residual_m']<semantic_min:
                semantic_min=value['residual_m'];worst=dict(time_s=row['time_s'],q=state.tolist(),body=value['worst']['body'])
    final=raw['trace'][-1]
    goal=np.array(record['result']['path'][-1]['end'])
    final_error=np.r_[final['base_q'],final['arm_q']]-goal
    result=dict(case=case_name,method=method,active_samples=active_samples,guard_crossings=guard_crossings,
                minimum_active_semantic_residual_m=semantic_min,worst=worst,
                minimum_sampled_collision_distance_m=collision_min,
                final_base_error_m_rad=float(np.max(np.abs(final_error[:3]))),
                final_arm_error_rad=float(np.max(np.abs(final_error[3:]))),
                maximum_arm_tracking_error_rad=raw['maximum_arm_tracking_error_rad'],
                maximum_base_position_error_m=raw['maximum_base_position_error_m'],
                unexpected_contact_count=raw['unexpected_contact_count'],
                effort_limits_satisfied=raw['actuator_effort_limits_satisfied'],
                duration_s=raw['duration_s'],observed_states=len(raw['trace']),
                scope='Doorway subtask only; force-driven arm and ideal actuated planar base, 1 ms SAP; observed-state checks at 50 ms, no continuous-time proof.')
    result['accepted']=bool(active_samples>0 and guard_crossings==1 and semantic_min>=-1e-6
                            and collision_min>=-1e-6 and max(abs(final_error))<=.02
                            and raw['unexpected_contact_count']==0 and raw['actuator_effort_limits_satisfied'])
    (output/'result.json').write_text(json.dumps(result,indent=2))
    scene.meshcat.SetObject('semantic_boundary',Box(.008,1.2,2.2),Rgba(1,.15,.15,.22))
    scene.meshcat.SetTransform('semantic_boundary',RigidTransform([0,0,1.1]))
    scene.meshcat.SetCameraPose([.8,-1.8,7.],[-.6,0,.8])
    (output/'scene.html').write_text(scene.meshcat.StaticHtml())
    print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case');p.add_argument('--method',choices=['base_only','semantic_gcs'])
    p.add_argument('--freeze-protocol',action='store_true')
    args=p.parse_args()
    if args.freeze_protocol:
        root=Path('outputs/results/arm_boundary_gcs_v1')
        protocol=json.loads((root/'protocol.json').read_text())
        attempts=[dict(case=c['name'],method=m) for c in protocol['cases'] if c['expected']=='positive'
                  for m in ('base_only','semantic_gcs')]
        (root/'execution_protocol.json').write_text(json.dumps(dict(attempts=attempts,
            code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            final_configuration_tolerance=.02,semantic_tolerance_m=1e-6,
            collision_tolerance_m=1e-6,unexpected_contact_threshold_N=.1,
            maximum_speeds=[.2,.2,.3]+[.3]*7,reference_dt=.15,
            controller='execute_delivery_dynamics.execute; delivery criterion replaced by doorway-specific audit'),indent=2))
    else:
        if not args.case or not args.method:p.error('--case and --method required')
        run(args.case,args.method)


if __name__=='__main__':main()
