"""Necessary semantic check for the planner's posture-independent fixed suffix.

Rejects one grounded placement/template, never proves general infeasibility.
No persistent cache: every world, docking/grasp, rule, and placement is rechecked.
"""
import copy,json,time
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform
from .build_scene import set_robot_configuration
from . import stateful_transport_grounding as core

def released_suffix(scene,world,action,placement,source_frames):
    """Exactly the same translated free-object release/retreat states as execution."""
    source=RigidTransform(np.asarray(world['poses'][action['object_id']]))
    grasp=next(f for f in reversed(source_frames) if f['mode']=='close')
    set_robot_configuration(scene,grasp['base_q'],grasp['arm_q'],grasp['gripper_q'])
    X_GC=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@source
    translated=core.translated_destination_suffix(source_frames,np.asarray(placement)-source.translation())
    lower=[f for f in translated if f['mode']=='lower']
    if not lower:raise ValueError('Fixed suffix requires a lower phase')
    final=lower[-1];set_robot_configuration(scene,final['base_q'],final['arm_q'],final['gripper_q'])
    released=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)@X_GC
    frames=[]
    for f in translated:
        if f['mode'] not in ('release','retreat'):continue
        f['world_objects']=copy.deepcopy(world['poses']);f['world_objects'][action['object_id']]=released.GetAsMatrix4().tolist()
        f['world_supports']=dict(world['supports']);f['world_supports'][action['object_id']]=action['destination_support']
        f['payload_mode']='free';frames.append(f)
    if not frames or frames[-1]['mode']!='retreat':raise ValueError('Fixed suffix requires release and retreat')
    return frames

def check_destination_suffix(scene,world,action,placement,spec,source_frames,folder):
    from .service_cafe_study import reference_passage_audit
    began=time.monotonic();frames=released_suffix(scene,world,action,placement,source_frames)
    audit=reference_passage_audit(scene,frames,stop_on_failure=True)
    premise=dict(world_sha256=core.digest(world),action=action,placement=list(placement),
        source_frames_sha256=core.digest(source_frames),free_suffix_sha256=core.digest(frames),
        config_sha256=core.digest(scene.config),rules=copy.deepcopy(scene.config['human_passage']),
        template='restore_transport_posture_then_translated_lower_release_retreat_v1',
        audit_joint_steps=[.01]*10+[.001]*2)
    report=dict(satisfied=audit['satisfied'],premise=premise,premise_sha256=core.digest(premise),audit=audit,
        checked_modes=['release','retreat'],full_transport_plans_before_check=0,
        wall_seconds=time.monotonic()-began,
        scope='Necessary condition for this exact fixed suffix and world only. Passing does not accept a complete action. Failing excludes every transport offset restoring to this suffix, not alternative retreat plans or general physical feasibility.')
    (Path(folder)/'destination_suffix_check.json').write_text(json.dumps(report,indent=2))
    if not audit['satisfied']:raise core.TransportRejected('fixed_destination_suffix_passage_rejected',report)
    return report
