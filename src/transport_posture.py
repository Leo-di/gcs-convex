"""Attached-payload reconfiguration before translation, with the existing IRIS/GCS gates."""
import numpy as np
from .build_scene import set_robot_configuration
from .grasp_planner import solve_arm_ik,cartesian_arm_path
from .green_room_gcs import guide_edge_validator,solve_operation


def plan_transport_posture(scene,base,arm,X_GC,offset,artifact):
    offset=np.asarray(offset,float)
    if offset.shape!=(3,) or not np.all(np.isfinite(offset)):raise ValueError('Finite world-frame transport offset required')
    closed=scene.gripper_closed_positions
    set_robot_configuration(scene,base,arm,closed)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)
    point=np.asarray(scene.config['gripper']['grasp_point_in_body'])
    target=solve_arm_ik(scene,base_q=np.asarray(base),target_position=X.multiply(point)+offset,
        target_orientation=X.rotation(),initial_arm_q=np.asarray(arm),position_tolerance=.0002,
        orientation_tolerance=.005,minimum_clearance=scene.config['pick_task']['minimum_clearance'],stage='transport_posture')
    guide=cartesian_arm_path(scene,np.asarray(base),np.asarray(arm),target,20,stage='transport_posture',
        edge_validator=guide_edge_validator(scene,base,arm,closed,semantic=True,X_GC=X_GC))
    path,report,_=solve_operation(scene,np.asarray(base),guide,closed,'transport_posture',X_GC=X_GC,artifacts=artifact)
    report['requested_gripper_offset_world_m']=offset.tolist()
    return path,report
