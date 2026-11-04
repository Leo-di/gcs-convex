"""Task-adapter candidates for an upright cylinder and a +Y grasp-point gripper.

The existing grasp RPY library becomes offsets around the bearing from the
docking position to the target grasp point. This is not an arbitrary-object
grasp generator; collision, IK, semantic, path and physical checks still apply.
"""
import copy
import math
import numpy as np
from src.semantic_refinement import generate_candidates


def facing_yaw(base_xy,target_xy):
    delta=np.asarray(target_xy,float)-np.asarray(base_xy,float)
    if delta.shape!=(2,) or not np.all(np.isfinite(delta)) or np.linalg.norm(delta)<1e-10:
        raise ValueError('Distinct finite planar base and target positions required')
    yaw=math.atan2(delta[1],delta[0])-math.pi/2
    return math.atan2(math.sin(yaw),math.cos(yaw))


def object_facing_candidates(config):
    # The pose library retains roll/pitch and yaw offsets. World yaw is formed
    # separately for each docking position, including grasp-position offsets.
    point=np.asarray(config['gripper']['grasp_point_in_body'],float)
    if point.shape!=(3,) or not np.allclose(point[[0,2]],0) or point[1]<=0:
        raise ValueError('This upright-cup adapter requires a positive local Y grasp point')
    if any(not np.allclose(rpy[:2],0) for rpy in config['semantic_refinement']['grasp_rpy_candidates']):
        raise ValueError('This adapter supports upright yaw-only grasp offsets')
    rows=copy.deepcopy(generate_candidates(config))
    for row in rows:
        target=np.asarray(config['objects']['cup']['center'][:2])+np.asarray(row['grasp_position_offset_m'][:2])
        relative_yaw=row['grasp_rpy'][2];heading=facing_yaw(row['base_q'][:2],target)
        row['grasp_rpy'][2]=math.atan2(math.sin(heading+relative_yaw),math.cos(heading+relative_yaw))
        row['grasp_heading_mode']='object_facing'
        row['grasp_yaw_offset_rad']=relative_yaw
    # Keep the same navigation objective; use the relative orientation offset
    # for ties so world-coordinate yaw wrapping does not change the ordering.
    rows.sort(key=lambda p:(round(p['ranking_score'],10),p['grasp_yaw_offset_rad'],p['base_q'],p['grasp_position_offset_m']))
    for i,row in enumerate(rows):row['id']=f'candidate_{i:03d}'
    return rows
