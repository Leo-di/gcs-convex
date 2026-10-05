"""Prepare target-only task variations; never regenerate offline transit covers."""
import copy
import json
from pathlib import Path
import time
import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from .build_scene import build_scene, set_robot_configuration
from .semantic_candidate_gcs import read_library, PICK_PHASES, PLACE_PHASES
from .grasp_planner import solve_arm_ik, cartesian_arm_path
from .semantic_gcs_experiment import guards
from .audit_replay import audit_replay


def change_place(config, candidate, *, placement_base=None):
    """Same scene, same grasp, new IK place primitive; no new transit path."""
    result=copy.deepcopy(candidate)
    old=result['rows']; anchor=[r for r in old if r['phase']=='lift_cup'][-1]
    old_place=[r for r in old if r['phase']=='lower_into_bin']
    base=np.asarray(old_place[0]['base_q'] if placement_base is None else placement_base)
    X_GC=RigidTransform(np.asarray(anchor['gripper_pose'])).inverse() @ RigidTransform(np.asarray(anchor['cup_pose']))
    scene=build_scene(config=config,visualize=False)
    task=config['delivery_task']; center=np.asarray(task['place_cup_center'])
    target=RigidTransform(RotationMatrix(),center)
    before=RigidTransform(RotationMatrix(),center+[0,0,task['preplace_height']])
    point=np.asarray(config['gripper']['grasp_point_in_body'])
    def ik(pose,initial,stage):
        X=pose @ X_GC.inverse()
        return solve_arm_ik(scene,base_q=base,target_position=X.multiply(point),
            initial_arm_q=initial,position_tolerance=.0002,minimum_clearance=task['minimum_clearance'],
            target_orientation=X.rotation(),orientation_tolerance=.005,stage=stage)
    pre_q=ik(before,np.asarray(old_place[0]['arm_q']),'development_preplace')
    end_q=ik(target,pre_q,'development_place')
    lowering=cartesian_arm_path(scene,base,pre_q,end_q,35,stage='development_lower')
    end_q=lowering[-1]
    set_robot_configuration(scene,base,end_q,scene.gripper_closed_positions)
    achieved=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body) @ X_GC
    retreat=cartesian_arm_path(scene,base,end_q,pre_q,35,stage='development_retreat')
    rows=[]
    def add(phase,arm,gripper,attached):
        set_robot_configuration(scene,base,arm,gripper)
        g=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)
        cup=g @ X_GC if attached else achieved
        r=copy.deepcopy(old_place[0]);r.update(phase=phase,base_q=base.tolist(),arm_q=np.asarray(arm).tolist(),
            gripper_q=np.asarray(gripper).tolist(),attached=attached,cup_grasped=attached,
            gripper_pose=g.GetAsMatrix4().tolist(),cup_pose=cup.GetAsMatrix4().tolist())
        rows.append(r)
    for arm in lowering:add('lower_into_bin',arm,scene.gripper_closed_positions,True)
    for grip in np.linspace(scene.gripper_closed_positions,scene.gripper_open_positions,25):
        add('release_cup',end_q,grip,False)
    for arm in retreat:add('retreat_arm',arm,scene.gripper_open_positions,False)
    # This only audits the replacement primitive; no reference transit toward
    # the new endpoint is generated or fed into online IRIS.
    local=audit_replay(rows,config,10)
    if not local['satisfied']:
        raise RuntimeError('New place primitive failed dense audit: '+str(local['first_violation']))
    result['rows']=[r for r in old if r['phase'] not in PLACE_PHASES]+rows
    result['guards']=guards(result['rows'],config)
    result['new_place_audit']=local
    return result


def prepare(root):
    config,library=read_library(Path('outputs/results/semantic_candidates/library'))
    disks=json.loads(Path('outputs/results/semantic_joint_gcs/shared_model.json').read_text())['empty_cover']
    order=sorted(library)
    cells=[
        dict(id='policy_all',allowed=order),
        dict(id='policy_exclude_110',allowed=order[1:]),
        dict(id='policy_exclude_110_115',allowed=order[2:]),
        dict(id='policy_only_125',allowed=['x125']),
        dict(id='inspect_south',allowed=order,sites=[{'name':'south','xy':[-2.5,-1.]}]),
        dict(id='inspect_either',allowed=['x110','x125'],sites=[{'name':'south','xy':[-2.5,-1.]},{'name':'north','xy':[-2.5,1.]}]),
    ]
    for name,shift in [('x_minus',[-.04,0,0]),('x_plus',[.04,0,0]),
                       ('y_minus',[0,-.04,0]),('y_plus',[0,.04,0])]:
        cells.append(dict(id='target_'+name,allowed=order,shift=shift))
    (root/'development_cases.json').write_text(json.dumps(cells,indent=2))
    for cell in cells:
        began=time.perf_counter();c=copy.deepcopy(config);candidates={};rejected={};removed={}
        if 'sites' in cell:c['inspection_sites']=cell['sites']
        if 'shift' in cell:
            c['delivery_task']['place_cup_center']=(np.asarray(c['delivery_task']['place_cup_center'])+cell['shift']).tolist()
        for name in order:
            candidate=copy.deepcopy(library[name])
            removed[name]=[i for i,r in enumerate(candidate['cover']['regions']) if r['independent_holdout_audit']['collisions']]
            candidate['cover']['regions']=[r for r in candidate['cover']['regions'] if not r['independent_holdout_audit']['collisions']]
            try:
                if 'shift' in cell:candidate=change_place(c,candidate)
            except RuntimeError as e:
                rejected[name]=str(e)
            else:candidates[name]=candidate
        bundle={'cell':cell,'config':c,'candidates':candidates,'disks':disks,
                'preparation_seconds':time.perf_counter()-began,'primitive_rejections':rejected,
                'known_colliding_cached_regions_removed':removed}
        dest=root/'inputs';dest.mkdir(exist_ok=True)
        (dest/(cell['id']+'.json')).write_text(json.dumps(bundle))
        print('PREPARED',cell['id'],'candidates',list(candidates),'rejected',rejected,flush=True)
    return cells
