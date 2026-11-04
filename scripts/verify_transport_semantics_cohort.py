"""Uniform full-object evaluation and measured arm-volume intrusion witnesses."""
import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import beta

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from pydrake.geometry import Box, Cylinder, Sphere, Capsule
from pydrake.math import RigidTransform
from src.build_scene import build_scene, set_robot_configuration
from src.stateful_transport_grounding import make_config
from src.whole_robot_room import WholeRobotRoomOracle


def interior_points(shape):
    """Strictly interior points: a positive hit is a material-volume witness."""
    if isinstance(shape,Box):
        return np.array(list(itertools.product([-.9,0.,.9],repeat=3)))*np.array([shape.width(),shape.depth(),shape.height()])/2
    if isinstance(shape,Cylinder):
        return np.array([[r*np.cos(t),r*np.sin(t),z] for r in [0.,.5*shape.radius(),.95*shape.radius()]
            for t in np.linspace(0,2*np.pi,32,endpoint=False) for z in [-.45*shape.length(),0.,.45*shape.length()]])
    if isinstance(shape,Sphere):
        return np.vstack([np.zeros(3),np.eye(3)*shape.radius()*.9,-np.eye(3)*shape.radius()*.9])
    if isinstance(shape,Capsule):
        return np.array([[0.,0.,z] for z in [-shape.length()/2,0.,shape.length()/2]])
    raise ValueError(f'Unsupported witness shape {type(shape)}')


def inside_points(points, pose, zone):
    world=points@pose.rotation().matrix().T+pose.translation()
    mask=np.all((world>np.asarray(zone['lower_xyz'])+1e-6)&(world<np.asarray(zone['upper_xyz'])-1e-6),axis=1)
    return world[mask], np.flatnonzero(mask)


def cylinder_axes(shape, pose):
    if not isinstance(shape,Cylinder):
        raise ValueError('Independent object evaluator supports cylinders only')
    R=pose.rotation().matrix()
    return pose.translation(),shape.radius()*np.sqrt(R[:,0]**2+R[:,1]**2)+shape.length()/2*np.abs(R[:,2])


def object_residuals(geometry, poses, spec, phase):
    transformed={key:[(shape,pose,X_BG,*cylinder_axes(shape,pose@X_BG))
                     for shape,X_BG in shapes]
                 for key,shapes in geometry.items() for pose in [RigidTransform(np.asarray(poses[key]))]}
    values={}
    for z in spec['object_exclusions']:
        lo,hi=np.array(z['lower_xyz']),np.array(z['upper_xyz'])
        values[z['id']]=min(float(max(np.max(c-h-hi),np.max(lo-c-h))-z['margin_m'])
            for shape,pose,X_BG,c,h in transformed[z['object_id']])
    for z in spec['object_corridors']:
        if phase!=z['action_id']+'.'+z['mode']:continue
        values[z['id']]=max(min(float(np.minimum(c-h-np.array(cell['lower_xyz']),np.array(cell['upper_xyz'])-c-h).min()-z['margin_m'])
            for shape,pose,X_BG,c,h in transformed[z['object_id']]) for cell in z['cells'])
    return values


def interval(k,n):
    return [float(beta.ppf(.025,k,n-k+1)) if k else 0.,
            float(beta.ppf(.975,k+1,n-k)) if k<n else 1.]


def verify(root):
    root=Path(root).resolve();protocol=json.loads((root/'protocol.json').read_text())
    rows=json.loads((root/'comparison.json').read_text())
    expected={(j['scene'],j['method']) for j in protocol['jobs']}
    if len(rows)!=len(expected) or {(r['scene'],r['method']) for r in rows}!=expected:
        raise ValueError('Cohort incomplete or duplicate conditions')
    for rel,h in protocol['input_sha256'].items():
        assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==h,rel
    for rel,h in protocol['source_sha256'].items():
        assert hashlib.sha256((root/'source_snapshot'/rel).read_bytes()).hexdigest()==h,rel
        assert hashlib.sha256((PROJECT/rel).read_bytes()).hexdigest()==h,rel
    evidence=[]
    for row in rows:
        folder=root/row['scene']/row['method']/'run'
        record=dict(scene=row['scene'],method=row['method'],exit=row['exit'],accepted=row['accepted'],
                    wall_seconds=row['wall_seconds'],joint_success=False,dynamic_scope_witness=False)
        if not (folder/'physics.json').exists():
            search=json.loads((folder/'search.json').read_text()) if (folder/'search.json').exists() else {}
            record['reason']=row['result'].get('reason') or [r.get('reason') for r in search.get('branches',[]) if r.get('reason')] or 'No complete measured trace; see run.log'
            evidence.append(record);continue
        spec=json.loads((root/row['scene']/'all_constraints.json').read_text())
        scene=build_scene(config=make_config(spec),visualize=False);insp=scene.scene_graph.model_inspector()
        geometry={key:[(insp.GetShape(g),insp.GetPoseInFrame(g)) for g in scene.plant.GetCollisionGeometriesForBody(body)] for key,body in scene.object_bodies.items()}
        arm=[]
        for idx in scene.plant.GetBodyIndices(scene.arm_instance):
            body=scene.plant.get_body(idx)
            for g in scene.plant.GetCollisionGeometriesForBody(body):
                arm.append((body,insp.GetName(g),insp.GetPoseInFrame(g),interior_points(insp.GetShape(g))))
        room=WholeRobotRoomOracle.from_xy_bounds(scene,scene.config['green_room']['x_bounds'],scene.config['green_room']['y_bounds'],margin=.02)
        physics=json.loads((folder/'physics.json').read_text());trace=physics['trace']
        minima={z['id']:float('inf') for z in spec['object_exclusions']+spec['object_corridors']}
        counts=Counter();room_min=float('inf');samples=[];witnesses=[];object_hits=[];moving={}
        for index,f in enumerate(trace):
            set_robot_configuration(scene,f['base_q'],f['arm_q'],f['gripper_q'])
            values=object_residuals(geometry,{k:v['pose'] for k,v in f['objects'].items()},spec,f['phase'])
            for key,value in values.items():minima[key]=min(minima[key],value);counts[key]+=1
            r=room.query_current()['residual_m'] if f['mode']!='enter_room' else None
            if r is not None:room_min=min(room_min,r)
            active=[]
            for z in spec['object_exclusions']:
                for shape,X_BG in geometry[z['object_id']]:
                    pose=RigidTransform(np.array(f['objects'][z['object_id']]['pose']))@X_BG
                    points,ids=inside_points(interior_points(shape),pose,z)
                    if len(points) and len(object_hits)<20:
                        object_hits.append(dict(time_s=f['time_s'],phase=f['phase'],zone=z['id'],point_world_m=points[0].tolist()))
                for body,name,X_BG,pts in arm:
                    pose=scene.plant.EvalBodyPoseInWorld(scene.plant_context,body)@X_BG
                    points,ids=inside_points(pts,pose,z)
                    if not len(points):continue
                    active.append(dict(zone=z['id'],body=body.name()))
                    key=(z['id'],name)
                    if key not in moving:moving[key]=dict(first_time_s=f['time_s'],first_center=pose.translation().tolist(),max_center_displacement_m=0.,samples=0)
                    m=moving[key];m['samples']+=1;m['last_time_s']=f['time_s']
                    m['max_center_displacement_m']=max(m['max_center_displacement_m'],float(np.linalg.norm(pose.translation()-m['first_center'])))
                    if len(witnesses)<30:
                        witnesses.append(dict(frame_index=index,time_s=f['time_s'],phase=f['phase'],zone=z['id'],body=body.name(),geometry=name,
                            point_local_m=pts[ids[0]].tolist(),point_world_m=points[0].tolist(),shape_pose_world=pose.GetAsMatrix4().tolist(),cup_residual_same_zone_m=values[z['id']]))
            samples.append(dict(time_s=f['time_s'],phase=f['phase'],mode=f['mode'],room_residual_m=r,
                residuals=values,arm_intrusions=active,cup_xyz=np.asarray(f['objects']['cup']['pose'])[:3,3].tolist()))
        moving_rows=[dict(zone=z,geometry=g,**v) for (z,g),v in moving.items()]
        dynamic=any(v['samples']>=3 and v['max_center_displacement_m']>.005 and v['last_time_s']-v['first_time_s']>.1 and minima[v['zone']]>=-1e-8 for v in moving_rows)
        satisfied=bool(counts and all(counts[k]>0 and v>=-1e-8 for k,v in minima.items()) and room_min>=-1e-8)
        record.update(saved_samples=len(trace),minimum_residuals_m=minima,constraint_samples=dict(counts),minimum_operation_room_m=room_min,
            all_semantics_satisfied=satisfied,joint_success=bool(row['accepted'] and satisfied),dynamic_scope_witness=dynamic,
            arm_intrusion_saved_states=sum(bool(s['arm_intrusions']) for s in samples),moving_arm_geometries=moving_rows,
            arm_witnesses=witnesses,object_intrusion_witnesses=object_hits,unexpected_contacts=physics['unexpected_contact_count'],
            actions=row['result'].get('actions',{}),replay='/'+str((folder/'scene.html').relative_to(PROJECT)))
        (folder/'uniform_semantics.json').write_text(json.dumps(samples))
        (folder/'scope_witnesses.json').write_text(json.dumps(dict(arm=witnesses,moving=moving_rows,object=object_hits),indent=2))
        evidence.append(record)
        print(row['scene'],row['method'],record['joint_success'],dynamic,minima,flush=True)
    statistics=[]
    for method in protocol['methods']:
        subset=[r for r in evidence if r['method']==method];n=len(subset);k=sum(r['joint_success'] for r in subset)
        statistics.append(dict(method=method,total=n,physical_completions=sum(r['accepted'] for r in subset),
            completed_and_all_semantics=k,exact95_interval=interval(k,n),
            completed_dynamic_scope_witnesses=sum(r['joint_success'] and r['dynamic_scope_witness'] for r in subset)))
    result=dict(rows=evidence,statistics=statistics,scope='All conditions retained. Uniform full-object cylinder axis-support residuals at saved measurements; room residual uses existing complete-shape oracle. Positive interior material points prove arm/object intersection at those saved states. Absence of points proves nothing; time spans do not certify continuous occupancy. One planner seed, six local IID scenes, not arbitrary layouts.')
    (root/'verified_summary.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    verify(p.parse_args().root)
