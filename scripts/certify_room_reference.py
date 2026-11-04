"""Model-based whole-body room bounds for piecewise-linear configuration references.

Uses exact shape support at interval midpoints plus a global kinematic
point-displacement bound. Subdivision tightens a bound; it is not merely a
sampling test. Floating-point geometry is not interval arithmetic. Actual
tracking, collision freedom and contact dynamics are outside this certificate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.build_scene import build_scene
from src.green_room_gcs import room_oracle
from src.semantic_plan import configuration
from src.whole_robot_room import arm_point_reach_bound, shape_support


def certify_frames(scene,frames,*,max_depth=20,max_queries=200000,guard_m=1e-8):
    if not frames:raise ValueError('Nonempty reference required')
    if type(max_depth) is not int or max_depth<0 or type(max_queries) is not int or max_queries<1:
        raise ValueError('Nonnegative depth and positive query budget required')
    if not np.isfinite(guard_m) or guard_m<=0:raise ValueError('Positive numerical guard required')
    oracle=room_oracle(scene);reach=arm_point_reach_bound(scene,oracle)
    qs=[configuration(f) for f in frames]
    finger_bound=max(np.abs(scene.gripper_open_positions).sum(),np.abs(scene.gripper_closed_positions).sum())
    if any(np.abs(q[10:]).sum()>finger_bound+1e-12 for q in qs):
        raise ValueError('Finger positions exceed the excursion assumed by the kinematic reach bound')
    queries=0;leaves=0;lower=float('inf');max_used_depth=0
    details=[]
    for edge,(a,b) in enumerate(zip(qs,qs[1:])):
        stack=[(a,b,0)]
        while stack:
            left,right,depth=stack.pop();mid=(left+right)/2
            if queries>=max_queries:
                return dict(status='unresolved',reason='Query budget exhausted',queries=queries,edge=edge)
            value=oracle.query(mid[:10],mid[10:]);queries+=1;max_used_depth=max(max_used_depth,depth)
            if value['residual_m'] < -guard_m:
                return dict(status='violated',queries=queries,edge=edge,witness=mid.tolist(),
                    residual_m=value['residual_m'],worst=value['worst'])
            arm_displacement=reach*np.abs(right[3:10]-left[3:10]).sum()/2+np.abs(right[10:]-left[10:]).sum()/2
            translation=np.linalg.norm(right[:2]-left[:2])/2
            yaw=abs(right[2]-left[2])/2
            rotation_displacement=0.
            if yaw:
                # Bound horizontal extent at midpoint, then enlarge by the
                # arm/gripper displacement before rotating about base origin.
                radius=0.
                for body,shape,X_BG,_ in oracle.geometry:
                    pose=scene.plant.EvalBodyPoseInWorld(scene.plant_context,body)@X_BG
                    rotation=pose.rotation().matrix()
                    extent=np.linalg.norm([shape_support(shape,rotation.T@axis) for axis in np.eye(3)[:2]])
                    radius=max(radius,float(np.linalg.norm(pose.translation()[:2]-mid[:2])+extent))
                rotation_displacement=(radius+arm_displacement)*yaw
            base_displacement=translation+rotation_displacement
            bound=min(value['base_residual_m']-base_displacement,
                      value['arm_gripper_residual_m']-arm_displacement-base_displacement)
            if bound>=guard_m:
                leaves+=1;lower=min(lower,float(bound));continue
            if depth>=max_depth:
                return dict(status='unresolved',reason='Subdivision depth exhausted',edge=edge,queries=queries,bound_m=float(bound))
            stack.extend([(left,mid,depth+1),(mid,right,depth+1)])
        details.append(dict(edge=edge,phase=frames[edge+1]['phase']))
    if len(qs)==1:
        value=oracle.query(qs[0][:10],qs[0][10:]);queries=1;lower=value['residual_m'];leaves=1
        if lower<guard_m:return dict(status='violated' if lower < -guard_m else 'unresolved',residual_m=lower)
    return dict(status='certified_model_bound',queries=queries,certified_intervals=leaves,
        minimum_interval_lower_bound_m=lower,arm_point_reach_bound_m=reach,max_subdivision_depth=max_used_depth,
        numerical_guard_m=guard_m,edges=len(qs)-1,
        scope='Entire piecewise-linear configuration reference under the stated support and kinematic model; floating point, not interval arithmetic. No measured-execution, collision or dynamic certificate.')


def certify_candidate(folder):
    folder=Path(folder);file=folder/'delivery_trace.json';cfg=json.loads((folder/'scene_config_snapshot.json').read_text())
    frames=json.loads(file.read_text());operation=[f for f in frames if f['phase']!='navigate_to_pickup']
    scene=build_scene(config=cfg,visualize=False);result=certify_frames(scene,operation)
    result['reference_sha256']=hashlib.sha256(file.read_bytes()).hexdigest()
    result['config_sha256']=hashlib.sha256((folder/'scene_config_snapshot.json').read_bytes()).hexdigest()
    result['certificate_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (folder/'room_reference_certificate.json').write_text(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--candidate',required=True);a=p.parse_args();print(json.dumps(certify_candidate(a.candidate),indent=2))
