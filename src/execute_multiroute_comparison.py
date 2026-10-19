"""Execute both prespecified full-reach alternatives with unchanged task tail."""
import argparse
import json
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform
from .build_scene import build_scene,set_robot_configuration
from .green_room_gcs import append_motion,audit_reference,room_oracle,cup_room_residual
from .execute_delivery_dynamics import execute
from .green_room_iris import make_checker
from .semantic_policy import policy_from_config,accepts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',required=True);p.add_argument('--experiment',required=True)
    args=p.parse_args();source=Path(args.source);root=Path(args.experiment)
    config=json.loads((source/'scene_config_snapshot.json').read_text())
    policy=policy_from_config(config)
    original=json.loads((source/'delivery_trace.json').read_text());original_report=json.loads((source/'result.json').read_text())
    rows=json.loads((root/'comparison.json').read_text())[0]['results'];results=[]
    for r in rows:
        out=root/('execution_'+r['method']);out.mkdir(exist_ok=False)
        if not r.get('numerically_valid',False):
            results.append({'method':r['method'],'accepted':False,'reason':'planning invalid'});continue
        scene=build_scene(config=config,visualize=False)
        dock=np.array(original_report['dock_q']);tucked=np.array(original_report['fixed_tucked_arm_q'])
        path=np.array([p for seg in r['segments'] for p in (seg['start'],seg['end'])])
        mode,checker,mapping=make_checker(config,dock,path[0],scene.gripper_open_positions)
        oracle=room_oracle(scene);violations=[];samples=0
        for j,(a,b) in enumerate(zip(path,path[1:])):
            n=max(2,int(np.ceil(np.max(abs(b-a))/.00125))+1)
            for t in np.linspace(0,1,n):
                q=(1-t)*a+t*b;samples+=1
                if not checker.CheckConfigCollisionFree(mapping(q)) or (policy=='full_motion' and not oracle.query(np.r_[dock,q],scene.gripper_open_positions)['satisfied']):
                    violations.append({'edge':j,'alpha':float(t),'q':q.tolist()})
        (out/'strict_reach_audit.json').write_text(json.dumps({'samples':samples,'violations':len(violations),'counterexamples':violations[:10]},indent=2))
        if violations:
            results.append({'method':r['method'],'accepted':False,'reason':'strict pre-execution reach audit rejected','violations':len(violations)})
            continue
        frames=[f for f in original if f['phase']=='navigate_to_pickup']
        append_motion(frames,np.repeat(dock[None,:],len(path),axis=0),path,
                      np.repeat(scene.gripper_open_positions[None,:],len(path),axis=0),'approach_cup')
        tail=[f for f in original if f['phase'] not in ('navigate_to_pickup','approach_cup')]
        assert np.max(abs(path[-1]-tail[0]['arm_q']))<1e-6
        frames+=tail
        set_robot_configuration(scene,dock,path[-1],scene.gripper_closed_positions)
        X_GC=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@scene.cup_initial_pose
        audit,trace=audit_reference(scene,frames,dock,tucked,X_GC)
        (out/'reference_audit.json').write_text(json.dumps(audit,indent=2));(out/'delivery_trace.json').write_text(json.dumps(frames))
        if not accepts(policy,geometry=audit['geometry_satisfied'],goals=audit['goal_semantics_satisfied'],motion=audit['motion_semantics_satisfied']):
            results.append({'method':r['method'],'accepted':False,'reason':'reference audit failed'});continue
        physics,dyn_scene=execute(config,frames,time_step=.001,time_scale=2.,visualize=True,grip=True)
        oracle=room_oracle(dyn_scene);room=[];cup=[];base=[]
        for f in physics['trace']:
            if f['phase']=='navigate_to_pickup':continue
            room.append(oracle.query(np.r_[f['base_q'],f['arm_q']],f['gripper_q'])['residual_m'])
            cup.append(cup_room_residual(dyn_scene,oracle,RigidTransform(np.array(f['cup_pose']))))
            base.append(float(np.linalg.norm(np.array(f['base_q'])[:2]-dock[:2])))
        checks={'room_min_m':min(room),'cup_room_min_m':min(cup),'base_error_max_m':max(base),
                'bilateral_lift':physics['lifted_with_bilateral_contact'],
                'hold_bilateral_fraction':physics['held_bilateral_contact_fraction'],
                'unexpected_contacts':physics['unexpected_contact_count'],
                'effort_limits':physics['actuator_effort_limits_satisfied'],'slip':physics['carry_slip_satisfied']}
        checks['geometry_satisfied']=bool(max(base)<=.01 and checks['bilateral_lift'] and checks['hold_bilateral_fraction']==1 and checks['unexpected_contacts']==0 and checks['effort_limits'] and checks['slip'])
        active=[f for f in physics['trace'] if f['phase']!='navigate_to_pickup']
        endpoints=[f for j,f in enumerate(active) if j==0 or j==len(active)-1 or active[j-1]['phase']!=f['phase'] or active[min(j+1,len(active)-1)]['phase']!=f['phase']]
        checks['goal_semantics_satisfied']=all(oracle.query(np.r_[f['base_q'],f['arm_q']],f['gripper_q'])['satisfied'] for f in endpoints)
        checks['motion_semantics_satisfied']=bool(min(room)>=0 and min(cup)>=0)
        checks['semantic_task_satisfied']=bool(audit['satisfied'] and checks['geometry_satisfied'] and checks['goal_semantics_satisfied'] and checks['motion_semantics_satisfied'])
        checks['semantic_policy']=policy
        checks['accepted']=accepts(policy,geometry=checks['geometry_satisfied'],goals=checks['goal_semantics_satisfied'],motion=checks['motion_semantics_satisfied'])
        checks['method']=r['method'];results.append(checks)
        physics['generic_delivery_audit_applicable']=False
        physics['generic_delivery_satisfied']=physics.pop('satisfied')
        physics['satisfied']=checks['accepted']
        physics['green_room_task_audit']=checks
        (out/'physics.json').write_text(json.dumps(physics));(out/'scene.html').write_text(dyn_scene.meshcat.StaticHtml())
        (root/'execution_comparison.json').write_text(json.dumps(results,indent=2));print(checks,flush=True)
    (root/'execution_comparison.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
