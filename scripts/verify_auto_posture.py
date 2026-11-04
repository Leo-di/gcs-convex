"""Recompute candidate provenance, full-action acceptance and measured passage access."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from scripts.verify_human_passage import verify as verify_passage
from src.build_scene import build_scene,set_robot_configuration
from src.transport_posture_search import generate_candidates,validate_search


def verify(run,out,*,pickup_resolver=None):
    result=json.loads((run/'result.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
    search=json.loads((run/'search.json').read_text());reference=json.loads((run/'reference.json').read_text())
    assert result['satisfied'] and search['satisfied'] and result['unexpected_contacts']==0
    assert result['object_pose_resets_after_initialization']==0 and result['cup_attachment'] is False
    assert all(x['satisfied'] for x in result['actions'].values())
    assert len(result['measured_action_preconditions'])==3 and all(r['satisfied'] for r in result['measured_action_preconditions'])
    spec=protocol['spec'];policy=spec.get('transport_posture_search');rows=[];scene=build_scene(config=protocol['config'],visualize=False)
    if policy is not None:validate_search(policy,spec['actions'])
    point=np.asarray(scene.config['gripper']['grasp_point_in_body'])
    def position(frame):
        set_robot_configuration(scene,frame['base_q'],frame['arm_q'],frame['gripper_q'])
        return scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).multiply(point)
    for selected in search['selected']:
        folder=run/f"branch_{selected['branch']:03d}";action=next(a for a in spec['actions'] if a['id']==selected['action'])
        pickup=folder/'pickup' if pickup_resolver is None else pickup_resolver(folder/'pickup')
        arm_frames=json.loads((pickup/'reference.json').read_text())
        held=next(f for f in reversed(arm_frames) if f['mode']=='hold')
        decision=None
        if policy is not None:
            decision=json.loads((folder/'transport_posture_search.json').read_text())
            expected=generate_candidates(scene,np.array(selected['pickup_candidate']['base_q']),held['arm_q'],policy)
            assert len(expected)==len(decision['generated_candidates'])<=policy['max_candidates']
            for a,b in zip(expected,decision['generated_candidates']):
                assert a['id']==b['id'] and np.allclose(a['offset_world_m'],b['offset_world_m'],atol=1e-12,rtol=0)
            chosen=decision['selected'];assert chosen in decision['generated_candidates']
            accepted=[r for r in decision['trials'] if r['status']=='fully_accepted']
            assert len(accepted)==1 and accepted[0]['id']==chosen['id'] and accepted[0]['full_plan_attempted']
            assert decision['full_plans']==sum(bool(r['full_plan_attempted']) for r in decision['trials'])<=policy['max_full_plans']
            trial=folder/'transport_candidates'/chosen['id']
            assert json.loads((trial/'result.json').read_text())['satisfied']
            offset=np.array(chosen['offset_world_m'])
        else:offset=np.array(action.get('transport_offset_m',[0.,0.,0.]))
        retract=[f for f in reference if f['phase']==action['id']+'.transport_retract']
        extend=[f for f in reference if f['phase']==action['id']+'.transport_extend']
        measured_offset=np.zeros(3);joint_length=0.
        if np.linalg.norm(offset)>1e-10:
            assert retract and extend
            measured_offset=position(retract[-1])-position(retract[0])
            # Endpoint IK and continuation IK each admit a 0.2 mm coordinate box.
            assert np.linalg.norm(measured_offset-offset)<2*np.sqrt(3)*.0002+1e-6
            assert np.allclose(retract[-1]['arm_q'],extend[0]['arm_q'],atol=1e-7)
            assert np.allclose(retract[0]['arm_q'],extend[-1]['arm_q'],atol=1e-7)
            for block in [retract,extend]:joint_length+=float(np.linalg.norm(np.diff([f['arm_q'] for f in block],axis=0),axis=1).sum())
        else:assert not retract and not extend
        phase=json.loads((folder/'phase_acceptance.json').read_text())
        assert all(r['geometry']['satisfied'] and r['semantic']['satisfied'] for r in phase.values())
        accepted_result=json.loads((folder/'result.json').read_text())
        assert accepted_result['satisfied'] and accepted_result['reference_audit']['satisfied']
        rows.append(dict(action=action['id'],selected_offset_world_m=offset.tolist(),realized_reference_offset_world_m=measured_offset.tolist(),
            gripper_displacement_m=float(np.linalg.norm(offset)),reference_offset_error_m=float(np.linalg.norm(measured_offset-offset)),reconfiguration_joint_path_length_rad=joint_length,
            search=decision,phase_acceptance_count=len(phase)))
    out.mkdir(parents=True,exist_ok=True)
    passage=verify_passage(run,out/'passage.json')
    physics=json.loads((run/'physics.json').read_text())
    phases={}
    for row in physics['trace']:
        phase=phases.setdefault(row['phase'],dict(start_s=row['time_s'],end_s=row['time_s']))
        phase['end_s']=row['time_s']
    summary=dict(method=run.name,automatic=policy is not None,task_satisfied=True,actions=rows,
        passage=passage,online_samples=result['online_samples'],unexpected_contacts=result['unexpected_contacts'],
        maximum_placement_error_m=max(x['final_position_error_m'] for x in result['actions'].values()),
        planning_wall_seconds=search['wall_seconds'],simulated_seconds=physics['duration_s'],saved_trace_end_s=physics['trace'][-1]['time_s'],measured_phases=phases,
        scope='Frozen three-action development task; recomputed state-derived candidates; full world/restore acceptance; saved measured states independently rechecked; finite Cartesian family and sampled time')
    (out/'verified.json').write_text(json.dumps(summary,indent=2));print('VERIFIED',run.name,flush=True)
    return summary
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.run,a.output)
