"""Search must be state-derived, bounded, and accept only a complete action."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from src.transport_posture_search import DEFAULT_POLICY,validate_search,generate_candidates,select_posture,PostureSearchRejected
from src.stateful_transport_grounding import validate_task,make_config
from src.scene_spec import load_scene_config
from src.build_scene import build_scene


def test_auto_policy_forbids_hidden_manual_action_offsets():
    task=load_scene_config('configs/replicacad_human_passage_transport.yaml')
    task['transport_posture_search']=copy.deepcopy(DEFAULT_POLICY)
    with pytest.raises(ValueError,match='forbids'):validate_task(task)
    for a in task['actions']:a.pop('transport_offset_m',None)
    validate_task(task)
    for value in [0,-1,True]:
        bad=dict(DEFAULT_POLICY,max_full_plans=value)
        with pytest.raises(ValueError):validate_search(bad)


def candidates():return [dict(id=str(i),offset_world_m=[0.,-i*.1,0.],movement_cost_m=i*.1) for i in range(3)]


def test_restore_failure_triggers_another_candidate_and_never_selects_screen_alone(tmp_path):
    calls=[]
    def evaluate(c):
        calls.append(c['id'])
        if c['id']=='0':raise RuntimeError('target extension collides')
        return 'complete_audited_action'
    answer,report=select_posture(candidates(),lambda c:dict(connected=True),evaluate,DEFAULT_POLICY,tmp_path/'search.json')
    assert answer=='complete_audited_action' and report['selected']['id']=='1'
    assert calls==['0','1'] and report['trials'][0]['status']=='full_action_rejected'
    assert json.loads((tmp_path/'search.json').read_text())['selected']['id']=='1'


def test_coarse_disconnection_is_deferred_and_can_still_be_recovered(tmp_path):
    calls=[]
    answer,report=select_posture(candidates()[:1],lambda c:dict(connected=False),lambda c:calls.append(c['id']) or 'accepted',DEFAULT_POLICY,tmp_path/'search.json')
    assert answer=='accepted' and calls==['0']
    assert report['trials'][0]['screen']['connected'] is False


def test_full_plan_budget_cannot_return_partial_success(tmp_path):
    def reject(c):raise RuntimeError('invalid complete action')
    with pytest.raises(PostureSearchRejected) as error:
        select_posture(candidates(),lambda c:dict(connected=True),reject,dict(DEFAULT_POLICY,max_full_plans=1),tmp_path/'search.json')
    assert error.value.details['selected'] is None and error.value.details['full_plans']==1


@pytest.mark.skipif(not Path('data/replicacad_drake/cafe/frl_apartment_table_01/asset.json').exists(),reason='Prepared ReplicaCAD meshes')
def test_candidates_follow_current_grasp_state_and_rotate_with_base():
    task=load_scene_config('configs/replicacad_human_passage_transport.yaml');s=build_scene(config=make_config(task),visualize=False)
    arm=s.config['pick_task']['navigation_arm_q']
    a=generate_candidates(s,np.array([0.,0.,0.]),arm,DEFAULT_POLICY)
    b=generate_candidates(s,np.array([2.,3.,np.pi/2]),arm,DEFAULT_POLICY)
    assert np.allclose(a[0]['offset_world_m'],0) and np.allclose(b[0]['offset_world_m'],0)
    # Same radial search choice is transformed by actual base/grasp orientation.
    radial=lambda rows:next(r for r in rows if r['radial_fraction']==.85 and r['lateral_fraction']==0 and r['height_offset_m']==0)
    u=np.array(radial(a)['offset_world_m']);v=np.array(radial(b)['offset_world_m'])
    assert np.allclose(v[:2],[-u[1],u[0]],atol=1e-10)
    assert not np.allclose(u,v) and len(a)<=DEFAULT_POLICY['max_candidates']
    assert {r['radial_fraction'] for r in a if r['lateral_fraction']==0 and r['height_offset_m']==0}==set(DEFAULT_POLICY['radial_fractions'])


def test_deferred_trials_use_measured_component_gap_instead_of_assuming_identity_is_best(tmp_path):
    calls=[];gaps={'0':.4,'1':.01,'2':.15}
    answer,report=select_posture(candidates(),lambda c:dict(connected=False,component_gap_m=gaps[c['id']]),
        lambda c:calls.append(c['id']) or 'complete',DEFAULT_POLICY,tmp_path/'search.json')
    assert calls==['1'] and report['selected']['id']=='1'


def test_missing_complete_action_is_never_accepted(tmp_path):
    with pytest.raises(PostureSearchRejected) as error:
        select_posture(candidates()[:1],lambda c:dict(connected=True),lambda c:None,DEFAULT_POLICY,tmp_path/'search.json')
    assert error.value.details['selected'] is None
