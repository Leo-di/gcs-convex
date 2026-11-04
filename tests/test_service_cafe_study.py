import copy
from pathlib import Path
import numpy as np
import pytest
from src.human_passage import HumanPassageEvaluator,validate_passage
from src.service_cafe_study import ordered_postures,reference_passage_audit
from src.build_scene import build_scene,set_robot_configuration
from src.stateful_transport_grounding import make_config,TransportRejected
from src.scene_spec import load_scene_config

@pytest.fixture
def cafe():
    if not Path('data/replicacad_drake/human_passage/service_island/asset.json').exists():pytest.skip('mesh assets required')
    cfg=make_config(load_scene_config('configs/replicacad_human_passage_transport.yaml'))
    return cfg,build_scene(config=cfg,visualize=False)

def test_multi_connection_rejects_when_only_secondary_is_blocked(cafe):
    cfg,s=cafe;set_robot_configuration(s,[.45,0.,0.],cfg['pick_task']['navigation_arm_q'],s.gripper_open_positions)
    assert HumanPassageEvaluator(s).current()['satisfied']
    s.config['human_passage']['connections']=[dict(id='access_to_dock',start_xy=[0.,-.75],goal_xy=[.45,0.])]
    h=HumanPassageEvaluator(s);r=h.current()
    assert r['connections'][h.spec['id']]['satisfied']
    assert not r['connections']['access_to_dock']['satisfied'] and not r['satisfied']
    assert r['worst_connection_id']=='access_to_dock' and r['motion_slack_m']<0
    s.config['human_passage']['enforce']=False
    assert HumanPassageEvaluator(s).current()['satisfied']
    assert not HumanPassageEvaluator(s,force=True).current()['satisfied']

def test_duplicate_connections_are_not_silently_overwritten(cafe):
    cfg,_=cafe;p=copy.deepcopy(cfg['human_passage']);p['connections']=[dict(id=p['id'],start_xy=p['start_xy'],goal_xy=p['goal_xy'])]
    with pytest.raises(ValueError):validate_passage(p)

def test_ordered_selector_checks_after_complete_plan_and_retries(tmp_path):
    candidates=[dict(id='zero',offset_world_m=[0,0,0]),dict(id='tucked',offset_world_m=[0,-.2,0])];calls=[]
    def full(c):
        calls.append(c['id'])
        if c['id']=='zero':raise TransportRejected('posthoc_passage_rejected',dict(violations=1))
        return 'complete accepted action'
    answer,report=ordered_postures(candidates,lambda c:dict(connected=False),full,dict(max_full_plans=2,time_budget_s=10),tmp_path/'search.json')
    assert calls==['zero','tucked'] and answer=='complete accepted action'
    assert report['full_plans']==2 and report['trials'][0]['failure_stage']=='posthoc_passage_rejected'
    assert report['selected']['id']=='tucked'

def test_posthoc_audit_checks_enabled_rules_even_when_planner_disabled(cafe):
    cfg,s=cafe;s.config['human_passage']['enforce']=False
    poses={k:X.GetAsMatrix4().tolist() for k,X in s.object_initial_poses.items()}
    def f(xy):return dict(base_q=[*xy,0.],arm_q=cfg['pick_task']['navigation_arm_q'],gripper_q=s.gripper_open_positions.tolist(),mode='arrive',phase='probe.arrive',object_id='cup',world_objects=poses)
    assert reference_passage_audit(s,[f([.45,0.])])['satisfied']
    r=reference_passage_audit(s,[f([4.6,-.8])],stop_on_failure=True)
    assert not r['satisfied'] and not r['complete'] and r['counterexamples']
