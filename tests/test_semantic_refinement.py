"""Candidate grounding, immutable scenes and acceptance gates."""
import copy
import numpy as np
import pytest
from src.scene_spec import load_scene_config
from src.semantic_refinement import (accepted_result, generate_candidates,
                                     make_candidate_config, screen_candidates)


@pytest.fixture
def config():
    return load_scene_config('configs/scene_green_room_search.yaml')


def test_grid_does_not_use_manual_dock_and_is_deterministic(config):
    a = generate_candidates(config)
    config['pick_task']['candidates'] = [{'base_q': [90, 90, 0]}]
    assert generate_candidates(config) == a
    assert len(a) == 110
    assert len({tuple(r['base_q'] + r['grasp_rpy']) for r in a}) == len(a)
    assert all(r['base_q'][2] == config['robot']['start'][2] for r in a)
    assert len({r['base_q'][0] for r in a}) > 1
    assert len({r['base_q'][1] for r in a}) > 1


def test_candidate_config_cannot_move_objects_or_initial_state(config):
    before = copy.deepcopy(config)
    child = make_candidate_config(config, generate_candidates(config)[-1])
    assert config == before
    for key in ['world', 'objects', 'robot', 'arm', 'gripper', 'green_room']:
        assert child[key] == before[key]
    child['objects']['cup']['center'][0] = 999
    assert config == before


def test_goal_inside_is_not_marked_as_executable(config):
    candidates = [dict(id='crosses', base_q=[.25, 0, 0], grasp_rpy=[0, 0, 0]),
                  dict(id='inside', base_q=[1.2, -.1, 0], grasp_rpy=[0, 0, 0])]
    rows = screen_candidates(config, candidates)
    assert rows[0]['reason'] == 'whole_robot_goal_outside_room'
    assert rows[0]['dock_room_residual_m'] < 0
    assert rows[1]['status'] == 'eligible'
    assert rows[1]['reason'] == 'goal_only_not_yet_connected'


def test_reference_success_cannot_be_reported_as_physical_success():
    report = {'satisfied': True, 'reference_audit': {'satisfied': True}, 'dynamics_executed': False}
    assert accepted_result(report, False)
    assert not accepted_result(report, True)
    report.update(dynamics_executed=True, physics={'satisfied': False})
    assert not accepted_result(report, True)
    report['physics']['satisfied'] = True
    assert accepted_result(report, True)
    report['reference_audit']['satisfied'] = False
    assert not accepted_result(report, True)
    assert not accepted_result({}, False)


@pytest.mark.parametrize('step', [0, -1, float('nan')])
def test_invalid_search_grid_rejected(config, step):
    config['semantic_refinement']['xy_step_m'] = step
    with pytest.raises(ValueError):
        generate_candidates(config)


def test_optional_yaw_search_preserves_xy_and_grasp_set(config):
    original=generate_candidates(config)
    config['semantic_refinement']['base_yaw_candidates']=[-.2,0.,.2]
    expanded=generate_candidates(config)
    assert len(expanded)==3*len(original)
    assert {tuple(c['base_q'][:2]+c['grasp_rpy']) for c in expanded}=={tuple(c['base_q'][:2]+c['grasp_rpy']) for c in original}
    assert {c['base_q'][2] for c in expanded}=={-.2,0.,.2}


@pytest.mark.parametrize('angles',[[],[float('nan')],[0.,0.],[[0.]]])
def test_invalid_yaw_search_rejected(config,angles):
    config['semantic_refinement']['base_yaw_candidates']=angles
    with pytest.raises(ValueError):generate_candidates(config)


def test_full_grasp_pose_search_and_immutable_goal_config(config):
    original = generate_candidates(config)
    config['semantic_refinement']['grasp_position_offsets_m'] = [[0., 0., -.004], [0., 0., .004]]
    config['semantic_refinement']['base_yaw_candidates'] = [-.15, .15]
    expanded = generate_candidates(config)
    assert len(expanded) == 4*len(original)
    assert {c['grasp_position_offset_m'][2] for c in expanded} == {-.004, .004}
    child = make_candidate_config(config, expanded[0])
    assert child['pick_task']['grasp_position_offset_m'] == expanded[0]['grasp_position_offset_m']
    assert child['objects'] == config['objects']


@pytest.mark.parametrize('offsets', [[], [[0., 0.]], [[0., 0., float('nan')]], [[0., 0., 0.], [0., 0., 0.]]])
def test_invalid_grasp_position_library_rejected(config, offsets):
    config['semantic_refinement']['grasp_position_offsets_m'] = offsets
    with pytest.raises(ValueError, match='grasp position'):
        generate_candidates(config)
