import copy
import numpy as np
import pytest

from src.build_scene import build_scene, set_robot_configuration
from src.scene_spec import load_scene_config
from src.semantic_plan import SemanticPlanEvaluator, ground_sequence, configuration


@pytest.fixture(scope='module')
def scene():
    return build_scene(config=load_scene_config('configs/scene_semantic_choice_v6.yaml'), visualize=False)


def frame(scene, arm=None, *, base=(.45, 0., 0.), phase='operate'):
    return {'base_q': list(base), 'arm_q': list(arm if arm is not None else scene.config['pick_task']['navigation_arm_q']),
            'gripper_q': scene.gripper_open_positions.tolist(), 'phase': phase}


def plan():
    return {'version': 1, 'predicates': {'inside': {'kind': 'whole_robot_in_room', 'region': 'room2', 'margin_m': .02}},
            'phases': [{'id': 'operate', 'entry': ['inside'], 'exit': ['inside'], 'invariants': ['inside']}]}


def test_real_shape_support_endpoints_do_not_certify_motion(scene):
    frames = [frame(scene, [-.8, -.53, 0., 0., 0., 0., 0.]), frame(scene, [.8, -.53, 0., 0., 0., 0., 0.])]
    evaluator = SemanticPlanEvaluator(scene, plan())
    assert all(evaluator.evaluate('inside', configuration(f))['satisfied'] for f in frames)
    report = evaluator.audit(frames)
    assert not report['satisfied'] and report['violations'] > 0
    assert report['counterexamples'][0]['scope'] == 'invariants'
    assert report['phases'][0]['predicates']['inside']['method'] == 'analytical_shape_support'


def test_room_semantics_change_by_phase_and_reference_must_share_endpoints(scene):
    supplied = plan()
    supplied['phases'] = [{'id': 'navigate', 'exit': ['inside']}, {'id': 'operate', 'invariants': ['inside']}]
    frames = [frame(scene, base=(-3., 0., 0.), phase='navigate'), frame(scene, phase='navigate'), frame(scene)]
    assert SemanticPlanEvaluator(scene, supplied).audit(frames)['satisfied']
    frames[-1]['base_q'][0] += .1
    with pytest.raises(ValueError, match='motion bridge required'):
        SemanticPlanEvaluator(scene, supplied).audit(frames)


def test_corridor_checks_all_robot_collision_volumes(scene, monkeypatch):
    monkeypatch.setitem(scene.config['semantic_regions'], 'customers', {'x_bounds': [-.1, .1], 'y_bounds': [-1., 1.]})
    supplied = plan()
    supplied['predicates']['clear'] = {'kind': 'keep_clear', 'region': 'customers', 'margin_m': .02}
    evaluator = SemanticPlanEvaluator(scene, supplied)
    assert not evaluator.evaluate('clear', configuration(frame(scene, base=(0., 0., 0.))))['satisfied']
    result = evaluator.evaluate('clear', configuration(frame(scene, base=(.8, 0., 0.))))
    assert result['satisfied'] and result['method'] == 'conservative_axis_separation'


def test_unknown_missing_empty_nonfinite_and_budget_fail_closed(scene):
    supplied = plan()
    supplied['predicates']['inside']['kind'] = 'learned_unavailable'
    with pytest.raises(ValueError, match='Unsupported'):
        SemanticPlanEvaluator(scene, supplied)
    supplied = plan()
    supplied['phases'][0]['invariants'] = ['typo']
    with pytest.raises(ValueError, match='Unknown predicate'):
        SemanticPlanEvaluator(scene, supplied)
    evaluator = SemanticPlanEvaluator(scene, plan())
    with pytest.raises(ValueError, match='empty'):
        evaluator.audit([])
    with pytest.raises(ValueError, match='phase order'):
        evaluator.audit([frame(scene, phase='missing')])
    with pytest.raises(ValueError, match='Finite'):
        evaluator.audit([frame(scene, base=(float('nan'), 0., 0.))])
    with pytest.raises(RuntimeError, match='budget exhausted'):
        evaluator.audit([frame(scene)], max_samples=1)
    assert evaluator.audit([frame(scene)])['satisfied']  # Singleton still evaluated.


def test_measured_execution_uses_explicit_tracking_tolerances(scene):
    supplied = {'version': 1, 'predicates': {'docked': {'kind': 'configuration_equals', 'indices': [0, 1, 2],
        'target': 'dock', 'tolerance': 1e-6, 'execution_tolerance': [.01, .01, .01]}},
        'phases': [{'id': 'operate', 'invariants': ['docked']}]}
    frames = [frame(scene, base=(.455, 0., 0.))]
    assert not SemanticPlanEvaluator(scene, supplied, bindings={'dock': [.45, 0., 0.]}).audit(frames)['satisfied']
    assert SemanticPlanEvaluator(scene, supplied, bindings={'dock': [.45, 0., 0.]}, execution=True).audit(frames)['satisfied']


def test_sequence_backtracks_to_previous_goal_when_later_bridge_fails(scene):
    initial = frame(scene, base=(.45, 0., 0.))
    bad = frame(scene, base=(.55, 0., 0.), phase='first')
    good = frame(scene, base=(.65, 0., 0.), phase='first')
    end = frame(scene, base=(.75, 0., 0.), phase='second')
    def connect(a, b, depth):
        if depth == 1 and a['base_q'][0] == .55:
            raise RuntimeError('Later bridge rejected')
        return [a, b]
    supplied = copy.deepcopy(initial)
    result = ground_sequence([[[bad], [good]], [[end]]], initial, connect=connect,
        validate=lambda frames, depth: True)
    assert result['satisfied'] and result['attempts'] == 4
    assert any(e['status'] == 'backtracked' for e in result['events'])
    assert result['frames'][-1] == end and initial == supplied
    result = ground_sequence([[[bad], [good]], [[end]]], initial, connect=connect,
        validate=lambda frames, depth: True, max_attempts=2)
    assert not result['satisfied'] and result['frames'] is None


def test_sequence_never_accepts_a_bridge_with_wrong_arm_or_fingers(scene):
    initial, end = frame(scene), frame(scene, base=(.65, 0., 0.))
    def connect(a, b, depth):
        b['gripper_q'] = [0., 0.]
        return [a, b]
    result = ground_sequence([[[end]]], initial, connect=connect, validate=lambda frames, depth: True)
    assert not result['satisfied'] and 'complete endpoints' in result['events'][0]['reason']


def test_actual_ik_uses_sampled_position_and_rotated_base_without_moving_cup(scene):
    from src.semantic_refinement import make_candidate_config
    from src.grasp_planner import plan_pick
    base = np.array([.45, 0., -.15])
    candidate = {'id': 'pose', 'base_q': base.tolist(), 'grasp_rpy': [0., 0., 0.],
                 'grasp_position_offset_m': [0., 0., -.004]}
    config = make_candidate_config(scene.config, candidate)
    current = build_scene(config=config, visualize=False)
    pick = plan_pick(current, base)
    set_robot_configuration(current, base, pick.grasp_q, current.gripper_open_positions)
    actual = current.plant.EvalBodyPoseInWorld(current.plant_context, current.gripper_body).multiply(
        np.asarray(config['gripper']['grasp_point_in_body']))
    expected = np.asarray(config['objects']['cup']['center']) + config['pick_task']['grasp_offset'] + np.array([0., 0., -.004])
    assert np.max(np.abs(actual-expected)) <= config['pick_task']['position_tolerance']+1e-7
    assert np.allclose(pick.grasp_pose['position_world_m'], expected)
    assert config['objects']['cup'] == scene.config['objects']['cup']


def test_cup_default_plan_matches_tracked_supplied_plan(scene):
    from src.semantic_plan import cup_semantic_plan
    supplied = load_scene_config('configs/professor_grounding.yaml')['semantic_plan']
    # Both independently state the same phase order and whole-motion contract.
    default = cup_semantic_plan(scene.config)
    assert [p['id'] for p in supplied['phases']] == [p['id'] for p in default['phases']]
    assert all('inside' in p['invariants'] for p in supplied['phases'][1:])


def test_explicit_plan_gate_applies_even_to_geometry_ablation():
    from src.semantic_policy import accepts
    for policy in ('geometry_only', 'goal_only', 'full_motion'):
        assert not accepts(policy, geometry=True, goals=True, motion=True, plan=False)


def test_generic_bridge_rejects_payload_and_closed_fingers_before_planning(scene, tmp_path):
    from src.semantic_motion_bridge import connect_open_gripper
    start, end = frame(scene), frame(scene, base=(.65, 0., 0.))
    end['payload_mode'] = 'held'
    with pytest.raises(ValueError, match='payload/contact'):
        connect_open_gripper(scene, start, end, plan=plan(), phase='operate', artifact_dir=tmp_path/'payload')
    end.pop('payload_mode')
    end['gripper_q'] = [0., 0.]
    with pytest.raises(ValueError, match='open gripper'):
        connect_open_gripper(scene, start, end, plan=plan(), phase='operate', artifact_dir=tmp_path/'closed')
    assert not list(tmp_path.iterdir())
