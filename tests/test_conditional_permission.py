import copy
import numpy as np
import pytest

from src.build_scene import build_scene, set_robot_configuration
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config, validate_task, search, TransportRejected
from src.conditional_permission import ConditionalPermissionEvaluator, MeasuredHolding, PermissionExecutionMonitor


def spec_with_permission():
    spec = load_scene_config('configs/room_entry_transport.yaml')
    spec['conditional_permissions'] = [dict(id='held_only', requires_holding='cup',
        lower_xyz=[.8, -.4, 0.], upper_xyz=[1., .3, .45], margin_m=.02)]
    return spec


def test_same_robot_geometry_allowed_only_with_named_held_object():
    cfg = make_config(spec_with_permission())
    scene = build_scene(config=cfg, visualize=False)
    set_robot_configuration(scene, [.9, 0., 0.], cfg['pick_task']['navigation_arm_q'], scene.gripper_open_positions)
    gate = ConditionalPermissionEvaluator(scene)
    assert not gate.current()['satisfied']
    assert not gate.current(['another_object'])['satisfied']
    assert gate.current(['cup'])['satisfied']
    set_robot_configuration(scene, [.45, 0., 0.], cfg['pick_task']['navigation_arm_q'], scene.gripper_open_positions)
    assert gate.current()['satisfied']


def observation(t, left=1., support=0., relative=(0., 0., 0.)):
    return dict(time_s=t, phase='arbitrary.phase', objects={'cup': dict(
        contacts_N=dict(left_finger=left, right_finger=1., table=support, dropoff_table=0., dropoff_bin=0.),
        relative_translation_m=list(relative))})


def test_measured_holding_revokes_on_contact_loss_support_or_slip():
    gate = MeasuredHolding(['cup'])
    assert not gate.observe(observation(0))[0]
    assert gate.observe(observation(.1))[0] == {'cup'}
    assert not gate.observe(observation(.11, relative=(.011, 0, 0)))[0]
    assert not gate.observe(observation(.12, left=0.))[0]
    assert not gate.observe(observation(.13))[0]
    assert gate.observe(observation(.23))[0] == {'cup'}
    assert not gate.observe(observation(.24, support=.2))[0]
    missing = observation(.25)
    del missing['objects']['cup']['contacts_N']['table']
    assert not gate.observe(missing)[0]
    with pytest.raises(ValueError, match='monotone'):
        gate.observe(observation(.1))


def test_actual_geometry_permission_revoked_at_same_pose_when_contact_disappears():
    cfg = make_config(spec_with_permission())
    scene = build_scene(config=cfg, visualize=False)
    set_robot_configuration(scene, [.9, 0., 0.], cfg['pick_task']['navigation_arm_q'], scene.gripper_closed_positions)
    monitor = PermissionExecutionMonitor(scene)
    assert not monitor.observe(observation(0.))['satisfied']
    assert monitor.observe(observation(.1))['satisfied']
    assert not monitor.observe(observation(.11, left=0.))['satisfied']


def test_locked_empty_hand_candidate_is_rejected_before_ik(monkeypatch, tmp_path):
    from src.multiobject_grounding import plan_action
    from src import grasp_planner
    spec = spec_with_permission()
    cfg = make_config(spec)
    def unexpected_ik(*args, **kwargs):
        pytest.fail('Locked candidate must be rejected before IK')
    monkeypatch.setattr(grasp_planner, 'plan_pick', unexpected_ik)
    with pytest.raises(TransportRejected) as exc:
        plan_action(cfg, spec['actions'][0], [.9, 0., 0.], {}, tmp_path)
    assert exc.value.stage == 'empty_hand_pickup_permission_rejected'
    assert (tmp_path/'empty_hand_permission_rejection.json').exists()


@pytest.mark.parametrize('change', [dict(requires_holding='unknown'), dict(upper_xyz=[.7, .3, .45]), dict(margin_m=float('nan'))])
def test_invalid_permissions_fail_closed(change):
    spec = spec_with_permission()
    spec['conditional_permissions'][0].update(change)
    with pytest.raises(ValueError):
        validate_task(spec)


def test_candidate_rejection_continues_with_same_world_and_selected_grasp(tmp_path):
    spec = spec_with_permission()
    action = spec['actions'][0]
    action['pickup_candidates'] = [dict(base_q=action['pickup_base'], grasp_rpy=[0.,0.,y]) for y in (-.2,-.18)]
    action['pickup_candidates'][1]['base_q'] = [.41, 0., 0.]
    spec['max_expansions'] = 2
    worlds = []
    def ground(config, world, action, placement, spec, folder):
        worlds.append(copy.deepcopy(world))
        if action['grasp_rpy'][2] == -.2:
            raise TransportRejected('closed_grasp_candidate_rejected', {})
        return [world['robot']], world, {}
    result = search(make_config(spec), spec, tmp_path, ground=ground)
    assert result['satisfied'] and result['expansions'] == 2
    assert worlds[0]['poses'] == worlds[1]['poses']
    assert worlds[0]['supports'] == worlds[1]['supports']
    assert worlds[1]['robot']['base_q'] == [.41, 0., 0.]
    assert result['initial_world'] == worlds[1]
    assert result['selected'][0]['pickup_candidate_index'] == 1
    assert result['selected'][0]['pickup_candidate']['grasp_rpy'][2] == -.18


def test_early_grasp_rejection_uses_real_geometry_and_saves_evidence(tmp_path):
    from src.grasp_planner import plan_pick
    from src.multiobject_grounding import accept_closed_grasp, GraspCandidateRejected
    cfg = load_scene_config('tests/fixtures/legacy_grasp.yaml')
    scene = build_scene(config=cfg, visualize=False)
    base = np.array([1.2, -.1, 0.])
    pick = plan_pick(scene, base)
    with pytest.raises(GraspCandidateRejected) as exc:
        accept_closed_grasp(scene, base, pick.grasp_q, tmp_path, 'ik')
    assert exc.value.stage == 'closed_grasp_candidate_rejected'
    assert (tmp_path/'ik_grasp_acceptance.json').exists()
