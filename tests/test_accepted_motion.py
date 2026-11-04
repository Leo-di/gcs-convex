import copy
import json
import numpy as np
import pytest

from src.accepted_motion import AcceptedMotion, CUP_PHASES, accept_cup_phase, compose_cup_motion
from src.build_scene import build_scene
from src.green_room_gcs import append_motion
from src.scene_spec import load_scene_config
from src.semantic_plan import configuration, cup_semantic_plan


@pytest.fixture(scope='module')
def scene():
    cfg = load_scene_config('configs/scene_semantic_choice_v6.yaml')
    cfg['grounding_semantic_plan'] = cup_semantic_plan(cfg)
    return build_scene(config=cfg, visualize=False)


def frame(scene, x=.45, phase='navigate_to_pickup'):
    return dict(base_q=[x, 0., 0.], arm_q=scene.config['pick_task']['navigation_arm_q'],
                gripper_q=scene.gripper_open_positions.tolist(), phase=phase)


def accept(scene, frames, **kwargs):
    return accept_cup_phase(scene, frames, np.array([.45, 0., 0.]),
        np.array(scene.config['pick_task']['navigation_arm_q']), None,
        start=configuration(frames[0]), end=configuration(frames[-1]), **kwargs)


def test_ingress_may_start_outside_but_must_finish_inside(scene):
    frames = [frame(scene, -1.), frame(scene)]
    segment = accept(scene, frames)
    assert segment.report['policy_satisfied']
    assert segment.report['navigation_endpoint_inside']['satisfied']
    with pytest.raises(RuntimeError, match='rejected'):
        accept(scene, [frame(scene, -1.), frame(scene, -.5)])


def test_operation_cannot_start_outside_or_move_base(scene):
    with pytest.raises(RuntimeError, match='rejected'):
        accept(scene, [frame(scene, -.5, 'approach_cup')])
    with pytest.raises(RuntimeError, match='rejected'):
        accept(scene, [frame(scene, .45, 'approach_cup'), frame(scene, .55, 'approach_cup')])


def test_wrong_full_endpoint_rejected_before_audit(scene):
    f = frame(scene)
    expected = configuration(f); expected[-1] += .001
    with pytest.raises(ValueError, match='endpoint mismatch'):
        accept_cup_phase(scene, [f], [.45, 0., 0.], f['arm_q'], None,
                         start=configuration(f), end=expected)


def test_budget_and_nonfinite_fail_closed(scene):
    with pytest.raises(RuntimeError, match='budget exhausted'):
        accept(scene, [frame(scene, -1.), frame(scene)], max_samples=2)
    f = frame(scene); f['base_q'][0] = float('nan')
    with pytest.raises(ValueError, match='Finite'):
        accept(scene, [f])


def test_accepted_snapshot_cannot_be_invalidated_by_caller(scene):
    frames = [frame(scene)]
    segment = accept(scene, frames)
    frames[0]['base_q'][0] = -1.
    returned = segment.frames; returned[0]['gripper_q'][0] = 1.
    assert segment.frames[0]['base_q'][0] == .45
    assert segment.frames[0]['gripper_q'][0] == scene.gripper_open_positions[0]


def test_close_and_payload_mode_contracts(scene):
    f = frame(scene, phase='close_gripper')
    g = copy.deepcopy(f); g['gripper_q'] = scene.gripper_closed_positions.tolist()
    g['arm_q'] = list(g['arm_q']); g['arm_q'][0] += .02
    with pytest.raises(ValueError, match='fixed robot pose'):
        accept(scene, [f, g])
    g['phase'] = 'lift_cup'
    with pytest.raises(ValueError, match='attachment transform'):
        accept(scene, [g])


def test_composition_checks_order_all_boundaries_and_task_endpoints(scene):
    # Deliberately fabricated acceptance records isolate composition from planning.
    segments = [AcceptedMotion(p, json.dumps([frame(scene, phase=p)]),
                               json.dumps({'policy_satisfied': True,
                                           'cup_boundary_poses': [np.eye(4).tolist()]*2})) for p in CUP_PHASES]
    q = configuration(frame(scene))
    assert compose_cup_motion(segments, start=q, end=q)[1]['accepted']
    with pytest.raises(ValueError, match='task order'):
        compose_cup_motion(segments[::-1], start=q, end=q)
    report = segments[3].report
    report['cup_boundary_poses'][0][0][3] = .05
    bad_payload = list(segments)
    bad_payload[3] = AcceptedMotion(CUP_PHASES[3], segments[3].frames_json, json.dumps(report))
    with pytest.raises(ValueError, match='Cup pose discontinuity'):
        compose_cup_motion(bad_payload, start=q, end=q)
    broken = copy.deepcopy(segments[2].frames)
    broken[0]['arm_q'][1] += .01
    segments[2] = AcceptedMotion(CUP_PHASES[2], json.dumps(broken), segments[2].report_json)
    with pytest.raises(ValueError, match='explicit bridge required'):
        compose_cup_motion(segments, start=q, end=q)


def test_retimer_rejects_truncated_or_nonfinite_inputs(scene):
    f = frame(scene)
    with pytest.raises(ValueError, match='equal lengths'):
        append_motion([], [f['base_q']], [f['arm_q']]*2, [f['gripper_q']], 'test')
    with pytest.raises(ValueError, match='Finite'):
        append_motion([], [[np.nan, 0., 0.]], [f['arm_q']], [f['gripper_q']], 'test')
