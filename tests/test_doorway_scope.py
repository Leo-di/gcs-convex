import copy
import numpy as np
import pytest
from src.build_scene import build_scene
from src.scene_spec import load_scene_config
from src.semantic_plan import SemanticPlanEvaluator
from src.semantic_refinement import screen_candidates
from src.doorway_scope_study import method_config


@pytest.fixture(scope='module')
def scene():
    return build_scene(config=load_scene_config('configs/scene_semantic_choice_v6.yaml'), visualize=False)


def evaluator(scene):
    return SemanticPlanEvaluator(scene, {'version': 1, 'predicates': {
        'base': {'kind': 'base_in_room', 'region': 'room2', 'margin_m': .02},
        'body': {'kind': 'whole_robot_in_room', 'region': 'room2', 'margin_m': .02}},
        'phases': [{'id': 'operate', 'invariants': ['base']}]})


def test_base_volume_not_center_and_rotation(scene):
    ev = evaluator(scene)
    arm = scene.config['pick_task']['navigation_arm_q']
    assert not ev.evaluate('base', np.r_[[.2, 0, 0], arm, scene.gripper_open_positions])['satisfied']
    assert ev.evaluate('base', np.r_[[.45, 0, 0], arm, scene.gripper_open_positions])['satisfied']
    assert not ev.evaluate('base', np.r_[[.45, 0, np.pi/4], arm, scene.gripper_open_positions])['satisfied']


def test_base_pass_does_not_imply_arm_pass(scene):
    ev = evaluator(scene)
    q = np.r_[[.45, 0, 0], [0, -.53, 0, 0, 0, 0, 0], scene.gripper_open_positions]
    assert ev.evaluate('base', q)['satisfied']
    assert not ev.evaluate('body', q)['satisfied']
    frames = [dict(base_q=q[:3].tolist(), arm_q=q[3:10].tolist(), gripper_q=q[10:].tolist(), phase='operate')]*2
    assert ev.audit(frames)['satisfied']
    frames = copy.deepcopy(frames); frames[-1]['base_q'][0] = .2
    assert not ev.audit(frames)['satisfied']


def test_study_preserves_scene_and_screens_base_goal():
    original = load_scene_config('configs/scene_doorway_table.yaml')
    before = copy.deepcopy(original)
    cfg = method_config(original, 'base_only')
    assert original == before
    for key in ['objects', 'world', 'robot', 'arm', 'gripper', 'green_room', 'semantic_refinement']:
        assert cfg[key] == original[key]
    rows = screen_candidates(cfg, [dict(id='out', base_q=[-.5, 0, 0], grasp_rpy=[0,0,0]),
                                   dict(id='in', base_q=[.5, 0, 0], grasp_rpy=[0,0,0])])
    assert rows[0]['reason'] == 'base_goal_outside_room'
    assert rows[1]['status'] == 'eligible'
    with pytest.raises(ValueError): method_config(original, 'bad')
