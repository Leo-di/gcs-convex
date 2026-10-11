"""Independent geometry and phase-boundary regressions for green-room tasks."""
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
from pydrake.geometry import Box, Cylinder, Sphere

from src.build_scene import build_scene, set_robot_configuration
from src.green_room_gcs import append_motion, make_operation_cover, semantic_box_bound
from src.scene_spec import load_scene_config
from src.whole_robot_room import WholeRobotRoomOracle, arm_point_reach_bound, shape_support


@pytest.fixture(scope='module')
def scene():
    return build_scene(config=load_scene_config('configs/scene_green_room.yaml'), visualize=False)


def test_base_origin_inside_does_not_imply_footprint_inside(scene):
    oracle = WholeRobotRoomOracle.from_xy_bounds(scene, [.1, 2.9], [-1.4, 1.4], margin=.02)
    tucked = scene.config['pick_task']['navigation_arm_q']
    value = oracle.query(np.r_[.25, 0, 0, tucked])
    assert .1 < .25 < 2.9
    assert not value['satisfied']
    assert value['residual_m'] <= -.17 + 1e-12
    assert any(body.model_instance() == scene.base_instance for body, *_ in oracle.geometry)
    assert {body.model_instance() for body, *_ in oracle.geometry} == {scene.base_instance, scene.arm_instance, scene.gripper_instance}


def test_exact_support_matches_transformed_box_corners(scene):
    normal = np.array([2., -3., 1.]) / np.sqrt(14)
    oracle = WholeRobotRoomOracle(scene, [normal], [.37], margin=.02)
    state = np.r_[1.2, -.1, .37, scene.arm_home_positions]
    value = oracle.query(state)
    supports = []
    for body, shape, local_pose, _ in oracle.geometry:
        pose = scene.plant.EvalBodyPoseInWorld(scene.plant_context, body) @ local_pose
        if isinstance(shape, Box):
            extent = np.array([shape.width(), shape.depth(), shape.height()]) / 2
            supports.extend(normal @ (pose @ (np.array(signs) * extent))
                            for signs in itertools.product((-1, 1), repeat=3))
        elif isinstance(shape, Sphere):
            supports.append(normal @ pose.translation() + shape.radius())
        else:
            pytest.fail('Unexpected shape in frozen robot')
    assert value['residual_m'] == pytest.approx(.37 - max(supports) - .02, abs=1e-12)


def test_gripper_origin_inside_can_hide_link_or_volume_violation():
    data = json.loads((Path(__file__).parent / 'fixtures/arm_boundary.json').read_text())
    example = build_scene(config=data['config'], visualize=False)
    oracle = WholeRobotRoomOracle(example, [[1, 0, 0]], [0])
    value = oracle.query(data['poses']['origin_misses']['state'])
    ee = example.plant.EvalBodyPoseInWorld(example.plant_context, example.gripper_body).translation()
    assert ee[0] < 0
    assert not value['satisfied']


def test_current_query_does_not_mutate_plant(scene):
    oracle = WholeRobotRoomOracle.from_xy_bounds(scene, [.1, 2.9], [-1.4, 1.4])
    set_robot_configuration(scene, [1.2, -.1, .2], scene.arm_home_positions, scene.gripper_closed_positions)
    before = scene.plant.GetPositions(scene.plant_context).copy()
    oracle.query_current()
    np.testing.assert_array_equal(scene.plant.GetPositions(scene.plant_context), before)


def test_cylinder_support_includes_cap_and_radial_extent():
    shape = Cylinder(.05, .18)
    assert shape_support(shape, [0, 0, 1]) == pytest.approx(.09)
    assert shape_support(shape, [1, 0, 0]) == pytest.approx(.05)
    n = np.array([3., 4., 12.]) / 13
    assert shape_support(shape, n) == pytest.approx(.05 * 5 / 13 + .09 * 12 / 13)


def test_box_semantic_bound_is_conservative_at_all_corners(scene):
    oracle = WholeRobotRoomOracle.from_xy_bounds(scene, [.1, 2.9], [-1.4, 1.4], margin=.02)
    center = np.asarray(scene.config['pick_task']['navigation_arm_q'])
    lo, hi = center - .005, center + .005
    base = np.array([1.2, -.1, 0])
    bound = semantic_box_bound(oracle, base, lo, hi, scene.gripper_open_positions, arm_point_reach_bound(scene, oracle))
    assert bound['whole_box_semantic_lower_bound_m'] > 0
    for signs in itertools.product((0, 1), repeat=7):
        arm = np.where(signs, hi, lo)
        assert oracle.query(np.r_[base, arm])['residual_m'] >= bound['whole_box_semantic_lower_bound_m'] - 1e-12


def test_operation_cover_rejects_invalid_dock_before_solving(scene):
    arm = np.asarray(scene.config['pick_task']['navigation_arm_q'])
    with pytest.raises(RuntimeError, match='contained operation box'):
        make_operation_cover(scene, np.array([.25, 0, 0]), np.array([arm, arm + .005]), scene.gripper_open_positions)


def test_stage_transition_shares_entire_configuration_and_rejects_jump():
    frames = []
    base = np.array([[1.2, -.1, 0], [1.2, -.1, 0]])
    arms = np.zeros((2, 7))
    grip = np.tile([-.055, .055], (2, 1))
    append_motion(frames, base, arms, grip, 'navigate_to_pickup')
    append_motion(frames, base, arms, grip, 'approach_cup')
    assert frames[-2]['base_q'] == frames[-1]['base_q']
    assert frames[-2]['arm_q'] == frames[-1]['arm_q']
    with pytest.raises(RuntimeError, match='Configuration jump'):
        append_motion(frames, base, arms + .1, grip, 'approach_cup')


def test_invalid_normals_are_rejected(scene):
    with pytest.raises(ValueError, match='unit room normals'):
        WholeRobotRoomOracle(scene, [[2, 0, 0]], [0])
