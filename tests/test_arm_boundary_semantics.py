import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from src.arm_boundary_semantics import ArmBoundaryOracle
from src.build_scene import build_scene
from src.collision_checker import evaluate_collision_status


@pytest.fixture(scope='module')
def example():
    data = json.loads((Path(__file__).parent / 'fixtures/arm_boundary.json').read_text())
    scene = build_scene(config=data['config'], visualize=False)
    return scene, data['poses']


def test_same_base_collision_free_but_different_arm_semantics(example):
    scene, poses = example
    oracle = ArmBoundaryOracle(scene)
    values = {}
    for name, record in poses.items():
        state = np.array(record['state'])
        np.testing.assert_array_equal(state[:3], [-.6, 0, 0])
        assert evaluate_collision_status(scene, state[:3], arm_q=state[3:]).collision_free
        values[name] = oracle.query(state)
        assert values[name]['base_center_residual_m'] > 0
    assert values['safe']['satisfied']
    assert not values['unsafe']['satisfied']
    assert not values['origin_misses']['satisfied']
    assert values['origin_misses']['gripper_origin_residual_m'] > 0


def test_support_matches_all_transformed_box_corners(example):
    scene, poses = example
    # Test an oblique plane too: x-only tests can hide rotation mistakes.
    normal = np.array([2., -3., 1.]) / np.sqrt(14)
    oracle = ArmBoundaryOracle(scene, normal, .37)
    for record in poses.values():
        value = oracle.query(record['state'])
        maxima = []
        for body, _, local_pose, extent, radius, _ in oracle.geometry:
            pose = scene.plant.EvalBodyPoseInWorld(scene.plant_context, body) @ local_pose
            if extent is not None:
                points = [pose @ (np.array(signs) * extent)
                          for signs in itertools.product((-1, 1), repeat=3)]
                maxima.append(max(normal @ point for point in points))
            else:
                maxima.append(normal @ (pose.translation() + radius * normal))
        assert value['residual_m'] == pytest.approx(.37 - max(maxima), abs=1e-12)
