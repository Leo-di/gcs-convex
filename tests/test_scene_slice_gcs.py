import numpy as np
import pytest

from src.build_scene import build_scene
from src.scene_slice_gcs import SliceOracle


@pytest.fixture(scope="module")
def oracle():
    scene = build_scene(visualize=False)
    return SliceOracle(scene, scene.arm_home_positions, scene.gripper_open_positions)


def test_whole_robot_clearance_disk(oracle):
    center = np.array([-3., 0.])
    distance, self_distance, _ = oracle.query(center)
    assert distance > .02 and self_distance > 0
    radius = distance-.02
    for angle in np.linspace(0, 2*np.pi, 24, endpoint=False):
        value, current_self, _ = oracle.query(center+radius*np.array([np.cos(angle),np.sin(angle)]))
        assert value >= .02-1e-6
        assert current_self == pytest.approx(self_distance, abs=1e-8)


def test_table_collision_uses_robot_volume_not_center_point(oracle):
    # Base center is below the tabletop, but full robot occupies the table.
    distance, _, bodies = oracle.query([1.6, .65])
    assert distance < 0
    assert bodies is not None


def test_translation_keeps_slice_joint_configuration(oracle):
    oracle.query([0., 0.])
    scene = oracle.scene
    np.testing.assert_allclose(scene.plant.GetPositions(scene.plant_context, scene.arm_instance), oracle.arm)
    np.testing.assert_allclose(scene.plant.GetPositions(scene.plant_context, scene.gripper_instance), oracle.gripper)
