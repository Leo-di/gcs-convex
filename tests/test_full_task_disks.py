import numpy as np
from pydrake.math import RigidTransform

from src.build_scene import build_scene, set_robot_configuration
from src.full_task_disks import state_clearance


def test_payload_is_part_of_moving_collision_set():
    scene = build_scene(visualize=False)
    set_robot_configuration(scene, [-3., 0., 0.])
    # Synthetic cup location isolates a payload/wall pair; this is not a
    # physically valid grasp and is used only to test moving-set membership.
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body,
                               RigidTransform([-.2, 1., 1.]))
    empty = state_clearance(scene, False)
    loaded = state_clearance(scene, True)
    assert loaded < .15
    assert empty > loaded + .1


def test_filtered_table_contact_is_restored_for_payload_disk():
    scene = build_scene(visualize=False)
    set_robot_configuration(scene, [-3., 0., 0.])
    # Initial cup rests on table: no positive free-space payload disk exists.
    assert state_clearance(scene, True) <= 1e-8
    assert state_clearance(scene, False) > .1
