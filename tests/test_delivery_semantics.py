import numpy as np

from src.build_scene import build_scene, set_robot_configuration
from src.delivery_planner import plan_delivery
from src.grasp_planner import plan_pick
from src.task_semantics import cup_bin_residuals


def test_release_preserves_achieved_cup_pose_and_support_guard():
    scene = build_scene(visualize=False)
    pick = plan_pick(scene, np.asarray(scene.config["reference_path"][-1]))
    delivery = plan_delivery(scene, pick)
    set_robot_configuration(scene, delivery.base_q, delivery.place_q,
                            scene.gripper_closed_positions)
    achieved = (scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
                @ delivery.X_GC)
    np.testing.assert_allclose(delivery.place_cup_pose.GetAsMatrix4(),
                               achieved.GetAsMatrix4(), atol=1e-12)
    residuals = cup_bin_residuals(achieved.GetAsMatrix4(), scene.config)
    assert all(value >= 0 for value in residuals.values()), residuals
