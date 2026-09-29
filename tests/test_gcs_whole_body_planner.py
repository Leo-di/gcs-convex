import numpy as np

from src.gcs_whole_body_planner import plan_carry_whole_body_gcs
from src.scene_spec import load_scene_config


def test_whole_body_gcs_uses_compact_arm_through_doorway() -> None:
    config = load_scene_config()
    start_arm = np.zeros(7)
    carry_arm = np.full(7, 0.10)
    goal_arm = np.full(7, 0.20)
    plan = plan_carry_whole_body_gcs(
        config,
        start_base_q=[1.2, -0.1, 0.0],
        start_arm_q=start_arm,
        carry_arm_q=carry_arm,
        goal_base_q=[-2.35, 1.3, np.pi / 2.0],
        goal_arm_q=goal_arm,
        allowed_region_names=[
            "room2_pickup_lane",
            "doorway_lane",
            "room1_central_lane",
            "room1_upper_lane",
        ],
    )

    assert plan.dimension == 10
    np.testing.assert_allclose(plan.arm_configurations[0], start_arm, atol=1e-6)
    np.testing.assert_allclose(plan.arm_configurations[-1], goal_arm, atol=1e-6)
    doorway = plan.configurations[np.abs(plan.configurations[:, 0]) <= 0.25]
    assert len(doorway) > 0
    assert np.max(np.abs(doorway[:, 3:] - carry_arm)) <= 0.015 + 1e-5
    assert "doorway_lane" in plan.region_sequence
