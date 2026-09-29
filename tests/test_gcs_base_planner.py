import numpy as np

from src.gcs_base_planner import plan_base_gcs
from src.scene_spec import load_scene_config


def test_gcs_connects_rooms_through_doorway() -> None:
    config = load_scene_config()
    plan = plan_base_gcs(
        config,
        np.array([-3.0, 0.0, 0.0]),
        np.array([1.2, -0.1, 0.0]),
    )

    np.testing.assert_allclose(plan.configurations[0], [-3.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(plan.configurations[-1], [1.2, -0.1, 0.0], atol=1e-6)
    assert "doorway_lane" in plan.region_sequence
    assert plan.path_length > 4.0
    assert plan.number_of_segments >= 3


def test_gcs_reaches_dropoff_lane() -> None:
    config = load_scene_config()
    plan = plan_base_gcs(
        config,
        np.array([1.2, -0.1, 0.0]),
        np.array([-2.35, 1.3, np.pi / 2.0]),
    )

    np.testing.assert_allclose(plan.configurations[-1], [-2.35, 1.3, np.pi / 2.0], atol=1e-6)
    assert "room1_upper_lane" in plan.region_sequence
    assert plan.angular_travel >= np.pi / 2.0 - 1e-3

    # Check the non-overlap core of the doorway. In the overlap with the
    # adjacent room region, either convex set may legitimately be active.
    doorway = plan.configurations[
        np.abs(plan.configurations[:, 0]) <= 0.25 + 1e-6
    ]
    assert len(doorway) > 0
    assert np.max(np.abs(doorway[:, 2])) <= 0.30 + 1e-5
