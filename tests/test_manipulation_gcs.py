"""Tests for mode-specific manipulation IRIS regions and GCS."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.manipulation_gcs import (
    MODE_SPECS,
    SITE_REGION_NAMES,
    _phase_domain,
    _manipulation_spec,
    _reference_corridor_cover,
    select_phase_seeds,
)
from src.scene_spec import load_scene_config


def test_mode_specs_cover_the_four_manipulation_phases():
    assert set(MODE_SPECS) == {"pregrasp", "grasp", "release", "retreat"}
    # The cup is attached exactly while it moves with the gripper.
    assert [MODE_SPECS[m].attached for m in ("pregrasp", "grasp", "release", "retreat")] == [
        False,
        True,
        True,
        False,
    ]
    # Grasping modes close the gripper, approach and retreat open it.
    assert [MODE_SPECS[m].gripper_closed for m in ("pregrasp", "grasp", "release", "retreat")] == [
        False,
        True,
        True,
        False,
    ]
    # Only retreat plans against the cup resting in the bin.
    assert [MODE_SPECS[m].cup_at_final_pose for m in ("pregrasp", "grasp", "release", "retreat")] == [
        False,
        False,
        False,
        True,
    ]
    assert MODE_SPECS["pregrasp"].site == "pickup"
    assert MODE_SPECS["grasp"].site == "pickup"
    assert MODE_SPECS["release"].site == "dropoff"
    assert MODE_SPECS["retreat"].site == "dropoff"


def test_site_regions_exist_in_scene_config():
    config = load_scene_config()
    names = {spec["name"] for spec in config["gcs_base_planning"]["regions"]}
    for site, region in SITE_REGION_NAMES.items():
        assert region in names, f"site {site} references unknown region {region}"


def test_phase_seeds_include_endpoints_and_respect_threshold():
    # A smooth ramp: greedy selection must space seeds by at least the
    # threshold while always keeping both endpoints.  The forced endpoint
    # can leave a smaller final gap, never a larger interior one.
    values = np.linspace(0.0, 2.0, 101)
    reference = np.tile(values[:, None], (1, 10))
    weights = np.ones(10)
    seeds = select_phase_seeds(reference, weights, threshold=0.9)
    np.testing.assert_allclose(seeds[0], reference[0])
    np.testing.assert_allclose(seeds[-1], reference[-1])
    distances = np.linalg.norm(np.diff(seeds, axis=0), axis=1)
    # Interior gaps respect the threshold.
    assert np.all(distances[:-1] >= 0.9 - 1e-9)
    # The greedy stride cannot overshoot the threshold by more than one
    # reference step (2 * sqrt(10) * 0.02 here).
    assert np.all(distances <= 0.9 + 2.0 * np.sqrt(10.0) * 0.02 + 1e-9)
    # Every seed is a reference sample.
    for seed in seeds:
        assert any(np.allclose(seed, row) for row in reference)


def test_phase_seeds_reject_constant_reference():
    reference = np.zeros((5, 10))
    with pytest.raises(ValueError, match="constant"):
        select_phase_seeds(reference, np.ones(10), threshold=0.5)


def test_phase_domain_clips_joints_and_uses_site_box():
    config = load_scene_config()
    scene = SimpleNamespace(config=config)
    spec = _manipulation_spec(config)
    site = next(
        item
        for item in config["gcs_base_planning"]["regions"]
        if item["name"] == SITE_REGION_NAMES["pickup"]
    )
    seed = np.array([1.1, -0.1, 0.0] + [0.0] * 7)
    seed[3] = 2.95  # Near the first joint's upper limit (2.967).
    domain = _phase_domain(scene, seed, site, spec)
    assert domain.PointInSet(seed, 1e-9)
    # The clipped joint window still reaches the plant limit.
    joint_upper = np.asarray(config["whole_body_planning"]["joint_upper_limits"])
    at_limit = seed.copy()
    at_limit[3] = joint_upper[0]
    assert domain.PointInSet(at_limit, 1e-9)
    # Beyond the joint radius (on a joint that is not clipped) is excluded.
    beyond = seed.copy()
    beyond[4] = seed[4] + spec["joint_radius"] + 0.1
    assert not domain.PointInSet(beyond, 1e-9)
    # The base stays inside the site lane box and yaw bounds.
    outside_base = seed.copy()
    outside_base[0] = float(site["upper"][0]) + 0.5
    assert not domain.PointInSet(outside_base, 1e-9)
    outside_yaw = seed.copy()
    outside_yaw[2] = float(site["yaw_bounds"][1]) + 0.1
    assert not domain.PointInSet(outside_yaw, 1e-9)


def test_manipulation_margin_can_be_tighter_than_carrying_margin():
    config = load_scene_config()
    spec = _manipulation_spec(config)
    manipulation_margin = config["manipulation_gcs"][
        "configuration_space_margin"
    ]
    carrying_margin = config["iris_np2_cover"][
        "configuration_space_margin"
    ]
    assert spec["configuration_space_margin"] == manipulation_margin
    assert 0.0 < manipulation_margin < carrying_margin
    assert manipulation_margin < config["pick_task"]["minimum_clearance"]


def test_pregrasp_uses_ordered_contact_corridor():
    config = load_scene_config()
    spec = config["manipulation_gcs"]
    assert spec["pregrasp_region_method"] == "reference_corridor"
    assert spec["contact_corridor_stride"] >= 1
    assert 0.0 < spec["contact_corridor_padding"] < 0.01


def test_reference_corridor_has_only_ordered_overlapping_edges():
    config = load_scene_config()
    scene = SimpleNamespace(config=config)
    spec = _manipulation_spec(config)
    site = next(
        item
        for item in config["gcs_base_planning"]["regions"]
        if item["name"] == SITE_REGION_NAMES["pickup"]
    )
    reference = np.zeros((7, 10))
    reference[:, :3] = [1.1, -0.1, 0.0]
    reference[:, 3] = np.linspace(0.0, 0.3, len(reference))
    regions, results, waypoints, edges = _reference_corridor_cover(
        scene, reference, site, spec
    )
    assert len(regions) == len(waypoints) - 1 == len(results)
    assert edges == [(index, index + 1) for index in range(len(regions) - 1)]
    assert regions[0].PointInSet(reference[0])
    assert regions[-1].PointInSet(reference[-1])
    assert all(
        regions[index].IntersectsWith(regions[index + 1])
        for index in range(len(regions) - 1)
    )


def test_disabled_manipulation_gcs_returns_none_without_planning():
    from src.animate_delivery_demo import _try_manipulation_phase

    scene = SimpleNamespace(config={"manipulation_gcs": {"enabled": False}})
    attempt: dict = {}
    result = _try_manipulation_phase(
        scene, "grasp", np.zeros((3, 10)), attempt, {},
    )
    assert result is None
    assert attempt == {}


def test_main_scene_audit_rejects_filtered_checker_shortcut(monkeypatch):
    from src import animate_delivery_demo, manipulation_gcs

    fake_plan = SimpleNamespace(configurations=np.zeros((2, 10)))
    monkeypatch.setattr(
        manipulation_gcs,
        "plan_manipulation_phase",
        lambda **kwargs: fake_plan,
    )
    monkeypatch.setattr(
        manipulation_gcs,
        "phase_plan_to_dict",
        lambda plan: {"mode": "pregrasp"},
    )
    monkeypatch.setattr(
        animate_delivery_demo,
        "_audit_manipulation_phase_geometry",
        lambda *args, **kwargs: {
            "satisfied": False,
            "violation_count": 1,
        },
    )
    scene = SimpleNamespace(config={"manipulation_gcs": {"enabled": True}})
    attempt = {}
    result = animate_delivery_demo._try_manipulation_phase(
        scene,
        "pregrasp",
        np.zeros((2, 10)),
        attempt,
        {},
    )
    assert result is None
    report = attempt["pregrasp_manipulation_gcs"]
    assert report["status"] == "fallback_to_reference"
    assert report["rejected_plan"]["main_scene_geometry_audit"][
        "violation_count"
    ] == 1


def test_plan_manipulation_phase_grasp_mode_end_to_end():
    """Grow attached-cup regions for the lift and solve through them."""

    from src.build_scene import build_scene
    from src.delivery_planner import plan_delivery
    from src.grasp_planner import IKSolveError, plan_pick
    from src.manipulation_gcs import phase_plan_to_dict, plan_manipulation_phase

    config = load_scene_config()
    scene = build_scene(config=config, visualize=False)
    pick_plan = None
    for candidate in config["pick_task"]["candidates"]:
        try:
            pick_plan = plan_pick(
                scene, np.asarray(candidate["base_q"], dtype=float)
            )
            break
        except IKSolveError:
            continue
    assert pick_plan is not None, "no configured pickup candidate solved IK"
    delivery_plan = plan_delivery(scene, pick_plan)

    reference = np.asarray(
        [
            np.concatenate([pick_plan.base_q, arm_q])
            for arm_q in np.linspace(
                np.asarray(pick_plan.grasp_q, dtype=float),
                np.asarray(pick_plan.lift_q, dtype=float),
                45,
            )
        ],
        dtype=float,
    )
    plan = plan_manipulation_phase(
        mode="grasp",
        config=config,
        reference_q10=reference,
        X_GC=delivery_plan.X_GC,
    )
    # Exact endpoints are preserved.
    np.testing.assert_allclose(plan.configurations[0], reference[0], atol=1e-9)
    np.testing.assert_allclose(plan.configurations[-1], reference[-1], atol=1e-9)
    # Every region passed its independent holdout audit with a seed inside.
    assert plan.region_results
    for item in plan.region_results:
        assert item.region.PointInSet(item.seed)
        assert item.holdout_collisions == 0
    # The trajectory audit found no collisions and visits at least one region.
    assert plan.audit_collisions == 0
    assert plan.region_sequence
    summary = phase_plan_to_dict(plan)
    assert summary["mode"] == "grasp"
    assert summary["number_of_regions"] == len(plan.region_results)
    assert summary["trajectory_audit"]["collisions"] == 0
