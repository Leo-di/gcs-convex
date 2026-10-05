from copy import deepcopy

import numpy as np
import pytest

from src.scene_spec import load_scene_config
from src.semantic_monitor import monitor_delivery
from src.task_semantics import cup_bin_residuals, near_pickup_residual


def valid_trace():
    config = load_scene_config()
    cup = np.eye(4)
    cup[:3, 3] = config["objects"]["cup"]["center"]
    gripper = cup.copy()
    gripper[:3, 3] += (np.array(config["pick_task"]["grasp_offset"]) -
                         np.array(config["gripper"]["grasp_point_in_body"]))
    relation = np.linalg.inv(gripper) @ cup

    def row(position, attached, base, opened=False, retreat=False):
        c = cup.copy()
        c[:3, 3] = position
        g = c @ np.linalg.inv(relation)
        if retreat:
            g[2, 3] += 0.3
        return {"base_q": base, "cup_pose": c.tolist(), "gripper_pose": g.tolist(),
                "gripper_q": config["gripper"]["open_positions" if opened else "closed_positions"],
                "attached": attached, "collision_residual": 0.01}

    pickup = config["objects"]["cup"]["center"]
    placed = config["delivery_task"]["place_cup_center"]
    return config, [
        row(pickup, False, [1.2, -0.1, 0]),
        row(pickup, True, [1.2, -0.1, 0]),
        row(np.array(pickup) + [0, 0, 0.3], True, [1.2, -0.1, 0]),
        row(placed, True, [-2.35, 1.3, 1.57]),
        row(placed, False, [-2.35, 1.3, 1.57]),
        row(placed, False, [-2.35, 1.3, 1.57], opened=True, retreat=True),
    ]


def test_valid_pick_carry_release_and_retreat():
    config, trace = valid_trace()
    report = monitor_delivery(trace, config)
    assert report["satisfied"], report["violations"]
    assert report["events"] == [{"index": 1, "action": "pick"}, {"index": 4, "action": "release"}]


@pytest.mark.parametrize("case,rule", [
    ("remote_pick", "pick.near_pickup"),
    ("dropped_cup", "release.pre.bin_support"),
    ("attachment_drift", "attachment.position"),
    ("outside_release", "release.bin_footprint"),
    ("repeat_pick", "pick_order"),
    ("unreleased", "incomplete_task"),
    ("no_retreat", "finish.retreat_distance"),
    ("collision", "collision.clearance"),
    ("teleport_before_pick", "unattached_motion.position"),
    ("bad_alignment", "pick.grasp_alignment"),
    ("open_during_carry", "carrying.gripper_closed"),
])
def test_counterexamples_cannot_be_hidden_by_later_success(case, rule):
    config, trace = valid_trace()
    trace = deepcopy(trace)
    if case == "remote_pick":
        trace[1]["base_q"] = [0.25, -0.45, 0]
    elif case == "dropped_cup":
        trace[2]["attached"] = False
    elif case == "attachment_drift":
        trace[2]["cup_pose"][0][3] += 0.02
    elif case == "outside_release":
        trace[4]["cup_pose"][0][3] += 0.4
    elif case == "repeat_pick":
        trace[5]["attached"] = True
    elif case == "unreleased":
        trace = trace[:4]
    elif case == "no_retreat":
        trace = trace[:5]
    elif case == "collision":
        trace[2]["collision_residual"] = -0.001
    elif case == "teleport_before_pick":
        trace[0]["cup_pose"][0][3] += 0.1
    elif case == "bad_alignment":
        trace[1]["gripper_pose"][0][3] += 0.1
    elif case == "open_during_carry":
        trace[2]["gripper_q"] = config["gripper"]["open_positions"]
    # Cached Booleans and phase labels must not overrule the geometry.
    for row in trace:
        row.update(phase="success", near_pickup_table=True, cup_in_dropoff_bin=True)
    report = monitor_delivery(trace, config)
    assert not report["satisfied"]
    assert rule in {v["rule"] for v in report["violations"]}
    assert report["first_violation"] is not None


def test_empty_trace_is_not_success():
    assert not monitor_delivery([], load_scene_config())["satisfied"]


def test_pickup_lane_is_not_a_uniform_near_region():
    config = load_scene_config()
    assert near_pickup_residual([0.25, -0.45, 0], config) < 0
    assert "near_pickup_table" not in config["semantic_product_planning"]["region_predicates"]["room2_pickup_lane"]


def test_bin_footprint_accounts_for_tilt():
    config, trace = valid_trace()
    pose = np.array(trace[4]["cup_pose"])
    pose[0, 3] += 0.11
    assert cup_bin_residuals(pose, config)["bin_footprint"] > 0
    pose[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    assert cup_bin_residuals(pose, config)["bin_footprint"] < 0
