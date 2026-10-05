"""Shared geometric guards for the kinematic cup-delivery task."""

import numpy as np
from pydrake.math import RollPitchYaw

from .scene_spec import semantic_regions
from .semantic_predicates import base_inside_region_residual


def near_pickup_residual(base_q, config) -> float:
    """Distance margin in metres, shared by guards and recorded predicates."""
    center = np.asarray(config["objects"]["table"]["center"][:2])
    return float(config["scenario"]["near_pickup_distance"] -
                 np.linalg.norm(np.asarray(base_q)[:2] - center))


def cup_bin_residuals(cup_pose, config) -> dict[str, float]:
    """Cylinder footprint and support in an open-top bin, in metres.

    Accounts for orientation and allows protrusion above the rim.
    Geometric support is not a force or stability certificate.
    """
    pose = np.asarray(cup_pose, dtype=float)
    cup = config["objects"]["cup"]
    spec = config["objects"]["dropoff_bin"]
    center, size = np.asarray(spec["center"]), np.asarray(spec["size"])
    axis = pose[:3, 2]
    extent = (float(cup["height"]) / 2 * np.abs(axis) +
              float(cup["radius"]) * np.sqrt(np.maximum(0, 1 - axis**2)))
    horizontal = size[:2] / 2 - float(spec["wall_thickness"]) - extent[:2]
    footprint = float(np.min(horizontal - np.abs(pose[:2, 3] - center[:2])))
    bottom = center[2] - size[2] / 2 + float(spec["wall_thickness"])
    tolerance = config["semantic_monitor"]["support_tolerance_m"]
    support = float(tolerance - abs(pose[2, 3] - extent[2] - bottom))
    return {"bin_footprint": footprint, "bin_support": support}


def pick_guard_residuals(observation, config) -> dict[str, float]:
    """Endpoint conditions; approach-path collision checks are separate."""
    gripper = np.asarray(observation["gripper_pose"])
    cup = np.asarray(observation["cup_pose"])
    grasp_point = gripper @ np.r_[config["gripper"]["grasp_point_in_body"], 1.0]
    target = cup[:3, 3] + np.asarray(config["pick_task"]["grasp_offset"])
    # Existing IK uses per-axis position bounds.
    alignment = float(config["pick_task"]["position_tolerance"] -
                      np.max(np.abs(grasp_point[:3] - target)))
    closed = np.asarray(config["gripper"]["closed_positions"])
    tolerance = config["semantic_monitor"]["gripper_tolerance_m"]
    residuals = {
        "near_pickup": near_pickup_residual(observation["base_q"], config),
        "pickup_room": base_inside_region_residual(
            observation["base_q"], semantic_regions(config)["room2"]),
        "grasp_alignment": alignment,
        "gripper_closed": float(tolerance - np.max(np.abs(np.asarray(observation["gripper_q"]) - closed))),
    }
    if "gripper_rpy" in config["pick_task"]:
        target_rotation = RollPitchYaw(config["pick_task"]["gripper_rpy"]).ToRotationMatrix().matrix()
        error = np.arccos(np.clip((np.trace(target_rotation.T @ gripper[:3, :3]) - 1) / 2, -1, 1))
        residuals["grasp_orientation"] = float(config["pick_task"].get("orientation_tolerance", 0.06) - error)
    return residuals


def place_guard_residuals(observation, config) -> dict[str, float]:
    return {
        **cup_bin_residuals(observation["cup_pose"], config),
        "dropoff_room": base_inside_region_residual(
            observation["base_q"], semantic_regions(config)["room1"]),
    }
