"""Kinematic planning helpers for the cup transfer and placement phases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydrake.math import RigidTransform, RotationMatrix

from .build_scene import SceneHandles, set_robot_configuration
from .grasp_planner import PickPlan, solve_arm_ik


@dataclass(frozen=True)
class DeliveryPlan:
    base_q: np.ndarray
    carry_q: np.ndarray
    preplace_q: np.ndarray
    place_q: np.ndarray
    X_GC: RigidTransform
    preplace_cup_pose: RigidTransform
    place_cup_pose: RigidTransform


def _grasp_point_target(
    scene: SceneHandles,
    X_WG: RigidTransform,
) -> np.ndarray:
    point_G = np.asarray(
        scene.config["gripper"]["grasp_point_in_body"],
        dtype=float,
    )
    return np.asarray(X_WG.multiply(point_G), dtype=float)


def plan_delivery(scene: SceneHandles, pick_plan: PickPlan) -> DeliveryPlan:
    """Solve upright pre-place and place configurations at the green bin."""

    task = scene.config["delivery_task"]
    base_q = np.asarray(task["transfer_path"][-1], dtype=float)
    place_center = np.asarray(task["place_cup_center"], dtype=float)
    preplace_center = place_center + np.array(
        [0.0, 0.0, float(task["preplace_height"])],
    )

    set_robot_configuration(
        scene,
        pick_plan.base_q,
        pick_plan.grasp_q,
        scene.gripper_closed_positions,
    )
    scene.plant.SetFreeBodyPose(
        scene.plant_context,
        scene.cup_body,
        scene.cup_initial_pose,
    )
    X_WG_grasp = scene.gripper_body.body_frame().CalcPoseInWorld(
        scene.plant_context,
    )
    X_GC = X_WG_grasp.inverse() @ scene.cup_initial_pose

    place_rotation = RotationMatrix.MakeZRotation(float(task.get("place_yaw", 0.0)))
    preplace_cup_pose = RigidTransform(place_rotation, preplace_center)
    place_cup_pose = RigidTransform(place_rotation, place_center)
    X_WG_preplace = preplace_cup_pose @ X_GC.inverse()
    X_WG_place = place_cup_pose @ X_GC.inverse()

    carry_center = np.array(
        [
            pick_plan.base_q[0],
            pick_plan.base_q[1],
            float(task["carry_cup_height"]),
        ]
    )
    carry_cup_pose = RigidTransform(RotationMatrix(), carry_center)
    X_WG_carry = carry_cup_pose @ X_GC.inverse()

    tolerance = float(task["position_tolerance"])
    orientation_tolerance = float(task["orientation_tolerance"])
    clearance = float(task["minimum_clearance"])

    carry_q = solve_arm_ik(
        scene,
        stage="carry",
        base_q=pick_plan.base_q,
        target_position=_grasp_point_target(scene, X_WG_carry),
        initial_arm_q=pick_plan.lift_q,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
        target_orientation=X_WG_carry.rotation(),
        orientation_tolerance=orientation_tolerance,
    )
    preplace_q = solve_arm_ik(
        scene,
        stage="preplace",
        base_q=base_q,
        target_position=_grasp_point_target(scene, X_WG_preplace),
        initial_arm_q=pick_plan.lift_q,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
        target_orientation=X_WG_preplace.rotation(),
        orientation_tolerance=orientation_tolerance,
    )
    place_q = solve_arm_ik(
        scene,
        stage="place",
        base_q=base_q,
        target_position=_grasp_point_target(scene, X_WG_place),
        initial_arm_q=preplace_q,
        position_tolerance=float(task["place_position_tolerance"]),
        minimum_clearance=clearance,
        target_orientation=X_WG_place.rotation(),
        orientation_tolerance=float(task["place_orientation_tolerance"]),
    )

    # Releasing must preserve the achieved pose, not teleport to the IK target.
    set_robot_configuration(scene, base_q, place_q, scene.gripper_closed_positions)
    place_cup_pose = (
        scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context) @ X_GC
    )

    return DeliveryPlan(
        base_q=base_q,
        carry_q=carry_q,
        preplace_q=preplace_q,
        place_q=place_q,
        X_GC=X_GC,
        preplace_cup_pose=preplace_cup_pose,
        place_cup_pose=place_cup_pose,
    )
