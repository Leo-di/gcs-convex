"""Inverse-kinematics helpers for the cup pick demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydrake.math import RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from .build_scene import SceneHandles, set_robot_configuration


@dataclass(frozen=True)
class PickPlan:
    base_q: np.ndarray
    pregrasp_q: np.ndarray
    grasp_q: np.ndarray
    lift_q: np.ndarray
    grasp_target: np.ndarray


def _arm_position_indices(scene: SceneHandles) -> list[int]:
    indices: list[int] = []
    for joint_index in scene.plant.GetJointIndices(scene.arm_instance):
        joint = scene.plant.get_joint(joint_index)
        if joint.num_positions() == 1:
            indices.append(joint.position_start())
    if len(indices) != 7:
        raise RuntimeError(f"Expected 7 arm positions, got {len(indices)}.")
    return indices


def solve_arm_ik(
    scene: SceneHandles,
    *,
    base_q: np.ndarray,
    target_position: np.ndarray,
    initial_arm_q: np.ndarray,
    position_tolerance: float,
    minimum_clearance: float,
    target_orientation: RotationMatrix | None = None,
    orientation_tolerance: float = 0.06,
) -> np.ndarray:
    """Solve arm IK while the base, gripper, and cup remain fixed."""

    set_robot_configuration(
        scene,
        base_q,
        initial_arm_q,
        scene.gripper_open_positions,
    )
    scene.plant.SetFreeBodyPose(
        scene.plant_context,
        scene.cup_body,
        scene.cup_initial_pose,
    )

    q0 = scene.plant.GetPositions(scene.plant_context).copy()
    arm_indices = _arm_position_indices(scene)
    fixed_indices = np.asarray(
        [
            index
            for index in range(scene.plant.num_positions())
            if index not in arm_indices
        ],
        dtype=int,
    )

    ik = InverseKinematics(scene.plant, scene.plant_context, True)
    q = ik.q()
    program = ik.prog()
    program.AddBoundingBoxConstraint(
        q0[fixed_indices],
        q0[fixed_indices],
        q[fixed_indices],
    )

    grasp_point = np.asarray(
        scene.config["gripper"]["grasp_point_in_body"],
        dtype=float,
    )
    target_position = np.asarray(target_position, dtype=float)
    ik.AddPositionConstraint(
        scene.gripper_body.body_frame(),
        grasp_point,
        scene.plant.world_frame(),
        target_position - position_tolerance,
        target_position + position_tolerance,
    )
    if target_orientation is not None:
        ik.AddOrientationConstraint(
            scene.plant.world_frame(),
            RotationMatrix(),
            scene.gripper_body.body_frame(),
            target_orientation,
            orientation_tolerance,
        )
    ik.AddMinimumDistanceLowerBoundConstraint(
        minimum_clearance,
        influence_distance_offset=0.02,
    )

    program.AddQuadraticErrorCost(
        np.eye(scene.plant.num_positions()),
        q0,
        q,
    )
    program.SetInitialGuess(q, q0)
    result = Solve(program)
    if not result.is_success():
        raise RuntimeError(
            f"IK failed for target {target_position.tolist()} "
            f"with solver {result.get_solver_id().name()}."
        )

    solution = result.GetSolution(q)
    return solution[arm_indices]


def plan_pick(scene: SceneHandles, base_q: np.ndarray) -> PickPlan:
    task = scene.config["pick_task"]
    cup_center = np.asarray(
        scene.config["objects"]["cup"]["center"],
        dtype=float,
    )
    tolerance = float(task["position_tolerance"])
    clearance = float(task["minimum_clearance"])

    pregrasp_target = cup_center + np.asarray(task["pregrasp_offset"], dtype=float)
    grasp_target = cup_center + np.asarray(task["grasp_offset"], dtype=float)
    lift_target = cup_center + np.asarray(task["lift_offset"], dtype=float)

    pregrasp_q = solve_arm_ik(
        scene,
        base_q=base_q,
        target_position=pregrasp_target,
        initial_arm_q=scene.arm_home_positions,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
    )
    grasp_q = solve_arm_ik(
        scene,
        base_q=base_q,
        target_position=grasp_target,
        initial_arm_q=pregrasp_q,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
    )
    lift_q = solve_arm_ik(
        scene,
        base_q=base_q,
        target_position=lift_target,
        initial_arm_q=grasp_q,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
    )

    return PickPlan(
        base_q=np.asarray(base_q, dtype=float),
        pregrasp_q=pregrasp_q,
        grasp_q=grasp_q,
        lift_q=lift_q,
        grasp_target=grasp_target,
    )
