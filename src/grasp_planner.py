"""Inverse-kinematics helpers for the cup pick demonstration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pydrake.math import RotationMatrix, RollPitchYaw
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


class IKSolveError(RuntimeError):
    def __init__(self, details: dict):
        self.details = details
        super().__init__(f"IK failed at {details['stage']} for target {details['target_position']} "
                         f"with solver {details['solver']}: {details['solution_result']}")


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
    stage: str = "ik",
    maximum_joint_step: float | None = None,
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
    if "ik_solver" in scene.config:
        limits = scene.config["whole_body_planning"]
        program.AddBoundingBoxConstraint(np.asarray(limits["joint_lower_limits"]),
                                         np.asarray(limits["joint_upper_limits"]),
                                         q[arm_indices])
    if maximum_joint_step is not None:
        if not np.isfinite(maximum_joint_step) or maximum_joint_step <= 0:
            raise ValueError('maximum_joint_step must be positive and finite')
        program.AddBoundingBoxConstraint(np.asarray(initial_arm_q)-maximum_joint_step,
                                         np.asarray(initial_arm_q)+maximum_joint_step, q[arm_indices])

    grasp_point = np.asarray(
        scene.config["gripper"]["grasp_point_in_body"],
        dtype=float,
    )
    target_position = np.asarray(target_position, dtype=float)
    position_margin = min(1e-6, position_tolerance * 0.01)
    ik.AddPositionConstraint(
        scene.gripper_body.body_frame(),
        grasp_point,
        scene.plant.world_frame(),
        target_position - (position_tolerance - position_margin),
        target_position + (position_tolerance - position_margin),
    )
    if target_orientation is not None:
        # Keep the IK solution inside the semantic guard, rather than relying
        # on identical floating-point feasibility at its angular boundary.
        angular_margin = min(1e-5, orientation_tolerance * 0.01)
        ik.AddOrientationConstraint(
            scene.plant.world_frame(),
            target_orientation,
            scene.gripper_body.body_frame(),
            RotationMatrix(),
            orientation_tolerance - angular_margin,
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
    # Reuse exactly the same constraints and objective across deterministic
    # initial guesses. A failed local solve is not an infeasibility proof.
    retry_count = int(scene.config.get("ik_solver", {}).get("max_restarts", 0))
    rng = np.random.default_rng(1701)
    lower = scene.plant.GetPositionLowerLimits()[arm_indices]
    upper = scene.plant.GetPositionUpperLimits()[arm_indices]
    if maximum_joint_step is not None:
        lower = np.maximum(lower, np.asarray(initial_arm_q)-maximum_joint_step)
        upper = np.minimum(upper, np.asarray(initial_arm_q)+maximum_joint_step)
    for _ in range(retry_count):
        if result.is_success():
            break
        seed = q0.copy()
        seed[arm_indices] = rng.uniform(lower, upper)
        program.SetInitialGuess(q, seed)
        result = Solve(program)
    if not result.is_success():
        raise IKSolveError({
            "stage": stage, "target_position": target_position.tolist(),
            "base_q": np.asarray(base_q).tolist(),
            "solver": result.get_solver_id().name(),
            "solution_result": str(result.get_solution_result()),
            "position_tolerance_m": position_tolerance,
            "orientation_constrained": target_orientation is not None,
            "orientation_tolerance_rad": orientation_tolerance if target_orientation is not None else None,
        })

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
    orientation = (RollPitchYaw(task["gripper_rpy"]).ToRotationMatrix()
                   if "gripper_rpy" in task else None)
    orientation_tolerance = float(task.get("orientation_tolerance", 0.06))

    pregrasp_target = cup_center + np.asarray(task["pregrasp_offset"], dtype=float)
    grasp_target = cup_center + np.asarray(task["grasp_offset"], dtype=float)
    lift_target = cup_center + np.asarray(task["lift_offset"], dtype=float)

    pregrasp_q = solve_arm_ik(
        scene,
        stage="pregrasp",
        base_q=base_q,
        target_position=pregrasp_target,
        initial_arm_q=scene.arm_home_positions,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
        target_orientation=orientation,
        orientation_tolerance=orientation_tolerance,
    )
    grasp_q = solve_arm_ik(
        scene,
        stage="grasp",
        base_q=base_q,
        target_position=grasp_target,
        initial_arm_q=pregrasp_q,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
        target_orientation=orientation,
        orientation_tolerance=orientation_tolerance,
    )
    lift_q = solve_arm_ik(
        scene,
        stage="lift",
        base_q=base_q,
        target_position=lift_target,
        initial_arm_q=grasp_q,
        position_tolerance=tolerance,
        minimum_clearance=clearance,
        target_orientation=orientation,
        orientation_tolerance=orientation_tolerance,
    )

    return PickPlan(
        base_q=np.asarray(base_q, dtype=float),
        pregrasp_q=pregrasp_q,
        grasp_q=grasp_q,
        lift_q=lift_q,
        grasp_target=grasp_target,
    )


def cartesian_arm_path(scene: SceneHandles, base_q: np.ndarray,
                       start_q: np.ndarray, end_q: np.ndarray, count: int,
                       *, stage: str, maximum_joint_step: float = .12,
                       edge_validator=None, maximum_subdivisions: int = 12) -> np.ndarray:
    """Track a straight gripper translation and shortest rotation with IK.

    The start configuration and target end pose are preserved. The endpoint
    is solved on the continuation branch, rather than jumping to an unrelated
    IK solution with the same pose. Each configuration satisfies the same
    collision constraints as endpoint IK; the assembled path still requires
    the independent, unfiltered cup/gripper audit.
    """
    poses = []
    for arm_q in (start_q, end_q):
        set_robot_configuration(scene, base_q, arm_q, scene.gripper_open_positions)
        poses.append(scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context))
    point = np.asarray(scene.config["gripper"]["grasp_point_in_body"])
    positions = [pose.multiply(point) for pose in poses]
    rotations = [pose.rotation().ToQuaternion() for pose in poses]
    if count < 2 or maximum_subdivisions < 0:
        raise ValueError('Need at least two samples and a nonnegative subdivision limit')
    path = [np.asarray(start_q)]
    def advance(left, alpha, depth):
        try:
            candidate = solve_arm_ik(
            scene, base_q=base_q,
            target_position=(1-alpha)*positions[0] + alpha*positions[1],
            target_orientation=RotationMatrix(rotations[0].slerp(alpha, rotations[1])),
            initial_arm_q=path[-1], position_tolerance=0.0002,
            orientation_tolerance=0.005,
            minimum_clearance=float(scene.config["pick_task"]["minimum_clearance"]),
            stage=f"{stage}_{alpha:.8f}", maximum_joint_step=maximum_joint_step,
            )
            valid = np.max(np.abs(candidate-path[-1])) <= maximum_joint_step+1e-7
            if valid and edge_validator is not None:
                valid = edge_validator(path[-1], candidate)
            if valid:
                path.append(candidate)
                return
        except IKSolveError:
            pass
        if depth >= maximum_subdivisions:
            raise RuntimeError(f'{stage}: continuous IK refinement exhausted at [{left}, {alpha}]; no branch-jump fallback')
        midpoint = (left+alpha)/2
        advance(left, midpoint, depth+1)
        advance(midpoint, alpha, depth+1)
    for left, alpha in zip(np.linspace(0,1,count)[:-1], np.linspace(0,1,count)[1:]):
        advance(float(left), float(alpha), 0)
    return np.asarray(path)
