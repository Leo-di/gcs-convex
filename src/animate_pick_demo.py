"""Animate base motion followed by an IK-based cup pick and lift."""

from __future__ import annotations

import json

import numpy as np

from .animate_demo import interpolate_waypoints
from .build_scene import build_scene, set_robot_configuration
from .collision_checker import evaluate_collision_status
from .grasp_planner import plan_pick
from .scene_spec import PROJECT_ROOT


def _interpolate(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    return np.asarray(
        [(1.0 - alpha) * start + alpha * end for alpha in np.linspace(0.0, 1.0, count)],
        dtype=float,
    )


def main() -> None:
    scene = build_scene()
    base_waypoints = np.asarray(scene.config["reference_path"], dtype=float)
    base_path = interpolate_waypoints(base_waypoints, samples_per_segment=18)
    pick_plan = plan_pick(scene, base_path[-1])

    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray, object]] = []

    for base_q in base_path:
        frames.append(
            (
                base_q,
                scene.arm_home_positions,
                scene.gripper_open_positions,
                scene.cup_initial_pose,
            )
        )

    for arm_q in _interpolate(scene.arm_home_positions, pick_plan.pregrasp_q, 45):
        frames.append(
            (
                pick_plan.base_q,
                arm_q,
                scene.gripper_open_positions,
                scene.cup_initial_pose,
            )
        )

    for arm_q in _interpolate(pick_plan.pregrasp_q, pick_plan.grasp_q, 35):
        frames.append(
            (
                pick_plan.base_q,
                arm_q,
                scene.gripper_open_positions,
                scene.cup_initial_pose,
            )
        )

    for gripper_q in _interpolate(
        scene.gripper_open_positions,
        scene.gripper_closed_positions,
        25,
    ):
        frames.append(
            (
                pick_plan.base_q,
                pick_plan.grasp_q,
                gripper_q,
                scene.cup_initial_pose,
            )
        )

    set_robot_configuration(
        scene,
        pick_plan.base_q,
        pick_plan.grasp_q,
        scene.gripper_closed_positions,
    )
    X_WG_grasp = scene.gripper_body.body_frame().CalcPoseInWorld(
        scene.plant_context
    )
    X_GC = X_WG_grasp.inverse() @ scene.cup_initial_pose

    for arm_q in _interpolate(pick_plan.grasp_q, pick_plan.lift_q, 45):
        set_robot_configuration(
            scene,
            pick_plan.base_q,
            arm_q,
            scene.gripper_closed_positions,
        )
        X_WG = scene.gripper_body.body_frame().CalcPoseInWorld(
            scene.plant_context
        )
        frames.append(
            (
                pick_plan.base_q,
                arm_q,
                scene.gripper_closed_positions,
                X_WG @ X_GC,
            )
        )

    frames.extend([frames[-1]] * 30)

    minimum_distance = float("inf")
    collision_free = True
    scene.meshcat.StartRecording()
    for index, (base_q, arm_q, gripper_q, cup_pose) in enumerate(frames):
        set_robot_configuration(scene, base_q, arm_q, gripper_q)
        scene.plant.SetFreeBodyPose(
            scene.plant_context,
            scene.cup_body,
            cup_pose,
        )
        status = evaluate_collision_status(
            scene,
            base_q,
            arm_q=arm_q,
            gripper_q=gripper_q,
        )
        minimum_distance = min(minimum_distance, status.minimum_signed_distance)
        collision_free = collision_free and status.collision_free
        scene.root_context.SetTime(index * 0.05)
        scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()

    final_cup_pose = scene.plant.GetFreeBodyPose(
        scene.plant_context,
        scene.cup_body,
    )
    result = {
        "frames": len(frames),
        "collision_free": collision_free,
        "minimum_signed_distance": minimum_distance,
        "base_goal": pick_plan.base_q.tolist(),
        "pregrasp_q": pick_plan.pregrasp_q.tolist(),
        "grasp_q": pick_plan.grasp_q.tolist(),
        "lift_q": pick_plan.lift_q.tolist(),
        "initial_cup_position": scene.cup_initial_pose.translation().tolist(),
        "final_cup_position": final_cup_pose.translation().tolist(),
    }
    output_path = PROJECT_ROOT / "outputs" / "results" / "pick_result.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"Meshcat URL: {scene.meshcat.web_url()}")
    print(f"Saved pick result: {output_path}")
    print("Collision free:", collision_free)
    print("Minimum signed distance:", minimum_distance)
    print("Final cup position:", final_cup_pose.translation())
    input("Press Enter to close the pick animation... ")


if __name__ == "__main__":
    main()
