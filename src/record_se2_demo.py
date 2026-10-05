"""Record a Meshcat 3D animation of the semantic GCS base navigation.

Builds the two-room scene, plans the empty-hand base route with Drake GCS, and
records the base motion (with yaw) into a self-contained 3D HTML page.

By default it uses the fast box-lane `plan_base_gcs` so a 3D recording is
produced in seconds. Pass --se2 to use the current-geometry SE(2) disk cover
(this is slower and needs a clearable scene).
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from .build_scene import build_scene, set_robot_configuration
from .gcs_base_planner import plan_base_gcs
from .geometry_outbound_gcs import plan_geometry_outbound_se2
from .scene_spec import load_scene_config

LANES = ["room1_central_lane", "room1_upper_lane", "doorway_lane", "room2_pickup_lane"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("outputs/results/grasp_fix_web/planar/se2_demo"))
    parser.add_argument("--start", nargs=3, type=float, default=[-3.0, 0.0, 1.2])
    parser.add_argument("--goal", nargs=3, type=float, default=[1.2, -0.1, 0.0])
    parser.add_argument("--se2", action="store_true")
    parser.add_argument("--cart", nargs=3, type=float, default=None)
    args = parser.parse_args()

    config = load_scene_config()
    if args.cart is not None:
        config["objects"]["mobile_cart"]["center"] = list(args.cart)

    scene = build_scene(config=config, visualize=True)
    start = np.asarray(args.start, float)
    goal = np.asarray(args.goal, float)

    began = time.perf_counter()
    if args.se2:
        plan, meta = plan_geometry_outbound_se2(scene, start, goal, allowed_region_names=LANES)
        algorithm = "SE(2)"
    else:
        plan = plan_base_gcs(config, start, goal, allowed_region_names=LANES)
        algorithm = "box-lane GCS (SE(2))"
        meta = {"minimum_distance_m": None}
    solve_seconds = time.perf_counter() - began

    qs = plan.configurations
    scene.meshcat.StartRecording()
    arm = scene.arm_home_positions
    gripper = scene.gripper_open_positions
    for frame, q in enumerate(qs):
        set_robot_configuration(scene, q, arm, gripper)
        scene.root_context.SetTime(float(frame) / float(max(1, len(qs) - 1)) * plan.trajectory_duration)
        scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "index.html").write_text(scene.meshcat.StaticHtml())
    (args.output / "result.json").write_text(json.dumps({
        "algorithm": algorithm,
        "start": start.tolist(),
        "goal": goal.tolist(),
        "cart_center": config["objects"]["mobile_cart"]["center"],
        "solve_seconds": solve_seconds,
        "frames": len(qs),
        "yaw_range": [float(qs[:, 2].min()), float(qs[:, 2].max())],
        "path_length": float(np.linalg.norm(np.diff(qs[:, :2], axis=0), axis=1).sum()),
        "min_distance": meta.get("minimum_distance_m"),
    }, indent=2, default=str))
    print(json.dumps({
        "output": str(args.output),
        "algorithm": algorithm,
        "solve_seconds": solve_seconds,
        "frames": len(qs),
        "yaw_range": [float(qs[:, 2].min()), float(qs[:, 2].max())],
        "min_distance": meta.get("minimum_distance_m"),
    }, indent=2))


if __name__ == "__main__":
    main()
