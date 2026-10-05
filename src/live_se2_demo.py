"""Live Drake/Meshcat demo: SE(2) navigation around a blocking cart.

Keeps a Meshcat server running so the page stays interactive. The scene is
visible immediately; after the SE(2) solve finishes, the outbound animation is
published as a recording and then looped live so the base visibly navigates
around the cart with the tucked arm posture.
"""
import argparse
import time

import numpy as np

from .build_scene import build_scene, set_robot_configuration
from .geometry_outbound_gcs import (
    plan_geometry_outbound_se2,
    select_navigation_arm,
)
from .scene_spec import load_scene_config

LANES = ["room1_central_lane", "room1_upper_lane", "doorway_lane", "room2_pickup_lane"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", nargs=3, type=float, default=[-3.0, 0.0, 0.0])
    parser.add_argument("--goal", nargs=3, type=float, default=[1.2, -0.1, 0.0])
    parser.add_argument("--cart", nargs=3, type=float, default=[-1.25, -0.165, 0.41])
    args = parser.parse_args()

    config = load_scene_config()
    config["objects"]["mobile_cart"]["center"] = list(args.cart)
    scene = build_scene(config=config, visualize=True)
    print("MESHCAT_URL " + scene.meshcat.web_url(), flush=True)

    start = np.asarray(args.start, float)
    goal = np.asarray(args.goal, float)
    arm = select_navigation_arm(scene, [start, goal])
    print("NAVIGATION_ARM " + " ".join(f"{v:.3f}" for v in arm), flush=True)

    print("SOLVING", flush=True)
    plan, meta = plan_geometry_outbound_se2(
        scene, start, goal, allowed_region_names=LANES, arm_q=arm,
    )
    print(
        "SOLVED regions=%d min_dist=%.4f frames=%d"
        % (meta["region_count"], meta["minimum_distance_m"], len(plan.configurations)),
        flush=True,
    )

    qs = plan.configurations
    gripper = scene.gripper_open_positions

    scene.meshcat.StartRecording()
    for frame, q in enumerate(qs):
        set_robot_configuration(scene, q, arm, gripper)
        scene.root_context.SetTime(
            float(frame) / float(max(1, len(qs) - 1)) * plan.trajectory_duration
        )
        scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()
    print("RECORDING_PUBLISHED", flush=True)

    # Keep the server alive and loop the base motion in near real time so the
    # page shows continuous motion without touching the playback controls.
    loop = qs[:: max(1, len(qs) // 220)]
    while True:
        for q in loop:
            set_robot_configuration(scene, q, arm, gripper)
            scene.diagram.ForcedPublish(scene.root_context)
            time.sleep(0.05)
        time.sleep(1.5)


if __name__ == "__main__":
    main()
