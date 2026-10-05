"""Live replay of a complete validated delivery trace in Drake/Meshcat.

Reads a ``delivery_trace.json`` produced by ``animate_delivery_demo``
(every frame's base, arm, gripper, and cup pose) and loops the whole
task—navigate, pre-grasp, approach, grasp, lift, carry through the
doorway, lower into the bin, release, retreat—in a live Meshcat session
so the page stays interactive. This is a replay of a validated
trajectory, not a re-solve; the solver-facing demos live elsewhere.

Usage:
LCM_DEFAULT_URL=memq:// .venv/bin/python -m src.live_delivery_replay \
    --trace outputs/results/delivery_trace.json [--port 7001]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pydrake.math import RigidTransform

from .build_scene import build_scene, set_robot_configuration
from .scene_spec import PROJECT_ROOT, load_scene_config

PHASE_ORDER = [
    "navigate_to_pickup",
    "arm_to_pregrasp",
    "approach_cup",
    "close_gripper",
    "lift_cup",
    "whole_body_carry_to_dropoff",
    "lower_into_bin",
    "release_cup",
    "retreat_arm",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trace",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "results" / "delivery_trace.json",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--loop-seconds", type=float, default=0.04)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    if not trace:
        raise SystemExit(f"Empty trace: {args.trace}")
    config = load_scene_config(args.config) if args.config else load_scene_config()
    # The trace was produced with a specific cart placement; the default
    # scene configuration matches the nominal trace.
    scene = build_scene(config=config, visualize=True)
    print("MESHCAT_URL " + scene.meshcat.web_url(), flush=True)

    phases = [row["phase"] for row in trace]
    print(
        "PHASES " + " -> ".join(
            f"{name}({phases.count(name)})" for name in PHASE_ORDER if name in phases
        ),
        flush=True,
    )
    print(f"FRAMES {len(trace)}", flush=True)

    def publish(row, timestamp: float) -> None:
        set_robot_configuration(
            scene,
            np.asarray(row["base_q"], dtype=float),
            np.asarray(row["arm_q"], dtype=float),
            np.asarray(row["gripper_q"], dtype=float),
        )
        scene.plant.SetFreeBodyPose(
            scene.plant_context,
            scene.cup_body,
            RigidTransform(np.asarray(row["cup_pose"], dtype=float)),
        )
        scene.root_context.SetTime(timestamp)
        scene.diagram.ForcedPublish(scene.root_context)

    # Publish one full recording first so the scrub bar covers the task.
    scene.meshcat.StartRecording()
    for index, row in enumerate(trace):
        publish(row, index * 0.05)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()
    print("RECORDING_PUBLISHED", flush=True)

    # Then loop the motion live so the page shows continuous motion.
    import time

    step = max(1, len(trace) // 240)
    while True:
        for index in range(0, len(trace), step):
            publish(trace[index], index * 0.05)
            time.sleep(args.loop_seconds)
        time.sleep(1.5)


if __name__ == "__main__":
    main()
