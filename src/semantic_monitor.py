"""Offline finite-trace monitor; does not trust phase names or cached predicates.

Attachment is a recorded simulator relation, not contact-force evidence.
Transitions are inferred from its edges. Geometry is checked on both sides.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .task_semantics import pick_guard_residuals, place_guard_residuals


def monitor_delivery(trace: list[dict], config: dict) -> dict:
    settings = config["semantic_monitor"]
    violations, events, states = [], [], []
    state = "seeking_pickup"
    grasp_transform = None
    released_pose = None
    previous = None
    initial_cup = np.eye(4)
    initial_cup[:3, 3] = config["objects"]["cup"]["center"]

    def reject(index, rule, detail, residual=None):
        violations.append({"index": index, "rule": rule, "detail": detail,
                           "residual": None if residual is None else float(residual)})

    def check(index, residuals, prefix):
        for name, residual in residuals.items():
            if not np.isfinite(residual) or residual < -1e-8:
                reject(index, prefix + name, "Geometric guard violated", residual)

    def pose_check(index, actual, expected, prefix):
        position_error = np.linalg.norm(actual[:3, 3] - expected[:3, 3])
        angle = np.arccos(np.clip((np.trace(actual[:3, :3].T @ expected[:3, :3]) - 1) / 2, -1, 1))
        check(index, {
            "position": settings["pose_tolerance_m"] - position_error,
            "orientation": settings["pose_tolerance_rad"] - angle,
        }, prefix)

    for index, row in enumerate(trace):
        # Raw values are required: old traces cannot silently pass this monitor.
        cup = np.asarray(row["cup_pose"], dtype=float)
        gripper = np.asarray(row["gripper_pose"], dtype=float)
        attached = row["attached"]
        if not isinstance(attached, bool):
            raise ValueError("attached must be a Boolean simulator relation")
        if cup.shape != (4, 4) or gripper.shape != (4, 4) or not (
            np.isfinite(cup).all() and np.isfinite(gripper).all()
        ):
            raise ValueError("Expected finite 4x4 poses")
        check(index, {"clearance": row["collision_residual"]}, "collision.")
        was_attached = previous["attached"] if previous else False
        if index == 0 and attached:
            reject(index, "initial_attachment", "Task must start with an ungrasped cup")

        if attached and not was_attached:
            events.append({"index": index, "action": "pick"})
            if state != "seeking_pickup":
                reject(index, "pick_order", "Repeated or out-of-order attachment")
            check(index, pick_guard_residuals(row, config), "pick.")
            if previous:
                check(index, pick_guard_residuals(previous, config), "pick.pre.")
                pose_check(index, cup, np.asarray(previous["cup_pose"]), "pick.continuity.")
            grasp_transform = np.linalg.inv(gripper) @ cup
            state = "carrying"

        if was_attached and not attached:
            events.append({"index": index, "action": "release"})
            if state != "carrying":
                reject(index, "release_order", "Release outside carrying mode")
            check(index, place_guard_residuals(previous, config), "release.pre.")
            check(index, place_guard_residuals(row, config), "release.")
            pose_check(index, cup, np.asarray(previous["cup_pose"]), "release.continuity.")
            released_pose = cup.copy()
            state = "released"

        if attached and grasp_transform is not None:
            pose_check(index, cup, gripper @ grasp_transform, "attachment.")
            closed = np.asarray(config["gripper"]["closed_positions"])
            check(index, {"gripper_closed": settings["gripper_tolerance_m"] -
                          np.max(np.abs(np.asarray(row["gripper_q"]) - closed))}, "carrying.")
        elif state == "seeking_pickup":
            pose_check(index, cup, initial_cup, "unattached_motion.")
        elif state == "released" and released_pose is not None:
            pose_check(index, cup, released_pose, "released_motion.")
            check(index, place_guard_residuals(row, config), "delivered.")

        states.append(state)
        previous = row

    if not trace or state != "released":
        reject(len(trace) - 1, "incomplete_task", "Trace must include a pick and a release")
    else:
        opened = np.asarray(config["gripper"]["open_positions"])
        check(len(trace) - 1, {
            "gripper_open": settings["gripper_tolerance_m"] - np.max(np.abs(np.asarray(trace[-1]["gripper_q"]) - opened)),
            "retreat_distance": np.linalg.norm(np.asarray(trace[-1]["gripper_pose"])[:3, 3] - released_pose[:3, 3]) - settings["retreat_distance_m"],
        }, "finish.")

    return {"satisfied": not violations, "evidence": "sampled_kinematic_trace",
            "frames_checked": len(trace), "events": events, "states": states,
            "violations": violations, "first_violation": violations[0] if violations else None,
            "limitations": ["No continuous-time or contact-force certificate",
                            "Attachment edges are inferred from the simulator relation; attempted no-op commands are not observed",
                            "Pose observations and collision distances are trusted simulator measurements"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--config", type=Path, required=True, help="Config snapshot from this run")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = monitor_delivery(json.loads(args.trace.read_text()), json.loads(args.config.read_text()))
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(serialized + "\n")
    print(json.dumps({key: report[key] for key in ("satisfied", "frames_checked", "first_violation")}, indent=2))
    raise SystemExit(0 if report["satisfied"] else 1)


if __name__ == "__main__":
    main()
