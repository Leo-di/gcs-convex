"""Dense sampled audit of the piecewise-linear joint replay (not a proof)."""
import argparse
import json
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix

from .build_scene import build_scene, set_robot_configuration
from .collision_checker import evaluate_collision_status
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit


def audit_replay(trace: list[dict], config: dict, subdivisions: int = 10) -> dict:
    if len(trace) < 2 or subdivisions < 1:
        raise ValueError("Need at least two frames and positive subdivisions")
    scene = build_scene(config=config, visualize=False)
    minimum = float("inf")
    violations = []
    count = 0
    for index in range(len(trace) - 1):
        a, b = trace[index:index + 2]
        X_CA, X_CB = (RigidTransform(np.asarray(r["cup_pose"])) for r in (a, b))
        X_GC = RigidTransform(np.asarray(a["gripper_pose"])).inverse() @ X_CA
        for alpha in np.linspace(0, 1, subdivisions + 1):
            if index and alpha == 0:
                continue
            q = {key: (1-alpha)*np.asarray(a[key]) + alpha*np.asarray(b[key])
                 for key in ("base_q", "arm_q", "gripper_q")}
            set_robot_configuration(scene, **q)
            attached = bool(a["attached"] and b["attached"])
            if attached:
                cup = scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context) @ X_GC
            else:
                rotation = RotationMatrix(X_CA.rotation().ToQuaternion().slerp(alpha, X_CB.rotation().ToQuaternion()))
                cup = RigidTransform(rotation, (1-alpha)*X_CA.translation() + alpha*X_CB.translation())
            scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, cup)
            status = evaluate_collision_status(scene, q["base_q"], arm_q=q["arm_q"], gripper_q=q["gripper_q"])
            audit = summarize_grasp_audit([{"attached": attached, "cup_gripper_audit": measure_cup_gripper(scene)}])
            minimum = min(minimum, status.minimum_signed_distance)
            if not status.collision_free or not audit["satisfied"]:
                violations.append({"edge": index, "alpha": float(alpha),
                    "environment_clearance_m": status.minimum_signed_distance,
                    "grasp_violation": audit["first_violation"]})
            count += 1
    return {"satisfied": not violations, "samples": count, "subdivisions": subdivisions,
            "minimum_environment_clearance_m": minimum, "violation_count": len(violations),
            "first_violation": violations[0] if violations else None,
            "scope": "Dense samples of linear joint interpolation, attached cup recomputed from FK; not continuous-time safety or dynamics"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--subdivisions", type=int, default=10)
    args = parser.parse_args()
    config = json.loads((args.directory / "scene_config_snapshot.json").read_text())
    trace = json.loads((args.directory / "delivery_trace.json").read_text())
    report = audit_replay(trace, config, args.subdivisions)
    (args.directory / "dense_replay_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
    raise SystemExit(0 if report["satisfied"] else 1)


if __name__ == "__main__":
    main()
