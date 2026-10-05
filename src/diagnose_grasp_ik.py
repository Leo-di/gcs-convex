"""Endpoint-only diagnostic: legacy grasp versus an explicit side-grasp pose.

This does not validate a full path or establish physical grasp feasibility.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .build_scene import build_scene, set_robot_configuration
from .delivery_planner import plan_delivery
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .grasp_planner import IKSolveError, plan_pick
from .pickup_candidates import generate_pickup_candidates
from .scene_spec import load_scene_config
from .task_semantics import pick_guard_residuals, place_guard_residuals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    base = load_scene_config()
    for shift in (0.0, 0.05):
        for model in ("legacy", "oriented_aperture"):
            config = deepcopy(base)
            config["objects"]["cup"]["center"][0] += shift
            if model == "oriented_aperture":
                config["gripper"].update(open_positions=[-0.055, 0.055], closed_positions=[-0.0475, 0.0475])
                config["pick_task"].update(gripper_rpy=[0, 0, 0], orientation_tolerance=0.005,
                                           position_tolerance=0.0002)
            name = f"cup_shift_{shift}_{model}"
            (args.output_dir / f"{name}_config.json").write_text(json.dumps(config, indent=2))
            scene = build_scene(config=config, visualize=False)
            candidates, _ = generate_pickup_candidates(config)
            for candidate in candidates:
                record = {"scene": name, "base_q": candidate["base_q"], "full_path_validated": False}
                try:
                    pick = plan_pick(scene, np.asarray(candidate["base_q"]))
                    set_robot_configuration(scene, pick.base_q, pick.grasp_q, scene.gripper_closed_positions)
                    observation = {"base_q": pick.base_q, "gripper_q": scene.gripper_closed_positions,
                                   "cup_pose": scene.cup_initial_pose.GetAsMatrix4(),
                                   "gripper_pose": scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context).GetAsMatrix4()}
                    record["pick_guard"] = pick_guard_residuals(observation, config)
                    record["grasp_geometry"] = summarize_grasp_audit([
                        {"cup_gripper_audit": measure_cup_gripper(scene)}])
                    delivery = plan_delivery(scene, pick)
                    record["place_guard"] = place_guard_residuals({"base_q": delivery.base_q,
                        "cup_pose": delivery.place_cup_pose.GetAsMatrix4()}, config)
                    record["ik_status"] = "endpoints_solved"
                except IKSolveError as error:
                    record.update(ik_status="failed", failure=error.details)
                rows.append(record)
    (args.output_dir / "summary.json").write_text(json.dumps({
        "scope": "Endpoint-only diagnostics, not path or physical grasp validation", "cases": rows}, indent=2))
    for row in rows:
        print(row["scene"], row["base_q"], row["ik_status"], row.get("failure", {}).get("stage"),
              "grasp_geometry:", row.get("grasp_geometry", {}).get("satisfied"))


if __name__ == "__main__":
    main()
