"""Project the saved full-robot dynamics run into a synchronized plan view.

No planning or simulation is performed. FK reconstructs body origins from the
recorded actual joint positions using the saved scene configuration.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from .build_scene import build_scene, set_robot_configuration


def export_projection(directory: Path) -> dict:
    config = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "result.json").read_text())
    scene = build_scene(config=config, visualize=False)
    plant, context = scene.plant, scene.plant_context
    links = [plant.GetBodyByName(f"iiwa_link_{i}", scene.arm_instance) for i in range(8)]
    fingers = [plant.GetBodyByName(name, scene.gripper_instance) for name in ("left_finger", "right_finger")]
    frames = []
    fk_residuals = []
    for row in result["trace"]:
        set_robot_configuration(scene, row["base_q"], row["arm_q"], row["gripper_q"])
        gripper_pose = plant.EvalBodyPoseInWorld(context, scene.gripper_body).GetAsMatrix4()
        relative_pose = np.linalg.inv(gripper_pose) @ np.asarray(row["cup_pose"])
        residual = max(
            np.max(np.abs(relative_pose[:3, 3] - row["relative_translation_m"])),
            np.max(np.abs(relative_pose[:3, :3] - row["relative_rotation_matrix"])),
        )
        if residual > 1e-8:
            raise ValueError(f"FK does not match recorded gripper/cup pose at {row['time_s']}: {residual}")
        fk_residuals.append(float(residual))
        xyz = lambda body: plant.EvalBodyPoseInWorld(context, body).translation().tolist()
        frames.append({**{key: row[key] for key in (
            "time_s", "phase", "reference_index", "base_q", "arm_q", "gripper_q",
            "cup_pose", "finger_contact_force_N", "support_contact_force_N")},
            "links_xyz": [xyz(link) for link in links],
            "gripper_xyz": xyz(scene.gripper_body),
            "fingers_xyz": [xyz(finger) for finger in fingers]})
    objects = [{"name": wall["name"], "kind": "wall", **wall} for wall in config["world"]["walls"]]
    for name, spec in config["objects"].items():
        if isinstance(spec, dict) and "size" in spec and "center" in spec:
            objects.append({"name": name, "kind": "furniture", **spec})
    for spec in config["objects"].get("distractors", []):
        objects.append({"kind": "distractor", **spec})
    return {"frames": frames, "objects": objects, "floor": config["world"]["floor"],
            "base_size": config["robot"]["size"], "cup": config["objects"]["cup"],
            "semantic_regions": config["semantic_regions"],
            "source": str(directory), "duration_s": result["duration_s"],
            "validation": {"matched_source_frames": len(frames),
                           "maximum_gripper_cup_fk_residual": max(fk_residuals)},
            "source_sha256": {name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                              for name in ("config.json", "result.json", "scene.html")},
            "scope": "Actual dynamics trace; arm origins reconstructed by FK. XY projection is not a collision certificate or a 10D region projection."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("outputs/results/full_dynamics/validated"))
    parser.add_argument("--output", type=Path, default=Path("outputs/results/grasp_fix_web/planar/sync"))
    args = parser.parse_args()
    data = export_projection(args.source)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"projection.json").write_text(json.dumps(data))
    shutil.copy2(args.source/"config.json", args.output/"config.json")
    shutil.copy2(args.source/"result.json", args.output/"execution.json")
    # Preserve the original recorded scene byte-for-byte. Parent controls its
    # animator through the same-origin frame after animation assets are loaded.
    shutil.copy2(args.source/"scene.html", args.output/"scene.html")
    template = Path(__file__).with_name("synced_scene_view.html").read_text()
    (args.output/"index.html").write_text(template.replace("__DATA__", json.dumps(data)))
    print(f"Exported {len(data['frames'])} actual-state samples to {args.output}")


if __name__ == "__main__":
    main()
