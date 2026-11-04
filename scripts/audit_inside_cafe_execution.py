"""Check DSL containment against measured robot and free-body object poses."""

import argparse
import json
from pathlib import Path
import sys
import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.tree import BodyIndex

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.build_scene import build_scene, set_robot_configuration
from src.semantic_compiler import compile_predicates
from src.whole_robot_room import shape_support
from src.whole_task_joint_gcs import write


def audit(run, constraints):
    cfg = json.loads((run / "config.json").read_text())
    trace = json.loads((run / "physics.json").read_text())["trace"]
    scene = build_scene(config=cfg, visualize=False)
    c = compile_predicates(
        constraints["dsl"], scene, gripper=scene.gripper_open_positions
    )
    inspector = scene.scene_graph.model_inspector()
    payload = constraints["held_predicates"][0]
    region = c.regions[payload["region"]]
    objects = {
        name: [
            (inspector.GetShape(gid), inspector.GetPoseInFrame(gid))
            for gid in scene.plant.GetCollisionGeometriesForBody(body)
        ]
        for name, body in scene.object_bodies.items()
    }
    worst_robot = worst_object = -np.inf
    bad = []
    rows = []
    for i, row in enumerate(trace):
        set_robot_configuration(scene, row["base_q"], row["arm_q"], row["gripper_q"])
        maximum = -np.inf
        for index, shape, offset, predicate, r in c.items:
            X = (
                scene.plant.EvalBodyPoseInWorld(
                    scene.plant_context, scene.plant.get_body(BodyIndex(index))
                )
                @ offset
            )
            for n, b in zip(r.normals, r.offsets):
                maximum = max(
                    maximum,
                    float(
                        n @ X.translation()
                        + shape_support(shape, X.rotation().matrix().T @ n)
                        + predicate.margin_m
                        - b
                    ),
                )
        cup_max = -np.inf
        for name, geometries in objects.items():
            for shape, offset in geometries:
                X = RigidTransform(np.asarray(row["objects"][name]["pose"])) @ offset
                for n, b in zip(region.normals, region.offsets):
                    cup_max = max(
                        cup_max,
                        float(
                            n @ X.translation()
                            + shape_support(shape, X.rotation().matrix().T @ n)
                            + payload["margin_m"]
                            - b
                        ),
                    )
        worst_robot = max(worst_robot, maximum)
        worst_object = max(worst_object, cup_max)
        if max(maximum, cup_max) > 1e-7 and len(bad) < 12:
            bad.append(
                dict(
                    index=i,
                    time_s=row["time_s"],
                    phase=row["phase"],
                    robot_g_m=maximum,
                    objects_g_m=cup_max,
                )
            )
        rows.append(
            dict(
                time_s=row["time_s"],
                phase=row["phase"],
                robot_g_m=maximum,
                objects_g_m=cup_max,
            )
        )
    report = dict(
        satisfied=not bad,
        samples=len(trace),
        robot_maximum_g_m=worst_robot,
        objects_maximum_g_m=worst_object,
        counterexamples=bad,
        scope="Every saved measured state; actual finger positions and actual free-body poses of all cups; no welded-payload shortcut",
    )
    write(run / "measured_inside_audit.json", report)
    write(run / "measured_inside_trace.json", rows)
    if bad:
        raise RuntimeError("Measured execution violates Inside")
    print(report, flush=True)
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument(
        "--constraints", type=Path, default=ROOT / "configs/inside_cafe.json"
    )
    a = p.parse_args()
    audit(a.run, json.loads(a.constraints.read_text()))
