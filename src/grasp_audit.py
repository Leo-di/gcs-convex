"""Audit cup/gripper and pickup-support distances independently of filters.

Small fingertip penetration tolerance is only a numerical geometric convention;
passing does not establish force closure or a dynamically stable grasp.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform

from .build_scene import build_scene, set_robot_configuration


def annotated_scene_html(
    static_html: str,
    audit: dict,
    wall_audit: dict | None = None,
) -> str:
    status = "PASS" if audit["satisfied"] else "FAIL"
    message = (f"Kinematic replay only | Unfiltered grasp geometry: {status} | "
               "Full-robot dynamics: NOT validated")
    if wall_audit is not None:
        wall_status = "PASS" if wall_audit["satisfied"] else "FAIL"
        payload_cm = 100.0 * wall_audit["minimum_payload_wall_distance_m"]
        robot_cm = 100.0 * wall_audit["minimum_robot_wall_distance_m"]
        margin_cm = 100.0 * wall_audit["required_clearance_m"]
        message += (
            f" | Payload-wall: {payload_cm:.1f} cm | Robot-wall: "
            f"{robot_cm:.1f} cm | required >= {margin_cm:.1f} cm: {wall_status}"
        )
    banner = ('<div style="position:fixed;bottom:0;left:0;right:0;z-index:99999;'
              'padding:10px;background:#8b2020;color:white;font:14px sans-serif;'
              'pointer-events:none">' + message + '</div>')
    if "<body>" not in static_html:
        raise ValueError("Unexpected Meshcat HTML format")
    return static_html.replace("<body>", "<body>\n" + banner, 1)


def measure_wall_clearance(scene) -> dict:
    """Measure payload/robot clearance to every configured wall geometry."""

    context = scene.scene_graph.GetMyContextFromRoot(scene.root_context)
    query = scene.scene_graph.get_query_output_port().Eval(context)
    inspector = query.inspector()
    wall_names = {
        f"{spec['name']}_collision" for spec in scene.config["world"]["walls"]
    }
    wall_ids = {
        geometry_id
        for geometry_id in inspector.GetAllGeometryIds()
        if inspector.GetName(geometry_id) in wall_names
    }
    cup_ids = set(scene.plant.GetCollisionGeometriesForBody(scene.cup_body))
    robot_ids = {
        geometry_id
        for instance in (
            scene.base_instance,
            scene.arm_instance,
            scene.gripper_instance,
        )
        for body_index in scene.plant.GetBodyIndices(instance)
        for geometry_id in scene.plant.GetCollisionGeometriesForBody(
            scene.plant.get_body(body_index)
        )
    }

    closest = {"payload_to_wall": None, "robot_to_wall": None}
    for pair in query.ComputeSignedDistancePairwiseClosestPoints(10.0):
        ids = {pair.id_A, pair.id_B}
        if not ids.intersection(wall_ids):
            continue
        category = None
        if ids.intersection(cup_ids):
            category = "payload_to_wall"
        elif ids.intersection(robot_ids):
            category = "robot_to_wall"
        if category is None:
            continue
        item = {
            "distance_m": float(pair.distance),
            "geometry_a": inspector.GetName(pair.id_A),
            "geometry_b": inspector.GetName(pair.id_B),
        }
        if closest[category] is None or item["distance_m"] < closest[category]["distance_m"]:
            closest[category] = item
    return closest


def summarize_wall_clearance(
    rows: list[dict],
    required_clearance_m: float,
) -> dict:
    """Summarize the explicit wall-clearance gate over a delivery trace."""

    if required_clearance_m < 0.0:
        raise ValueError("required_clearance_m must be nonnegative")
    payload = [
        {"index": index, "phase": row.get("phase"), **row["wall_clearance"]["payload_to_wall"]}
        for index, row in enumerate(rows)
        if row.get("attached", False)
        and row.get("wall_clearance", {}).get("payload_to_wall") is not None
    ]
    robot = [
        {"index": index, "phase": row.get("phase"), **row["wall_clearance"]["robot_to_wall"]}
        for index, row in enumerate(rows)
        if row.get("wall_clearance", {}).get("robot_to_wall") is not None
    ]
    payload_minimum = min(payload, key=lambda item: item["distance_m"]) if payload else None
    robot_minimum = min(robot, key=lambda item: item["distance_m"]) if robot else None
    satisfied = (
        payload_minimum is not None
        and robot_minimum is not None
        and payload_minimum["distance_m"] >= required_clearance_m
        and robot_minimum["distance_m"] >= required_clearance_m
    )
    return {
        "satisfied": satisfied,
        "required_clearance_m": required_clearance_m,
        "minimum_payload_wall_distance_m": (
            payload_minimum["distance_m"] if payload_minimum is not None else None
        ),
        "minimum_robot_wall_distance_m": (
            robot_minimum["distance_m"] if robot_minimum is not None else None
        ),
        "closest_payload_wall_pair": payload_minimum,
        "closest_robot_wall_pair": robot_minimum,
        "attached_frames_checked": len(payload),
        "robot_frames_checked": len(robot),
        "scope": (
            "Finite-sample Drake signed-distance gate for the attached payload and "
            "all robot collision geometry against every configured wall."
        ),
    }


def measure_cup_gripper(scene) -> dict:
    context = scene.scene_graph.GetMyContextFromRoot(scene.root_context)
    query = scene.scene_graph.get_query_output_port().Eval(context)
    inspector = query.inspector()
    pairs = []
    # Cup/table contact is also filtered for IK while the cup rests on its
    # initial support. Check it explicitly so that lifting cannot tunnel
    # through that table under the support exception.
    bodies = [scene.plant.get_body(i) for i in scene.plant.GetBodyIndices(scene.gripper_instance)]
    bodies.append(scene.plant.GetBodyByName("table"))
    for cup_id in scene.plant.GetCollisionGeometriesForBody(scene.cup_body):
        for body in bodies:
            for gripper_id in scene.plant.GetCollisionGeometriesForBody(body):
                distance = query.ComputeSignedDistancePairClosestPoints(cup_id, gripper_id).distance
                pairs.append({"body": body.name(), "geometry": inspector.GetName(gripper_id),
                              "distance_m": float(distance)})
    if not pairs:
        raise ValueError("Missing cup/gripper collision geometry")
    return {"minimum_distance_m": min(p["distance_m"] for p in pairs), "pairs": pairs}


def summarize_grasp_audit(rows: list[dict], finger_tolerance_m: float = 0.0005) -> dict:
    violations = []
    minimum = None
    contact_frames = 0
    for index, row in enumerate(rows):
        contact_distances = {"left_finger": [], "right_finger": []}
        if not row["cup_gripper_audit"]["pairs"]:
            violations.append({"index": index, "rule": "missing_geometry"})
        for pair in row["cup_gripper_audit"]["pairs"]:
            # Only the small named fingertip spheres are allowed this tolerance.
            tip = any(pair["geometry"].endswith(suffix) for suffix in
                      ("_top", "_bottom", "_left", "_right"))
            tolerance = finger_tolerance_m if tip else 0.0
            distance = pair["distance_m"]
            if tip and pair["body"] in contact_distances:
                contact_distances[pair["body"]].append(distance)
            item = {"index": index, **pair, "allowed_penetration_m": tolerance}
            if minimum is None or distance < minimum["distance_m"]:
                minimum = item
            if not np.isfinite(distance) or distance < -tolerance - 1e-8:
                violations.append(item)
        if row.get("attached", False):
            contact_frames += 1
            for body, distances in contact_distances.items():
                if not distances or not any(np.isfinite(d) and abs(d) <= finger_tolerance_m for d in distances):
                    violations.append({"index": index, "rule": "missing_bilateral_contact",
                                       "body": body, "distances_m": distances})
    return {"satisfied": bool(rows) and not violations, "frames_checked": len(rows),
            "minimum_pair": minimum, "violation_count": len(violations),
            "first_violation": violations[0] if violations else None,
            "finger_tolerance_m": finger_tolerance_m,
            "bilateral_contact_frames_checked": contact_frames,
            "contact_scope": "Both fingers must have a tip within the geometric contact tolerance when attached; no force-closure claim",
            "scope": "Cup versus all gripper and pickup-support proximity geometry, ignoring collision filters",
            "physical_grasp_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    trace = json.loads(args.trace.read_text())
    scene = build_scene(config=config, visualize=False)
    for row in trace:
        set_robot_configuration(scene, np.asarray(row["base_q"]), np.asarray(row["arm_q"]),
                                np.asarray(row["gripper_q"]))
        scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body,
                                   RigidTransform(np.asarray(row["cup_pose"])))
        row["cup_gripper_audit"] = measure_cup_gripper(scene)
    report = summarize_grasp_audit(trace)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["satisfied"] else 1)


if __name__ == "__main__":
    main()
