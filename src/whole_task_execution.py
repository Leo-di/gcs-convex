"""Reconstruct, audit, and execute ONLY the one shared full-task solution."""

import copy, json, time
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform
from .build_scene import build_scene
from .green_room_gcs import append_motion
from .whole_task_joint_gcs import make_modes, write, Q
from .stateful_transport_grounding import audit_action, semantic_plan, execute_reference


def audit_base_margin(scene, frames, margin, X_GC=None):
    from .build_scene import set_robot_configuration

    if margin <= 0:
        return dict(enabled=False, satisfied=True)
    base_ids = set(scene.plant.GetCollisionGeometriesForBody(scene.base_body))
    robot_instances = {scene.base_instance, scene.arm_instance, scene.gripper_instance}
    minimum = margin + 0.001
    bad = []
    samples = 0
    for i, f in enumerate(frames):
        a, b = Q(frames[max(0, i - 1)]), Q(f)
        count = max(
            1,
            int(
                np.ceil(
                    max(
                        np.linalg.norm(b[:2] - a[:2]) / 0.005,
                        np.max(abs(b[2:] - a[2:])) / 0.01,
                    )
                )
            ),
        )
        for t in np.linspace(0, 1, count + 1):
            q = (1 - t) * a + t * b
            set_robot_configuration(scene, q[:3], q[3:], f["gripper_q"])
            for key, matrix in f["world_objects"].items():
                pose = (
                    scene.plant.EvalBodyPoseInWorld(
                        scene.plant_context, scene.gripper_body
                    )
                    @ X_GC
                    if X_GC is not None and key == f["object_id"]
                    else RigidTransform(np.array(matrix))
                )
                scene.plant.SetFreeBodyPose(
                    scene.plant_context, scene.object_bodies[key], pose
                )
            query = scene.scene_graph.get_query_output_port().Eval(
                scene.scene_graph.GetMyContextFromRoot(scene.root_context)
            )
            ins = query.inspector()
            for pair in query.ComputeSignedDistancePairwiseClosestPoints(
                margin + 0.001
            ):
                if pair.id_A not in base_ids and pair.id_B not in base_ids:
                    continue
                other = pair.id_B if pair.id_A in base_ids else pair.id_A
                if (
                    scene.plant.GetBodyFromFrameId(
                        ins.GetFrameId(other)
                    ).model_instance()
                    in robot_instances
                ):
                    continue
                minimum = min(minimum, float(pair.distance))
                if pair.distance < margin - 1e-6 and len(bad) < 5:
                    bad.append(
                        dict(edge=i, alpha=float(t), distance_m=float(pair.distance))
                    )
            samples += 1
    return dict(
        enabled=True,
        satisfied=not bad,
        margin_m=margin,
        samples=samples,
        minimum_raw_clearance_lower_bound_m=minimum,
        counterexamples=bad,
        scope="Fresh SceneGraph base/environment distances on interpolated reference; no execution tracking certificate",
    )


def frames_from_global_solution(config, spec, source_reference, run):
    run = Path(run)
    segments = json.loads((run / "global_segments.json").read_text())
    contacts = json.loads((run / "optimized_contacts.json").read_text())
    modes, _ = make_modes(config, source_reference, spec["actions"])
    frames = []
    mode_audits = []
    boundary_audits = []
    for i, mode in enumerate(modes):
        selected = [r for r in segments if r["mode"] == mode.name]
        if not selected:
            raise RuntimeError("Global path skipped a required task mode")
        points = []
        for seg in selected:
            a, b = np.asarray(seg["a"]), np.asarray(seg["b"])
            if points and np.max(abs(points[-1] - a)) > 1e-6:
                raise RuntimeError("Within-mode global solution discontinuity")
            if not points:
                points.append(a)
            if np.max(abs(points[-1] - b)) > 1e-10:
                points.append(b)
        if len(points) < 2:
            raise RuntimeError("Whole-task motion mode has no motion")
        points = np.asarray(points)
        if i > 0 and np.max(abs(points[0] - contacts[i - 1]["q"])) > 1e-6:
            raise RuntimeError("Entry is not the globally shared contact state")
        if i < 6 and np.max(abs(points[-1] - contacts[i]["q"])) > 1e-6:
            raise RuntimeError("Exit is not the globally shared contact state")
        block = []
        phase = mode.action + (
            ".retreat" if i == 6 else ".arrive" if not mode.held else ".carry"
        )
        append_motion(
            block,
            points[:, :3],
            points[:, 3:],
            np.repeat([mode.grip], len(points), axis=0),
            phase,
        )
        if mode.held:
            poses = [mode.fk.transform(Q(f)) @ mode.X_GC for f in block]
            z = np.array([X.translation()[2] for X in poses])
            source_z = np.array(mode.template["world_objects"][mode.active])[2, 3]
            elevated = np.flatnonzero(z >= source_z + 0.10 - 1e-5)
            if not len(elevated):
                raise RuntimeError(
                    "Joint plan never lifts the cup by the task-required 10cm"
                )
            first, last = int(elevated[0]), int(elevated[-1])
        for j, f in enumerate(block):
            f.update(
                object_id=mode.active,
                mode="retreat"
                if i == 6
                else "arrive"
                if not mode.held
                else ("lift" if j < first else "lower" if j > last else "carry"),
                joint_whole_task=True,
                world_objects=copy.deepcopy(mode.template["world_objects"]),
                world_supports=copy.deepcopy(mode.template["world_supports"]),
                payload_mode="held" if mode.held else "free",
            )
            if mode.held:
                f["world_objects"][mode.active] = poses[j].GetAsMatrix4().tolist()
                f["phase"] = mode.action + "." + f["mode"]
        audit_scene = build_scene(
            config=copy.deepcopy(mode.scene.config), visualize=False
        )
        audit = audit_action(audit_scene, block, mode.X_GC, max_samples=250000)
        tracking = audit_base_margin(
            audit_scene,
            block,
            float(
                config.get("whole_task_planning", {}).get("base_tracking_margin_m", 0.0)
            ),
            mode.X_GC,
        )
        mode_audits.append(
            dict(
                mode=mode.name,
                frames=len(block),
                audit=audit,
                base_tracking_clearance=tracking,
            )
        )
        write(run / "motion_mode_audits.json", mode_audits)
        if not audit["satisfied"]:
            raise RuntimeError(
                "Global path rejected by independent mode audit: " + mode.name
            )
        if not tracking["satisfied"]:
            raise RuntimeError(
                "Global path rejected by independent base tracking-margin audit: "
                + mode.name
            )
        if frames:
            a, b = frames[-1], block[0]
            robot = float(
                np.max(abs(np.r_[Q(a), a["gripper_q"]] - np.r_[Q(b), b["gripper_q"]]))
            )
            objects = max(
                float(
                    np.max(
                        abs(np.asarray(a["world_objects"][k]) - b["world_objects"][k])
                    )
                )
                for k in a["world_objects"]
            )
            boundary_audits.append(
                dict(
                    left=a["phase"],
                    right=b["phase"],
                    robot_error=robot,
                    object_pose_error=objects,
                )
            )
            write(run / "boundary_audit.json", boundary_audits)
            if max(robot, objects) > 1e-6:
                raise RuntimeError(
                    "Shared global mode boundary discontinuity: "
                    + str(boundary_audits[-1])
                )
        frames.extend(block)
        if i < 6:
            contact = contacts[i]
            kind = "close" if i % 2 == 0 else "release"
            aid = contact["name"].rsplit("_", 1)[0]
            original = [f for f in source_reference if f["phase"] == aid + "." + kind]
            if not original:
                raise RuntimeError("Missing prescribed gripper transition")
            held_pose = (
                (mode.fk.transform(np.asarray(contact["q"])) @ mode.X_GC)
                .GetAsMatrix4()
                .tolist()
                if mode.held
                else None
            )
            for source in original:
                f = copy.deepcopy(source)
                f["base_q"] = contact["q"][:3]
                f["arm_q"] = contact["q"][3:]
                f["joint_whole_task"] = True
                if held_pose is not None:
                    f["world_objects"][mode.active] = held_pose
                frames.append(f)
    write(run / "boundary_audit.json", boundary_audits)
    write(run / "reference.json", frames)
    write(run / "config.json", config)
    plan = semantic_plan(config, frames)
    write(run / "semantic_plan.json", plan)
    return frames, plan


def execute_global_task(config, spec, frames, plan, search, run):
    run = Path(run)
    selection = copy.deepcopy(search["selected"])
    for item in selection:
        start = next(f for f in frames if f["phase"] == item["action"] + ".arrive")
        # Keep the world-state gate tied to the newly optimized action boundary.
        from .stateful_transport_grounding import digest

        item["report"]["input_world_sha256"] = digest(
            dict(
                poses=start["world_objects"],
                supports=start["world_supports"],
                robot={k: start[k] for k in ("base_q", "arm_q", "gripper_q")},
            )
        )
    tick = time.monotonic()
    result, physics, _ = execute_reference(
        config, spec, frames, plan, selection, visualize=False
    )
    result.update(
        method="one_global_whole_task_joint_GCS",
        execution_wall_seconds=time.monotonic() - tick,
        optimization_scope="All seven robot motion modes and six contact configurations from one shared task solve. Prescribed grasp/release finger profiles and task order.",
    )
    write(run / "result.json", result)
    write(run / "physics.json", physics)
    return result
