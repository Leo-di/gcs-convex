"""Execute a saved reference using forces, with a free cup throughout.

The base is an ideal fully actuated planar joint, not a wheel/contact model.
No state or object pose is overwritten after initialization.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from pydrake.systems.analysis import Simulator

from .build_scene import build_scene, set_robot_configuration
from .task_semantics import cup_bin_residuals


def execute(config, reference, *, time_step=.001, time_scale=3.0,
            visualize=False, grip=True):
    if time_scale <= 0 or len(reference) < 2:
        raise ValueError("Need a positive time scale and at least two reference frames")
    scene = build_scene(config=config, visualize=visualize, dynamics_time_step=time_step)
    plant = scene.plant
    root, context = scene.root_context, scene.plant_context
    first = reference[0]
    set_robot_configuration(scene, first["base_q"], first["arm_q"], first["gripper_q"])
    # BuildScene sets the initial free cup once; subsequent evolution is physics.
    simulator = Simulator(scene.diagram, root)
    plant.get_actuation_input_port().FixValue(context, np.zeros(plant.num_actuators()))
    actuators = [plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices()]
    base_indices = np.arange(scene.base_joint.velocity_start(), scene.base_joint.velocity_start()+3)
    frame_interval = .05 * time_scale
    duration = 1.0 + (len(reference)-1)*frame_interval + 2.0
    control_dt = .01
    if visualize:
        scene.meshcat.StartRecording()
    trace, unexpected = [], []
    initial_cup_z = float(scene.cup_initial_pose.translation()[2])
    for step in range(int(np.ceil(duration/control_dt))):
        t = min((step+1)*control_dt, duration)
        progress = np.clip((t-1.0)/frame_interval, 0, len(reference)-1)
        index = min(int(progress), len(reference)-2)
        alpha = progress-index
        a, b = reference[index:index+2]
        qd = {key: (1-alpha)*np.asarray(a[key])+alpha*np.asarray(b[key])
              for key in ("base_q", "arm_q", "gripper_q")}
        vd = {key: (np.asarray(b[key])-np.asarray(a[key]))/frame_interval
              if 1 < t < duration-2 else np.zeros_like(qd[key]) for key in qd}
        # Preload only while the reference is closing/holding, interpolate it
        # smoothly during opening. Actual finger positions are physics outputs.
        fraction = np.clip((.055 - abs(qd["gripper_q"][0]))/(.055-.0475), 0, 1) if grip else 0.0
        gd = np.array([-.055, .055])*(1-fraction)+np.array([-.040,.040])*fraction
        plant.get_desired_state_input_port(scene.arm_instance).FixValue(context, np.r_[qd["arm_q"], vd["arm_q"]])
        plant.get_desired_state_input_port(scene.gripper_instance).FixValue(context, np.r_[gd, [0,0]])
        forces = np.zeros(plant.num_velocities())
        # Feedforward goes through the actuators' own effort limits together
        # with PD feedback. The cup receives no applied generalized force.
        gravity = plant.CalcGravityGeneralizedForces(context)
        feedforward = np.zeros(plant.num_actuators())
        for actuator in actuators:
            feedforward[actuator.input_start()] = -gravity[actuator.joint().velocity_start()]
        plant.get_actuation_input_port().FixValue(context, feedforward)
        base_error = qd["base_q"]-plant.GetPositions(context, scene.base_instance)
        base_velocity_error = vd["base_q"]-plant.GetVelocities(context, scene.base_instance)
        forces[base_indices] += np.clip(np.array([5000,5000,3000])*base_error +
                                       np.array([500,500,200])*base_velocity_error,
                                       [-500,-500,-200], [500,500,200])
        plant.get_applied_generalized_force_input_port().FixValue(context, forces)
        simulator.AdvanceTo(t)
        contacts = plant.get_contact_results_output_port().Eval(context)
        finger = {"left_finger": 0.0, "right_finger": 0.0}
        support = {"table": 0.0, "dropoff_bin": 0.0}
        contact_pairs = []
        for ci in range(contacts.num_point_pair_contacts()):
            contact = contacts.point_pair_contact_info(ci)
            bodies = [plant.get_body(i) for i in (contact.bodyA_index(), contact.bodyB_index())]
            magnitude = float(np.linalg.norm(contact.contact_force()))
            contact_pairs.append((bodies, magnitude))
        inspector = scene.scene_graph.model_inspector()
        for ci in range(contacts.num_hydroelastic_contacts()):
            contact = contacts.hydroelastic_contact_info(ci)
            surface = contact.contact_surface()
            bodies = [plant.GetBodyFromFrameId(inspector.GetFrameId(gid)) for gid in (surface.id_M(), surface.id_N())]
            contact_pairs.append((bodies, float(np.linalg.norm(contact.F_Ac_W().translational()))))
        for bodies, magnitude in contact_pairs:
            names = [body.name() for body in bodies]
            if "cup" in names:
                other = names[1] if names[0] == "cup" else names[0]
                if other in finger:
                    finger[other] += magnitude
                elif other in support:
                    support[other] += magnitude
                    allowed = (a["phase"] in ("lower_into_bin", "release_cup", "retreat_arm")
                               if other == "dropoff_bin" else
                               a["phase"] in ("navigate_to_pickup", "arm_to_pregrasp", "approach_cup", "close_gripper", "lift_cup"))
                    if not allowed and magnitude > .1:
                        unexpected.append({"time_s": t, "bodies": names, "force_N": magnitude,
                                           "rule": "support_contact_in_wrong_mode"})
                elif magnitude > .1:
                    unexpected.append({"time_s": t, "bodies": names, "force_N": magnitude})
            elif magnitude > .1:
                unexpected.append({"time_s": t, "bodies": names, "force_N": magnitude})
        if step % 5 == 0 or t == duration:
            cup = plant.EvalBodyPoseInWorld(context, scene.cup_body)
            gripper = plant.EvalBodyPoseInWorld(context, scene.gripper_body)
            relative = gripper.inverse() @ cup
            trace.append({"time_s": t, "reference_index": index, "phase": a["phase"],
                          "base_q": plant.GetPositions(context, scene.base_instance).tolist(),
                          "arm_q": plant.GetPositions(context, scene.arm_instance).tolist(),
                          "gripper_q": plant.GetPositions(context, scene.gripper_instance).tolist(),
                          "cup_pose": cup.GetAsMatrix4().tolist(),
                          "relative_translation_m": relative.translation().tolist(),
                          "relative_rotation_matrix": relative.rotation().matrix().tolist(),
                          "cup_speed_m_s": float(np.linalg.norm(plant.EvalBodySpatialVelocityInWorld(context, scene.cup_body).translational())),
                          "cup_angular_speed_rad_s": float(np.linalg.norm(plant.EvalBodySpatialVelocityInWorld(context, scene.cup_body).rotational())),
                          "finger_contact_force_N": finger, "support_contact_force_N": support,
                          "base_tracking_error_m_rad": (qd["base_q"]-plant.GetPositions(context, scene.base_instance)).tolist(),
                          "arm_tracking_error_rad": (qd["arm_q"]-plant.GetPositions(context, scene.arm_instance)).tolist(),
                          "actuation_N_Nm": plant.get_net_actuation_output_port().Eval(context).tolist(),
                          "bin_residuals": cup_bin_residuals(cup.GetAsMatrix4(), config)})
        if step % 1000 == 0:
            print(f"t={t:.2f}/{duration:.2f}s phase={a['phase']}", flush=True)
    if visualize:
        scene.meshcat.StopRecording()
        scene.meshcat.PublishRecording()
    final = [row for row in trace if row["time_s"] >= duration-1]
    lifted = any(row["cup_pose"][2][3] > initial_cup_z+.10 and min(row["finger_contact_force_N"].values()) > .05 for row in trace)
    carried = any(row["cup_pose"][0][3] < -.1 and min(row["finger_contact_force_N"].values()) > .05 for row in trace)
    settled = all(min(row["bin_residuals"].values()) >= 0 and row["support_contact_force_N"]["dropoff_bin"] > .05
                  and max(row["finger_contact_force_N"].values()) < .05 and row["cup_speed_m_s"] < .01
                  and row["cup_angular_speed_rad_s"] < .05 for row in final)
    held = [row for row in trace if row["phase"] == "whole_body_carry_to_dropoff"]
    workspace = all(np.all(np.asarray(r["base_q"]) >= config["robot"]["lower_limits"])
                    and np.all(np.asarray(r["base_q"]) <= config["robot"]["upper_limits"]) for r in trace)
    bilateral_fraction = float(np.mean([min(row["finger_contact_force_N"].values()) > .05 for row in held])) if held else 0.
    reference_relative = next(np.asarray(r["relative_translation_m"]) for r in held) if held else np.zeros(3)
    max_slip = max((float(np.linalg.norm(np.asarray(r["relative_translation_m"])-reference_relative)) for r in held), default=0.)
    maximum_efforts = np.max(np.abs([r["actuation_N_Nm"] for r in trace]), axis=0)
    limits = np.array([a.effort_limit() for a in actuators])
    efforts_within_limits = bool(np.all(maximum_efforts <= limits + 1e-8))
    report = {"satisfied": bool(lifted and carried and settled and not unexpected and workspace
                               and bilateral_fraction == 1. and efforts_within_limits),
              "lifted_with_bilateral_contact": lifted, "carried_to_room1_with_bilateral_contact": carried,
              "released_and_settled_in_bin": settled, "unexpected_contact_count": len(unexpected),
              "first_unexpected_contact": unexpected[0] if unexpected else None,
              "unexpected_contacts": unexpected,
              "base_workspace_bounds_satisfied": workspace,
              "held_bilateral_contact_fraction": bilateral_fraction,
              "maximum_relative_translation_change_during_carry_m": max_slip,
              "final_max_cup_speed_m_s": max(row["cup_speed_m_s"] for row in final),
              "final_max_cup_angular_speed_rad_s": max(row["cup_angular_speed_rad_s"] for row in final),
              "actuator_effort_limits_N_Nm": limits.tolist(),
              "maximum_actuator_efforts_N_Nm": maximum_efforts.tolist(),
              "actuator_effort_limits_satisfied": efforts_within_limits,
              "acceptance": {"lift_m": .10, "cup_settled_speed_m_s": .01,
                             "cup_settled_angular_speed_rad_s": .05,
                             "minimum_finger_or_bin_contact_force_N": .05,
                             "unexpected_contact_force_threshold_N": .1,
                             "final_settle_window_s": 1.0,
                             "all_carry_observations_require_bilateral_contact": True},
              "maximum_arm_tracking_error_rad": max(max(abs(np.array(r["arm_tracking_error_rad"]))) for r in trace),
              "maximum_base_position_error_m": max(float(np.linalg.norm(r["base_tracking_error_m_rad"][:2])) for r in trace),
              "time_step_s": time_step, "control_period_s": control_dt, "time_scale": time_scale,
              "hydroelastic_support": config.get("execution_dynamics", {}).get("hydroelastic_support", False),
              "duration_s": duration, "grip_enabled": grip, "cup_attachment": False,
              "cup_pose_resets_after_initialization": 0,
              "scope": "Force-driven iiwa and WSG plus ideal fully actuated planar base; free cup, SAP contact; contact checks sampled at controller ticks; no wheel dynamics or hardware validation",
              "trace": trace}
    return report, scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-step", type=float, default=.001)
    parser.add_argument("--time-scale", type=float, default=3.0)
    parser.add_argument("--html", action="store_true")
    parser.add_argument("--no-grip", action="store_true")
    parser.add_argument("--point-only", action="store_true", help="Diagnostic baseline with single-point support contact")
    args = parser.parse_args()
    config = json.loads((args.reference/"scene_config_snapshot.json").read_text())
    config["execution_dynamics"] = {"hydroelastic_support": not args.point_only}
    reference = json.loads((args.reference/"delivery_trace.json").read_text())
    report, scene = execute(config, reference, time_step=args.time_step,
                            time_scale=args.time_scale, visualize=args.html, grip=not args.no_grip)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"result.json").write_text(json.dumps(report, indent=2))
    (args.output_dir/"config.json").write_text(json.dumps(config, indent=2))
    if args.html:
        (args.output_dir/"scene.html").write_text(scene.meshcat.StaticHtml())
    print(json.dumps({k:v for k,v in report.items() if k not in ("trace", "unexpected_contacts")},indent=2))
    raise SystemExit(0 if report["satisfied"] else 1)


if __name__ == "__main__":
    main()
