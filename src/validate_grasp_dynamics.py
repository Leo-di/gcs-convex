"""Local free-object grasp experiment, independent of the kinematic attachment.

The real WSG model is mounted on an actuated vertical test stage. A free 0.2 kg
cup starts on a support, fingers close under bounded PD actuation, then the
stage lifts 12 cm and holds. This tests local contact dynamics, not tracking
of the mobile manipulator's full trajectory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pydrake.geometry import Box, Cylinder, Meshcat, MeshcatVisualizer
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import (
    AddMultibodyPlantSceneGraph, ContactModel, CoulombFriction,
    DiscreteContactApproximation,
)
from pydrake.multibody.tree import (
    SpatialInertia, UnitInertia, PrismaticJoint, PdControllerGains,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder

from .build_scene import build_scene, set_robot_configuration
from .grasp_planner import plan_pick
from .scene_spec import load_scene_config


def run_experiment(config: dict, *, time_step: float = 0.001,
                   grip: bool = True, friction: float = 0.8,
                   mass: float = 0.2, html_output: Path | None = None) -> dict:
    # Use the actually achieved planned grasp, not a hand-adjusted pose.
    scene = build_scene(config=config, visualize=False)
    pick = plan_pick(scene, np.array([1.2, -0.1, 0.0]))
    set_robot_configuration(scene, pick.base_q, pick.grasp_q, scene.gripper_closed_positions)
    planned_gripper_pose = scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
    X_GC = planned_gripper_pose.inverse() @ scene.cup_initial_pose
    # Preserve orientation relative to gravity as well as the grasp transform.
    test_gripper_pose = RigidTransform(planned_gripper_pose.rotation(), [0, 0, 0.30])
    builder = DiagramBuilder()
    plant, graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kSap)
    plant.set_contact_model(ContactModel.kPoint)
    plant.set_penetration_allowance(0.0001)
    gripper = Parser(plant).AddModels(url=config["gripper"]["model_url"])[0]
    palm = plant.GetBodyByName("body", gripper)
    stage_instance = plant.AddModelInstance("test_stage")
    stage = plant.AddRigidBody("stage", stage_instance,
                              SpatialInertia(1.0, np.zeros(3), UnitInertia.SolidSphere(0.05)))
    joint = plant.AddJoint(PrismaticJoint("lift", plant.world_frame(),
                                         stage.body_frame(), [0, 0, 1]))
    lift_actuator = plant.AddJointActuator("lift_motor", joint, 200.0)
    lift_actuator.set_controller_gains(PdControllerGains(p=20000.0, d=200.0))
    plant.WeldFrames(stage.body_frame(), palm.body_frame(), test_gripper_pose)
    # Per-instance actuator enumeration is only valid after Finalize.
    for index in plant.GetJointActuatorIndices():
        actuator = plant.get_joint_actuator(index)
        if actuator.model_instance() != gripper:
            continue
        actuator.set_effort_limit(20.0)
        actuator.set_controller_gains(PdControllerGains(p=5000.0, d=20.0))
    spec = config["objects"]["cup"]
    radius, height = float(spec["radius"]), float(spec["height"])
    cup_instance = plant.AddModelInstance("free_cup")
    cup = plant.AddRigidBody("cup", cup_instance, SpatialInertia(
        mass, np.zeros(3), UnitInertia.SolidCylinder(radius, height, [0, 0, 1])))
    plant.RegisterCollisionGeometry(cup, RigidTransform(), Cylinder(radius, height),
                                    "cup", CoulombFriction(friction, friction))
    plant.RegisterVisualGeometry(cup, RigidTransform(), Cylinder(radius, height),
                                "cup_visual", [0.08, 0.40, 0.82, 1.0])
    initial_cup = test_gripper_pose @ X_GC
    support_top = float(initial_cup.translation()[2] - height / 2)
    plant.RegisterCollisionGeometry(plant.world_body(),
        RigidTransform([0, 0, support_top - 0.025]), Box(1, 1, 0.05),
        "support", CoulombFriction(friction, friction))
    plant.RegisterVisualGeometry(plant.world_body(),
        RigidTransform([0, 0, support_top - 0.025]), Box(0.4, 0.4, 0.05),
        "support_visual", [0.5, 0.3, 0.15, 1.0])
    plant.Finalize()
    meshcat = None
    if html_output is not None:
        meshcat = Meshcat()
        MeshcatVisualizer.AddToBuilder(builder, graph, meshcat)
        meshcat.SetCameraPose([0.5, -0.6, 0.6], [0, 0.1, 0.3])
        meshcat.StartRecording()
    diagram = builder.Build()
    simulator = Simulator(diagram)
    root = simulator.get_mutable_context()
    context = plant.GetMyMutableContextFromRoot(root)
    plant.SetPositions(context, gripper, config["gripper"]["open_positions"])
    plant.SetFreeBodyPose(context, cup, initial_cup)
    plant.get_actuation_input_port().FixValue(context, np.zeros(plant.num_actuators()))
    # An inward reference supplies preload; actual finger positions are physics
    # outputs, and effort is limited to 20 N per finger.
    command = [-0.045, 0.045] if grip else config["gripper"]["open_positions"]
    plant.get_desired_state_input_port(gripper).FixValue(context, [*command, 0, 0])
    plant.get_desired_state_input_port(stage_instance).FixValue(context, [0, 0])
    simulator.Initialize()
    rows = []
    for t in np.arange(0.01, 3.001, 0.01):
        alpha = float(np.clip((t - 0.6) / 1.0, 0, 1))
        desired_z = 0.12 * (3*alpha**2 - 2*alpha**3)
        desired_v = 0.12 * (6*alpha - 6*alpha**2) if 0 < alpha < 1 else 0.0
        plant.get_desired_state_input_port(stage_instance).FixValue(context, [desired_z, desired_v])
        simulator.AdvanceTo(float(t))
        pose = plant.EvalBodyPoseInWorld(context, cup)
        palm_pose = plant.EvalBodyPoseInWorld(context, palm)
        relative = palm_pose.inverse() @ pose
        rotation_error = relative.rotation().matrix() @ X_GC.rotation().matrix().T
        angle_error = float(np.arccos(np.clip((np.trace(rotation_error) - 1) / 2, -1, 1)))
        contacts = plant.get_contact_results_output_port().Eval(context)
        finger_forces = {"left_finger": 0.0, "right_finger": 0.0}
        for i in range(contacts.num_point_pair_contacts()):
            info = contacts.point_pair_contact_info(i)
            bodies = (info.bodyA_index(), info.bodyB_index())
            if cup.index() not in bodies:
                continue
            other = bodies[1] if bodies[0] == cup.index() else bodies[0]
            name = plant.get_body(other).name()
            if name in finger_forces:
                finger_forces[name] += float(np.linalg.norm(info.contact_force()))
        rows.append({"time_s": float(t), "cup_z_m": float(pose.translation()[2]),
                     "stage_z_m": float(joint.get_translation(context)),
                     "relative_translation_m": relative.translation().tolist(),
                     "relative_rotation_error_rad": angle_error,
                     "finger_positions_m": plant.GetPositions(context, gripper).tolist(),
                     "finger_contact_force_norms_N": finger_forces})
    if meshcat is not None:
        meshcat.StopRecording()
        meshcat.PublishRecording()
        html_output.parent.mkdir(parents=True, exist_ok=True)
        html_output.write_text(meshcat.StaticHtml())
    held = [r for r in rows if r["time_s"] >= 1.7]
    lift = min(r["cup_z_m"] for r in held) - initial_cup.translation()[2]
    slip = max(np.linalg.norm(np.array(r["relative_translation_m"]) - X_GC.translation()) for r in held)
    bilateral = all(all(f > 0.05 for f in r["finger_contact_force_norms_N"].values()) for r in held)
    rotation_slip = max(r["relative_rotation_error_rad"] for r in held)
    return {"scope": "Isolated actuated vertical-stage lift and hold; no cup weld/attachment or pose resets during simulation",
            "satisfied": bool(lift >= 0.10 and slip <= 0.005 and rotation_slip <= 0.10 and bilateral),
            "minimum_held_lift_m": float(lift), "maximum_translation_slip_m": float(slip),
            "maximum_rotation_slip_rad": rotation_slip,
            "acceptance_thresholds": {"minimum_lift_m": 0.10, "maximum_translation_slip_m": 0.005,
                                      "maximum_rotation_slip_rad": 0.10, "minimum_each_finger_force_N": 0.05},
            "bilateral_contact_throughout_hold": bilateral, "time_step_s": time_step,
            "duration_s": 3.0, "hold_start_s": 1.7, "friction": friction, "cup_mass_kg": mass,
            "contact_model": "point", "solver": "SAP", "grip_enabled": grip,
            "finger_force_limit_N": 20.0, "finger_command_m": command,
            "initial_X_GC": X_GC.GetAsMatrix4().tolist(), "trace": rows,
            "full_robot_dynamics_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--html", type=Path, help="Export the 0.5 ms contact simulation as an interactive recording")
    args = parser.parse_args()
    config = load_scene_config()
    runs = [run_experiment(config, time_step=dt, grip=grip,
                          html_output=args.html if dt == 0.0005 and grip else None)
            for dt, grip in [(0.001, True), (0.0005, True), (0.00025, True), (0.001, False)]]
    report = {"runs": runs, "config": config,
              "suite_satisfied": all(r["satisfied"] for r in runs[:-1]) and not runs[-1]["satisfied"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    for run in runs:
        print(json.dumps({k: v for k, v in run.items() if k not in ("trace", "initial_X_GC")}))
    raise SystemExit(0 if report["suite_satisfied"] else 1)


if __name__ == "__main__":
    main()
