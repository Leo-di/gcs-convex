from pathlib import Path
import numpy as np
import pytest

from src.animate_delivery_demo import plan_delivery_candidate
from src.build_scene import build_scene, set_robot_configuration
from src.candidate_planner import CandidateRejected
from src.grasp_audit import measure_cup_gripper, summarize_grasp_audit
from src.grasp_planner import plan_pick
from src.scene_spec import load_scene_config
from src.task_semantics import pick_guard_residuals


def test_real_legacy_grasp_penetration_is_not_hidden_by_filters():
    scene = build_scene(visualize=False, config=load_scene_config(Path(__file__).parent / "fixtures/legacy_grasp.yaml"))
    pick = plan_pick(scene, np.array([1.2, -0.1, 0]))
    set_robot_configuration(scene, pick.base_q, pick.grasp_q, scene.gripper_closed_positions)
    report = summarize_grasp_audit([{"cup_gripper_audit": measure_cup_gripper(scene)}])
    assert not report["satisfied"]
    assert report["minimum_pair"]["distance_m"] < -0.02


def test_strict_mode_rejects_legacy_grasp_before_iris():
    config = load_scene_config(Path(__file__).parent / "fixtures/legacy_grasp.yaml")
    config["validation"]["require_grasp_geometry"] = True
    scene = build_scene(visualize=False, config=config)
    attempt = {}
    with pytest.raises(CandidateRejected, match="Unfiltered grasp geometry"):
        plan_delivery_candidate(scene, np.array([1.2, -0.1, 0]), attempt)
    assert attempt["stage"] == "grasp_geometry"


def test_missing_geometry_does_not_pass_audit():
    assert not summarize_grasp_audit([])["satisfied"]
    assert not summarize_grasp_audit([{"cup_gripper_audit": {"pairs": []}}])["satisfied"]


def test_numerical_contact_allowance_applies_only_to_tip_geometry():
    def report(name):
        return summarize_grasp_audit([{"cup_gripper_audit": {"pairs": [
            {"body": "left_finger", "geometry": name, "distance_m": -0.0002}]}}])
    assert report("gripper::left_top")["satisfied"]
    assert not report("gripper::collision")["satisfied"]


def test_oriented_grasp_with_appropriate_aperture_reduces_endpoint_penetration():
    config = load_scene_config()
    config["gripper"].update(open_positions=[-0.055, 0.055], closed_positions=[-0.0475, 0.0475])
    config["pick_task"].update(gripper_rpy=[0, 0, 0], orientation_tolerance=0.005,
                               position_tolerance=0.0002)
    scene = build_scene(config=config, visualize=False)
    pick = plan_pick(scene, np.array([1.2, -0.1, 0]))
    set_robot_configuration(scene, pick.base_q, pick.grasp_q, scene.gripper_closed_positions)
    audit = summarize_grasp_audit([{"cup_gripper_audit": measure_cup_gripper(scene)}])
    assert audit["satisfied"], audit
    observation = {"base_q": pick.base_q, "gripper_q": scene.gripper_closed_positions,
                   "cup_pose": scene.cup_initial_pose.GetAsMatrix4(),
                   "gripper_pose": scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context).GetAsMatrix4()}
    assert pick_guard_residuals(observation, config)["grasp_orientation"] >= -1e-8
    observation["gripper_pose"][:3, :3] = np.diag([-1, -1, 1])
    assert pick_guard_residuals(observation, config)["grasp_orientation"] < 0


def test_attached_cup_requires_contact_from_both_fingers():
    pairs = [{"body": body, "geometry": body + "_top", "distance_m": 0.0001}
             for body in ("left_finger", "right_finger")]
    row = {"attached": True, "cup_gripper_audit": {"pairs": pairs}}
    assert summarize_grasp_audit([row])["satisfied"]
    pairs[1]["distance_m"] = 0.007
    report = summarize_grasp_audit([row])
    assert not report["satisfied"]
    assert report["first_violation"]["rule"] == "missing_bilateral_contact"


def test_default_filters_preserve_palm_and_finger_box_collisions():
    scene = build_scene(visualize=False)
    inspector = scene.scene_graph.model_inspector()
    cup_id = scene.plant.GetCollisionGeometriesForBody(scene.cup_body)[0]
    seen = {True: 0, False: 0}
    for body_index in scene.plant.GetBodyIndices(scene.gripper_instance):
        for gid in scene.plant.GetCollisionGeometriesForBody(scene.plant.get_body(body_index)):
            tip = any(inspector.GetName(gid).endswith(s) for s in ("_top", "_bottom", "_left", "_right"))
            assert inspector.CollisionFiltered(cup_id, gid) == tip
            seen[tip] += 1
    assert seen[True] == 8 and seen[False] == 3


def test_world_orientation_target_is_not_inverted():
    from pydrake.math import RotationMatrix
    from src.grasp_planner import solve_arm_ik
    scene = build_scene(visualize=False)
    pick = plan_pick(scene, np.array([1.2, -0.1, 0]))
    target = RotationMatrix.MakeZRotation(0.15)
    position = np.array(scene.config["objects"]["cup"]["center"]) + np.array([0, 0, .22])
    q = solve_arm_ik(scene, base_q=pick.base_q, target_position=position,
                     initial_arm_q=pick.pregrasp_q, position_tolerance=.0002,
                     minimum_clearance=.003, target_orientation=target,
                     orientation_tolerance=.005)
    set_robot_configuration(scene, pick.base_q, q, scene.gripper_open_positions)
    actual = scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context).rotation()
    angle = np.arccos(np.clip((np.trace(actual.matrix() @ target.matrix().T)-1)/2, -1, 1))
    assert angle <= .005 + 1e-6


def test_corrected_full_delivery_passes_unfiltered_geometry_and_semantics():
    scene = build_scene(visualize=False)
    result = plan_delivery_candidate(scene, np.array([1.2, -.1, 0]), {})
    assert result["grasp_audit"]["satisfied"]
    assert result["grasp_audit"]["bilateral_contact_frames_checked"] > 100
    assert result["collision_free"]
    assert result["monitor_report"]["satisfied"]
    retreat = np.array([r["arm_q"] for r in result["trace"] if r["phase"] == "retreat_arm"])
    assert np.max(np.abs(np.diff(retreat, axis=0))) < 0.1
    assert not result["grasp_audit"]["physical_grasp_validated"]


def test_dynamics_restores_all_cup_contacts_and_forbids_attachment():
    from pydrake.math import RigidTransform
    scene = build_scene(visualize=False, dynamics_time_step=.001)
    inspector = scene.scene_graph.model_inspector()
    cup = scene.plant.GetCollisionGeometriesForBody(scene.cup_body)[0]
    bodies = [scene.plant.get_body(i) for i in scene.plant.GetBodyIndices(scene.gripper_instance)]
    bodies.append(scene.plant.GetBodyByName('table'))
    for body in bodies:
        for gid in scene.plant.GetCollisionGeometriesForBody(body):
            assert not inspector.CollisionFiltered(cup, gid)
    assert scene.plant.time_step() == .001
    with pytest.raises(ValueError, match='free cup'):
        build_scene(visualize=False, dynamics_time_step=.001, attached_cup_X_GC=RigidTransform())
