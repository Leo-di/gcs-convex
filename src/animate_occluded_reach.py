"""Generate the complete Room 1 approach and doorway-reach animation.

The base approaches its staging pose with current-geometry Drake GCS. At the
staging pose, a collision-checked arm guide is converted into the existing
mode-specific manipulation GCS. The cup remains free during SAP execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.solvers import Solve

from .build_scene import build_scene, set_robot_configuration
from .grasp_planner import plan_pick, cartesian_arm_path, solve_arm_ik
from .manipulation_gcs import plan_manipulation_phase, phase_plan_to_dict
from .execute_delivery_dynamics import execute
from .scene_spec import load_scene_config
from .geometry_outbound_gcs import plan_geometry_outbound, select_navigation_arm
from .safe_pregrasp import plan_pregrasp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/scene_doorway_reach_semantic_v9.yaml')
    ap.add_argument('--output', default='outputs/results/doorway_arm_reach_semantic_v10/full_animation')
    args = ap.parse_args()
    config = load_scene_config(args.config)
    base_start = np.asarray(config['robot']['start'], float)
    base_q = np.asarray(config['pick_task']['candidates'][0]['base_q'], float)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    (out/'scene_config_snapshot.json').write_text(json.dumps(config, indent=2))
    print('Building Drake scene and solving pickup IK...', flush=True)
    scene = build_scene(config=config, visualize=False)
    print('Selecting a collision-free folded-arm posture and solving outbound base GCS...', flush=True)
    navigation_arm = select_navigation_arm(scene, [base_start, base_q])
    outbound, outbound_audit = plan_geometry_outbound(
        scene, base_start, base_q,
        allowed_region_names=config['pick_task']['outbound_regions'],
        arm_q=navigation_arm,
        spacing=.15,
        max_radius=.18,
    )
    np.savez_compressed(out/'base_navigation_gcs.npz', configurations=outbound.configurations,
                        trajectory_duration_s=outbound.trajectory_duration)
    (out/'base_navigation_gcs_audit.json').write_text(json.dumps(outbound_audit, indent=2))
    pick = plan_pick(scene, base_q)
    print('Pickup IK complete; planning collision-checked arm transition and Cartesian approach...', flush=True)
    pregrasp_transition = plan_pregrasp(scene, base_q, navigation_arm, pick.pregrasp_q)
    approach = cartesian_arm_path(scene, base_q, pick.pregrasp_q, pick.grasp_q, 25, stage='animation_approach')
    pregrasp_reference = np.vstack([
        np.column_stack([np.repeat(base_q[None, :], len(pregrasp_transition), axis=0), pregrasp_transition]),
        np.column_stack([np.repeat(base_q[None, :], len(approach)-1, axis=0), approach[1:]]),
    ])
    print('Running 10-D Drake GCS for pre-grasp and approach...', flush=True)
    pregrasp_plan = plan_manipulation_phase(mode='pregrasp', config=config,
                                            reference_q10=pregrasp_reference,
                                            scene_cache={})
    lift = cartesian_arm_path(scene, base_q, pick.grasp_q, pick.lift_q, 18, stage='animation_lift')
    # Closed-gripper grasp transform, used for attached-mode planning only.
    set_robot_configuration(scene, base_q, pick.grasp_q, scene.gripper_closed_positions)
    X_WG = scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
    X_GC = X_WG.inverse() @ scene.cup_initial_pose

    print('Running 10-D Drake GCS for the attached-cup withdrawal...', flush=True)
    # Solve IK with cup rigidly attached in the planning model. The cup itself
    # remains free in the subsequent physical simulation.
    # Grow attached-cup C-space regions online with Drake IRIS-NP2, then
    # optimize the connected cover with Drake's GCS trajectory optimizer.
    config.setdefault('manipulation_gcs', {})['grasp_region_method'] = 'iris_np2'
    (out/'scene_config_snapshot.json').write_text(json.dumps(config, indent=2))
    attached = build_scene(config=config, attached_cup_X_GC=X_GC, visualize=False)
    arm_refs = [pick.lift_q]
    set_robot_configuration(scene, base_q, pick.lift_q, scene.gripper_closed_positions)
    lift_gripper_pose = scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
    lift_cup_xyz = (lift_gripper_pose @ X_GC).translation()
    waypoints = [
        np.array([0.28, 0.78, 1.02]),
        np.array([0.15, 0.42, 1.02]),
        np.array([-0.07, 0.42, 1.02]),
    ]
    arm_seed = pick.lift_q.copy()
    targets = []
    cursor = np.asarray(lift_cup_xyz, float)
    for endpoint in waypoints:
        endpoint = np.asarray(endpoint, float)
        count = max(2, int(np.ceil(np.linalg.norm(endpoint - cursor) / 0.08)))
        targets.extend((1-alpha)*cursor + alpha*endpoint
                       for alpha in np.linspace(0., 1., count + 1)[1:])
        cursor = endpoint
    for i, cup_xyz in enumerate(targets):
        X_WC = RigidTransform(scene.cup_initial_pose.rotation(), cup_xyz)
        X_WG_target = X_WC @ X_GC.inverse()
        point = np.asarray(config['gripper']['grasp_point_in_body'], float)
        ee_target = X_WG_target.multiply(point)
        arm_seed = solve_arm_ik(
            attached, base_q=base_q, target_position=ee_target,
            target_orientation=X_WG_target.rotation(), initial_arm_q=arm_seed,
            position_tolerance=0.002, orientation_tolerance=0.06,
            minimum_clearance=0.003, stage=f'retract_waypoint_{i}',
        )
        arm_refs.append(arm_seed)
    refs = np.asarray([np.r_[base_q, q] for q in arm_refs])
    plan = plan_manipulation_phase(mode='grasp', config=config,
                                   reference_q10=refs, X_GC=X_GC,
                                   scene_cache={})
    print('Attached-cup GCS complete; assembling physical execution reference...', flush=True)

    # Smooth paths are generated by Drake GCS for the attached transport and
    # Cartesian IK for the free-cup approach/lift; references are time-scaled
    # for stable contact simulation.
    reference = []
    def add(path, phase, grip):
        for q in path:
            reference.append({'base_q': base_q.tolist(), 'arm_q': np.asarray(q).tolist(),
                              'gripper_q': np.asarray(grip).tolist(), 'phase': phase})
    # Keep every tenth point from the already-densely-audited GCS trajectory;
    # the geometric planner's full 1001-sample clearance audit is preserved.
    nav_path = outbound.configurations[::10]
    if not np.allclose(nav_path[-1], outbound.configurations[-1]):
        nav_path = np.vstack([nav_path, outbound.configurations[-1]])
    for q in nav_path:
        reference.append({'base_q': np.asarray(q).tolist(), 'arm_q': navigation_arm.tolist(),
                          'gripper_q': np.asarray(scene.gripper_open_positions).tolist(),
                          'phase': 'navigate_to_pickup'})
    # Label the first GCS transition by its closest pre-grasp waypoint.
    pre_cfg = pregrasp_plan.configurations
    split = int(np.argmin(np.linalg.norm(pre_cfg[:, 3:] - pick.pregrasp_q, axis=1)))
    for q in pre_cfg[:split+1, 3:]:
        reference.append({'base_q': base_q.tolist(), 'arm_q': q.tolist(),
                          'gripper_q': np.asarray(scene.gripper_open_positions).tolist(),
                          'phase': 'arm_to_pregrasp'})
    for q in pre_cfg[split+1:, 3:]:
        reference.append({'base_q': base_q.tolist(), 'arm_q': q.tolist(),
                          'gripper_q': np.asarray(scene.gripper_open_positions).tolist(),
                          'phase': 'approach_cup'})
    # Hold the grasp pose while the fingers close; force control closes onto cup.
    add(np.repeat(pick.grasp_q[None, :], 10, axis=0), 'close_gripper', scene.gripper_closed_positions)
    add(lift[1:], 'lift_cup', scene.gripper_closed_positions)
    add(plan.configurations[1:, 3:], 'whole_body_carry_to_dropoff', scene.gripper_closed_positions)
    (out/'delivery_trace.json').write_text(json.dumps(reference))
    print(f'Executing {len(reference)} reference frames with free-cup SAP contact...', flush=True)
    report, dyn_scene = execute(config, reference, time_step=0.001,
                                time_scale=2.0, visualize=True, grip=True)
    # Task-specific semantic audit for doorway retraction; the delivery runner's
    # bin-placement terminal condition is intentionally not used here.
    trace = report['trace']; door = report['doorway_semantics']
    bilateral = [min(r['finger_contact_force_N'].values()) for r in trace]
    room1_cup = [r for r in trace if r['cup_pose'][0][3] + config['objects']['cup']['radius'] < 0]
    custom = {
        'base_entered_room2': door['base_entered_room2'],
        'base_min_door_clearance_m': door['minimum_base_geometry_clearance_to_door_plane_m'],
        'end_effector_entered_room2': door['end_effector_entered_room2'],
        'maximum_end_effector_room2_depth_m': door['maximum_end_effector_room2_depth_m'],
        'arm_crossed_doorway': door['arm_crossed_doorway'],
        'cup_grasped_with_bilateral_contact': bool(any(x > 0.05 for x in bilateral)),
        'maximum_finger_contact_force_N': door['maximum_grasp_contact_force_N'],
        'cup_retracted_to_room1_while_contacting': bool(any(min(r['finger_contact_force_N'].values()) > .05 for r in room1_cup)),
        'non_expected_collision_count': report['unexpected_contact_count'],
        'dynamics_scope': report['scope'],
    }
    custom['satisfied'] = bool(
        not custom['base_entered_room2'] and custom['end_effector_entered_room2']
        and custom['arm_crossed_doorway'] and custom['cup_grasped_with_bilateral_contact']
        and custom['cup_retracted_to_room1_while_contacting']
        and custom['non_expected_collision_count'] == 0
    )
    # execute() also exposes a delivery-oriented "release into bin" gate.
    # This scenario intentionally ends with a physically held cup in Room 1,
    # so report success against the requested doorway-reach task contract.
    report['execution_task'] = 'doorway_arm_reach_with_room1_retraction'
    report['generic_delivery_release_and_settle_check_applicable'] = False
    report['satisfied'] = bool(
        custom['satisfied'] and report['lifted_with_bilateral_contact']
        and report['carried_to_room1_with_bilateral_contact']
        and report['base_workspace_bounds_satisfied']
        and report['actuator_effort_limits_satisfied']
        and report['carry_slip_satisfied']
    )
    report['doorway_reach_task_audit'] = custom
    report['gcs'] = phase_plan_to_dict(plan)
    report['pregrasp_gcs'] = phase_plan_to_dict(pregrasp_plan)
    report['outbound_gcs_audit'] = outbound_audit
    report['base_navigation'] = {
        'start': base_start.tolist(), 'staging': base_q.tolist(),
        'fixed_arm_q': navigation_arm.tolist(),
        'trajectory_duration_s': outbound.trajectory_duration,
        'sample_count_full_audit': len(outbound.configurations),
    }
    (out/'semantic_trace.json').write_text(json.dumps([
        {'time_s': r['time_s'], 'phase': r['phase'], **r['semantic'],
         'base_q': r['base_q'], 'ee_position_world_m': r['end_effector_position_world_m'],
         'base_geometry_door_clearance_m': r['base_geometry_door_clearance_m'],
         'cup_x_m': r['cup_pose'][0][3],
         'left_finger_contact_force_N': r['finger_contact_force_N']['left_finger'],
         'right_finger_contact_force_N': r['finger_contact_force_N']['right_finger']}
        for r in report['trace']], indent=2))
    (out/'dynamics_trace.json').write_text(json.dumps(report['trace']))
    report.pop('trace', None)
    (out/'result.json').write_text(json.dumps(report, indent=2))
    (out/'scene.html').write_text(dyn_scene.meshcat.StaticHtml())
    print(json.dumps({'audit': custom, 'gcs': report['gcs']}, indent=2), flush=True)
    if not custom['satisfied']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
