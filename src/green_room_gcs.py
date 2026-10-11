"""Staged green-room GCS: tucked navigation, contained reach, close and lift.

The operation cover is an ordered, full-dimensional 7-joint box corridor
around collision-checked guides. A conservative kinematic bound verifies
whole-robot room containment on each complete box, before GCS is called.
Collision safety is independently sampled, not certified for whole boxes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, HPolyhedron, Point
from pydrake.math import RigidTransform
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver

from .build_scene import build_scene, set_robot_configuration
from .collision_checker import evaluate_collision_status
from .geometry_outbound_gcs import plan_geometry_outbound
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .grasp_planner import cartesian_arm_path, plan_pick
from .safe_pregrasp import plan_pregrasp
from .scene_spec import load_scene_config
from .whole_robot_room import WholeRobotRoomOracle, arm_point_reach_bound, shape_support


def room_oracle(scene):
    spec = scene.config['green_room']
    return WholeRobotRoomOracle.from_xy_bounds(scene, spec['x_bounds'], spec['y_bounds'], margin=spec['margin_m'])


def semantic_box_bound(oracle, base, lower, upper, gripper, reach):
    center = (lower + upper) / 2
    residual = oracle.query(np.r_[base, center], gripper)['residual_m']
    displacement = float(reach * np.sum((upper - lower) / 2))
    return {'center_residual_m': residual, 'point_displacement_bound_m': displacement,
            'whole_box_semantic_lower_bound_m': residual - displacement}


def make_operation_cover(scene, base, reference, gripper, *, semantic=True):
    spec = scene.config['green_room']
    oracle = room_oracle(scene)
    reach = arm_point_reach_bound(scene, oracle)
    joint = scene.config['whole_body_planning']
    domain_lo, domain_hi = np.array(joint['joint_lower_limits']), np.array(joint['joint_upper_limits'])
    stride, padding = int(spec['corridor_stride']), float(spec['corridor_padding_rad'])
    if stride < 1 or padding <= 0:
        raise ValueError('Positive corridor stride and padding required')
    indices = sorted(set([*range(0, len(reference), stride), len(reference) - 1]))
    regions, records = [], []

    def add(a, b, depth=0):
        lo = np.maximum(np.minimum(a, b) - padding, domain_lo)
        hi = np.minimum(np.maximum(a, b) + padding, domain_hi)
        if np.any(lo > hi) or np.any(a < lo) or np.any(a > hi) or np.any(b < lo) or np.any(b > hi):
            raise ValueError('Reference violates joint bounds')
        bound = semantic_box_bound(oracle, base, lo, hi, gripper, reach)
        if semantic and bound['whole_box_semantic_lower_bound_m'] < 1e-6:
            if depth >= 16:
                raise RuntimeError('Cannot construct a contained operation box; no whole-C-space infeasibility claim')
            midpoint = (a + b) / 2
            add(a, midpoint, depth + 1)
            add(midpoint, b, depth + 1)
            return
        regions.append(HPolyhedron.MakeBox(lo, hi))
        records.append({'lower': lo.tolist(), 'upper': hi.tolist(), **bound})

    for i, j in zip(indices, indices[1:]):
        add(np.asarray(reference[i]), np.asarray(reference[j]))
    if not regions:
        raise ValueError('Need a nonempty operation reference')
    return regions, records, reach


def solve_operation(scene, base, reference, gripper, name, *, semantic=True):
    began = time.perf_counter()
    regions, records, reach = make_operation_cover(scene, base, reference, gripper, semantic=semantic)
    planner = GcsTrajectoryOptimization(7)
    edges = [(i, i + 1) for i in range(len(regions) - 1)]
    cover = planner.AddRegions(regions, edges, order=1, h_min=.01, h_max=20, name=name)
    source = planner.AddRegions([Point(reference[0])], order=0, name=f'{name}_start')
    target = planner.AddRegions([Point(reference[-1])], order=0, name=f'{name}_goal')
    planner.AddEdges(source, cover, edges_between_regions=[(0, 0)])
    planner.AddEdges(cover, target, edges_between_regions=[(len(regions) - 1, 0)])
    planner.AddPathLengthCost(1.)
    planner.AddTimeCost(.05)
    speed = float(scene.config['whole_body_planning']['maximum_joint_speed'])
    planner.AddVelocityBounds(np.full(7, -speed), np.full(7, speed))
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.max_rounded_paths = 5
    options.rounding_seed = 0
    options.solver = ClarabelSolver()
    options.restriction_solver = ClarabelSolver()
    prepared = time.perf_counter() - began
    tick = time.perf_counter()
    trajectory, result = planner.SolvePath(source, target, options)
    if not result.is_success():
        raise RuntimeError(f'{name} GCS failed: {result.get_solution_result()}')
    seconds = time.perf_counter() - tick
    path, selected_boxes = [], []
    continuity = 0.
    for i in range(trajectory.get_number_of_segments()):
        segment = trajectory.segment(i)
        a = segment.value(segment.start_time()).ravel()
        b = segment.value(segment.end_time()).ravel()
        if path:
            continuity = max(continuity, float(np.max(np.abs(path[-1] - a))))
        path.extend([a, b])
        if segment.end_time() > segment.start_time() + 1e-10:
            containing = [j for j, r in enumerate(regions) if r.PointInSet(a, 1e-6) and r.PointInSet(b, 1e-6)]
            if not containing:
                raise RuntimeError('GCS segment is not contained in an operation box')
            selected_boxes.append(containing[0])
    path = np.asarray(path)
    endpoint_error = max(float(np.max(abs(path[0] - reference[0]))), float(np.max(abs(path[-1] - reference[-1]))))
    if max(endpoint_error, continuity) > 1e-6:
        raise RuntimeError('GCS endpoint/continuity check failed')
    return path, {'name': name, 'region_count': len(regions), 'edges': edges,
                  'independent_variables': 7, 'order': 1, 'semantic_enforced': semantic,
                  'prepare_seconds': prepared, 'solve_seconds': seconds,
                  'objective': float(result.get_optimal_cost()), 'reach_bound_m': reach,
                  'minimum_whole_box_semantic_bound_m': min(r['whole_box_semantic_lower_bound_m'] for r in records),
                  'endpoint_error_rad': endpoint_error, 'continuity_error_rad': continuity,
                  'selected_boxes': selected_boxes, 'boxes': records,
                  'scope': 'Ordered guide-seeded 7D boxes; analytic room bound; sampled collision audits; no route discovery'}, planner.GetGraphvizString()


def append_motion(frames, base_path, arm_path, gripper_path, phase):
    """Retime a piecewise-linear path into uniform execution reference frames."""
    triples = list(zip(base_path, arm_path, gripper_path))
    if not triples:
        raise ValueError('Empty phase')
    if frames:
        previous = np.r_[frames[-1]['base_q'], frames[-1]['arm_q'], frames[-1]['gripper_q']]
        first = np.concatenate(triples[0])
        if np.max(abs(previous - first)) > 1e-6:
            raise RuntimeError(f'Configuration jump entering {phase}')
    for i, (base, arm, grip) in enumerate(triples):
        target = np.r_[base, arm, grip]
        previous = target if i == 0 else np.concatenate(triples[i - 1])
        count = max(1, int(np.ceil(max(np.max(abs(target[:3] - previous[:3])) / .02,
                                      np.max(abs(target[3:10] - previous[3:10])) / .025,
                                      np.max(abs(target[10:] - previous[10:])) / .0005))))
        for alpha in np.linspace(0, 1, count + 1)[1:]:
            q = (1 - alpha) * previous + alpha * target
            frames.append({'base_q': q[:3].tolist(), 'arm_q': q[3:10].tolist(),
                           'gripper_q': q[10:].tolist(), 'phase': phase})


def cup_room_residual(scene, oracle, cup_pose):
    inspector = scene.scene_graph.model_inspector()
    values = []
    for gid in scene.plant.GetCollisionGeometriesForBody(scene.cup_body):
        pose = cup_pose @ inspector.GetPoseInFrame(gid)
        shape = inspector.GetShape(gid)
        for n, b in zip(oracle.normals, oracle.offsets):
            values.append(float(b - n @ pose.translation() - shape_support(shape, pose.rotation().matrix().T @ n) - oracle.margin))
    if not values:
        raise ValueError('Cup has no collision volume')
    return min(values)


def audit_reference(scene, frames, dock, tucked, X_GC):
    oracle = room_oracle(scene)
    spec = scene.config['green_room']
    rows, grasp_rows = [], []
    minimum_collision = float('inf')
    nav_arm_error = op_base_error = 0.
    operation_min = float('inf')
    cup_min = float('inf')
    for i, b in enumerate(frames):
        a = frames[max(0, i - 1)]
        qa = np.r_[a['base_q'], a['arm_q'], a['gripper_q']]
        qb = np.r_[b['base_q'], b['arm_q'], b['gripper_q']]
        count = max(1, int(np.ceil(max(np.max(abs(qb[:3] - qa[:3])) / spec['maximum_audit_base_step_m'],
                                      np.max(abs(qb[3:10] - qa[3:10])) / spec['maximum_audit_joint_step_rad']))))
        for alpha in np.linspace(0, 1, count + 1):
            q = (1 - alpha) * qa + alpha * qb
            active = b['phase'] != 'navigate_to_pickup'
            attached = b['phase'] in ('lift_cup', 'whole_body_carry_to_dropoff')
            inside = oracle.query(q[:10], q[10:])
            pose = scene.cup_initial_pose
            if attached:
                pose = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body) @ X_GC
            scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, pose)
            collision = evaluate_collision_status(scene, q[:3], arm_q=q[3:10], gripper_q=q[10:], influence_distance=.5)
            raw = measure_cup_gripper(scene)
            grasp_rows.append({'attached': attached, 'cup_gripper_audit': raw})
            minimum_collision = min(minimum_collision, collision.minimum_signed_distance)
            cup_value = cup_room_residual(scene, oracle, pose)
            cup_min = min(cup_min, cup_value)
            if active:
                operation_min = min(operation_min, inside['residual_m'])
                op_base_error = max(op_base_error, float(np.max(abs(q[:3] - dock))))
            else:
                nav_arm_error = max(nav_arm_error, float(np.max(abs(q[3:10] - tucked))))
            rows.append({'edge': i, 'alpha': float(alpha), 'phase': b['phase'], 'semantic_active': active,
                         'whole_robot_residual_m': inside['residual_m'], 'worst': inside['worst'],
                         'collision_distance_m': collision.minimum_signed_distance,
                         'cup_room_residual_m': cup_value})
    grasp = summarize_grasp_audit(grasp_rows)
    endpoint = next(f for f in frames if f['phase'] != 'navigate_to_pickup')
    nav_last = [f for f in frames if f['phase'] == 'navigate_to_pickup'][-1]
    switch_error = float(np.max(abs(np.r_[nav_last['base_q'], nav_last['arm_q'], nav_last['gripper_q']]
                                    - np.r_[endpoint['base_q'], endpoint['arm_q'], endpoint['gripper_q']])))
    nav_inside = oracle.query(np.r_[nav_last['base_q'], nav_last['arm_q']], nav_last['gripper_q'])
    report = {'samples': len(rows), 'minimum_collision_distance_m': minimum_collision,
              'minimum_operation_room_residual_m': operation_min, 'minimum_cup_room_residual_m': cup_min,
              'navigation_endpoint_inside': nav_inside, 'phase_switch_error': switch_error,
              'navigation_arm_error_rad': nav_arm_error, 'operation_base_error_m_rad': op_base_error,
              'unfiltered_grasp_audit': grasp, 'scope': 'Dense samples of every executable linear reference edge; not continuous collision certification'}
    report['satisfied'] = bool(minimum_collision >= -1e-7 and operation_min >= -1e-8 and cup_min >= -1e-8
                               and nav_inside['satisfied'] and max(switch_error, nav_arm_error, op_base_error) < 1e-6 and grasp['satisfied'])
    return report, rows


def region_holdout(scene, base, records, gripper, *, X_GC=None):
    rng = np.random.default_rng(scene.config['green_room']['random_seed'])
    oracle = room_oracle(scene)
    collision_bad = semantic_bad = raw_bad = 0
    minimum = float('inf')
    per_region = int(scene.config['green_room']['random_region_samples'])
    for record in records:
        for arm in rng.uniform(record['lower'], record['upper'], size=(per_region, 7)):
            inside = oracle.query(np.r_[base, arm], gripper)
            pose = scene.cup_initial_pose if X_GC is None else scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body) @ X_GC
            scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, pose)
            collision = evaluate_collision_status(scene, base, arm_q=arm, gripper_q=gripper, influence_distance=.2)
            collision_bad += int(collision.minimum_signed_distance < -1e-7)
            semantic_bad += int(not inside['satisfied'])
            minimum = min(minimum, inside['residual_m'])
            raw_bad += int(not summarize_grasp_audit([{'attached': X_GC is not None, 'cup_gripper_audit': measure_cup_gripper(scene)}])['satisfied'])
    return {'samples': len(records) * per_region, 'collisions': collision_bad,
            'raw_grasp_violations': raw_bad, 'semantic_violations': semantic_bad,
            'minimum_room_residual_m': minimum, 'scope': 'Independent uniform holdout; collision counterexamples are retained'}


def run(config_path, output, *, semantic=True, dynamics=True, visualize=True):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    began = time.perf_counter()
    config = load_scene_config(config_path)
    (output / 'scene_config_snapshot.json').write_text(json.dumps(config, indent=2))
    scene = build_scene(config=config, visualize=False)
    spec = config['green_room']
    dock = np.asarray(config['pick_task']['candidates'][0]['base_q'], float)
    start = np.asarray(config['robot']['start'], float)
    tucked = np.asarray(config['pick_task']['navigation_arm_q'], float)
    oracle = room_oracle(scene)
    start_value = oracle.query(np.r_[start, tucked])
    dock_value = oracle.query(np.r_[dock, tucked])
    if not dock_value['satisfied']:
        raise RuntimeError(f'Frozen docking configuration is outside room: {dock_value}')
    print('Frozen dock passes whole-robot containment; solving grasp IK...', flush=True)
    pick = plan_pick(scene, dock)
    transition = plan_pregrasp(scene, dock, tucked, pick.pregrasp_q)
    approach = cartesian_arm_path(scene, dock, pick.pregrasp_q, pick.grasp_q, 25, stage='green_room_approach')
    reach_reference = np.vstack([transition, approach[1:]])
    # Use the continuation branch endpoint as the exact shared grasp state.
    grasp = reach_reference[-1]
    set_robot_configuration(scene, dock, grasp, scene.gripper_closed_positions)
    X_GC = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body).inverse() @ scene.cup_initial_pose
    lift_reference = cartesian_arm_path(scene, dock, grasp, pick.lift_q, 20, stage='green_room_lift')
    print('Solving tucked navigation GCS...', flush=True)
    nav, nav_report = plan_geometry_outbound(scene, start, dock, allowed_region_names=spec['navigation_regions'], arm_q=tucked)
    print('Solving fixed-base contained reach and lift GCS...', flush=True)
    reach, reach_report, reach_dot = solve_operation(scene, dock, reach_reference, scene.gripper_open_positions, 'reach_inside_room', semantic=semantic)
    lift, lift_report, lift_dot = solve_operation(scene, dock, lift_reference, scene.gripper_closed_positions, 'lift_inside_room', semantic=semantic)
    frames = []
    nav_indices = sorted(set([*range(0, len(nav.configurations), 10), len(nav.configurations) - 1]))
    nav_path = nav.configurations[nav_indices]
    repeat = lambda q, n: np.repeat(np.asarray(q)[None, :], n, axis=0)
    append_motion(frames, nav_path, repeat(tucked, len(nav_path)), repeat(scene.gripper_open_positions, len(nav_path)), 'navigate_to_pickup')
    append_motion(frames, repeat(dock, len(reach)), reach, repeat(scene.gripper_open_positions, len(reach)), 'approach_cup')
    append_motion(frames, repeat(dock, 2), repeat(grasp, 2), np.array([scene.gripper_open_positions, scene.gripper_closed_positions]), 'close_gripper')
    frames.extend([dict(frames[-1]) for _ in range(12)])
    append_motion(frames, repeat(dock, len(lift)), lift, repeat(scene.gripper_closed_positions, len(lift)), 'lift_cup')
    append_motion(frames, repeat(dock, 2), repeat(lift[-1], 2), repeat(scene.gripper_closed_positions, 2), 'whole_body_carry_to_dropoff')
    frames.extend([dict(frames[-1]) for _ in range(12)])
    print(f'Auditing {len(frames)} executable reference frames...', flush=True)
    audit, trace = audit_reference(scene, frames, dock, tucked, X_GC)
    region_audits = {
        'reach': region_holdout(scene, dock, reach_report['boxes'], scene.gripper_open_positions),
        'lift': region_holdout(scene, dock, lift_report['boxes'], scene.gripper_closed_positions, X_GC=X_GC)}
    report = {'task': 'green_room_whole_robot_pick', 'semantic_enforced': semantic,
              'start_inside': start_value, 'dock_inside': dock_value, 'dock_q': dock.tolist(),
              'fixed_tucked_arm_q': tucked.tolist(), 'navigation': nav_report,
              'reach': reach_report, 'lift': lift_report, 'reference_audit': audit,
              'region_holdout_audits': region_audits, 'reference_frames': len(frames),
              'gcs_solve_calls': 3, 'architecture': 'Staged 2D navigation + 7D reach + 7D lift; exact configuration sharing; fixed yaw',
              'trajectory_representation': 'Executable piecewise linear path; operation segments from order-1 GCS, navigation resampled from order-1 GCS and independently re-audited',
              'scope': 'Guide-seeded corridors; supplied semantic sequence; cup separately checked; XY-prism room; no automatic missing-mode discovery',
              'planning_seconds': time.perf_counter() - began}
    (output / 'delivery_trace.json').write_text(json.dumps(frames))
    (output / 'semantic_trace.json').write_text(json.dumps(trace))
    (output / 'reach_graph.dot').write_text(reach_dot)
    (output / 'lift_graph.dot').write_text(lift_dot)
    (output / 'result.json').write_text(json.dumps(report, indent=2))
    if not audit['satisfied']:
        raise RuntimeError(f'Green-room reference failed independent audit: {audit}')
    if dynamics:
        from .execute_delivery_dynamics import execute
        print('Reference passed; executing free-cup SAP simulation...', flush=True)
        physics, dyn_scene = execute(config, frames, time_step=.001, time_scale=2., visualize=visualize, grip=True)
        # The delivery evaluator expects return/release. This task instead
        # ends with a stationary lift/hold in the green room.
        dynamic_oracle = room_oracle(dyn_scene)
        active_values, cup_values, fixed_base_errors = [], [], []
        for row in physics['trace']:
            if row['phase'] == 'navigate_to_pickup':
                continue
            value = dynamic_oracle.query(np.r_[row['base_q'], row['arm_q']], row['gripper_q'])
            row['whole_robot_room'] = value
            active_values.append(value['residual_m'])
            cup_values.append(cup_room_residual(dyn_scene, dynamic_oracle, RigidTransform(np.asarray(row['cup_pose']))))
            fixed_base_errors.append(float(np.linalg.norm(np.asarray(row['base_q'])[:2] - dock[:2])))
        task_audit = {'minimum_operation_room_residual_m': min(active_values),
                      'minimum_cup_room_residual_m': min(cup_values),
                      'maximum_operation_base_position_error_m': max(fixed_base_errors),
                      'bilateral_lift': physics['lifted_with_bilateral_contact'],
                      'hold_bilateral_fraction': physics['held_bilateral_contact_fraction'],
                      'unexpected_contacts': physics['unexpected_contact_count'],
                      'effort_limits_satisfied': physics['actuator_effort_limits_satisfied'],
                      'slip_satisfied': physics['carry_slip_satisfied'],
                      'scope': 'Room geometry re-audited at saved 50 ms actual-state samples; contacts at 10 ms; no continuous-time certificate'}
        task_audit['satisfied'] = bool(min(active_values) >= 0 and min(cup_values) >= 0
                                       and max(fixed_base_errors) <= .01 and task_audit['bilateral_lift']
                                       and task_audit['hold_bilateral_fraction'] == 1.
                                       and task_audit['unexpected_contacts'] == 0
                                       and task_audit['effort_limits_satisfied'] and task_audit['slip_satisfied'])
        (output / 'dynamics_trace.json').write_text(json.dumps(physics.pop('trace')))
        physics['generic_delivery_audit_applicable'] = False
        physics['generic_delivery_satisfied'] = physics.pop('satisfied')
        physics['satisfied'] = task_audit['satisfied']
        physics['green_room_task_audit'] = task_audit
        (output / 'physics.json').write_text(json.dumps(physics, indent=2))
        report['physics'] = task_audit
        if visualize:
            (output / 'scene.html').write_text(dyn_scene.meshcat.StaticHtml())
    report['satisfied'] = bool(audit['satisfied'] and (not dynamics or report['physics']['satisfied']))
    report['dynamics_executed'] = dynamics
    report['total_seconds'] = time.perf_counter() - began
    report['source_sha256'] = {str(p.relative_to(Path.cwd())): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in [Path(__file__), Path(__file__).with_name('whole_robot_room.py'), Path(config_path).resolve()]}
    (output / 'result.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ['satisfied', 'reference_frames', 'planning_seconds', 'dynamics_executed']}, indent=2), flush=True)
    if not report['satisfied']:
        raise RuntimeError('Green-room execution failed task audit; artifacts retained')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/scene_green_room.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-dynamics', action='store_true')
    parser.add_argument('--no-visualize', action='store_true')
    parser.add_argument('--no-semantic', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, semantic=not args.no_semantic, dynamics=not args.no_dynamics, visualize=not args.no_visualize)


if __name__ == '__main__':
    main()
