"""Endpoint-valid arm motions: matched goal-only/full-motion semantic probe.

Uses the real whole-robot oracle, unaltered scene and official IRIS/GCS.
The same guide generator is used; only the semantic constraint changes.
This is an arm-only reference experiment, not a grasp or dynamics success.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .build_scene import build_scene
from .collision_checker import evaluate_collision_status
from .green_room_gcs import room_oracle, solve_operation
from .safe_pregrasp import plan_pregrasp
from .scene_spec import PROJECT_ROOT, load_scene_config
from .semantic_policy import accepts


def audit_path(scene, base, path, *, step=.0025):
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Positive finite audit step required')
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 7 or not np.all(np.isfinite(path)):
        raise ValueError('A finite seven-joint path with two endpoints is required')
    oracle = room_oracle(scene)
    endpoints = [oracle.query(np.r_[base, q], scene.gripper_open_positions) for q in path[[0, -1]]]
    lower = np.asarray(scene.config['whole_body_planning']['joint_lower_limits'])
    upper = np.asarray(scene.config['whole_body_planning']['joint_upper_limits'])
    rows = []
    for i, (a, b) in enumerate(zip(path, path[1:])):
        for t in np.linspace(0, 1, max(2, int(np.ceil(np.max(np.abs(b-a))/step))+1)):
            q = (1-t)*a+t*b
            value = oracle.query(np.r_[base, q], scene.gripper_open_positions)
            collision = evaluate_collision_status(scene, base, arm_q=q,
                gripper_q=scene.gripper_open_positions, influence_distance=.5)
            rows.append({'edge': i, 'alpha': float(t), 'arm_q': q.tolist(),
                'room_residual_m': value['residual_m'], 'worst': value['worst'],
                'collision_distance_m': collision.minimum_signed_distance,
                'joint_limits_satisfied': bool(np.all(q >= lower) and np.all(q <= upper))})
    minimum = min(row['room_residual_m'] for row in rows)
    clearance = min(row['collision_distance_m'] for row in rows)
    violations = [row for row in rows if row['room_residual_m'] < -1e-8]
    result = {'samples': len(rows), 'endpoints': endpoints,
        'geometry_satisfied': bool(clearance >= -1e-7 and all(row['joint_limits_satisfied'] for row in rows)),
        'goal_semantics_satisfied': all(e['satisfied'] for e in endpoints),
        'motion_semantics_satisfied': not violations,
        'minimum_room_residual_m': minimum, 'minimum_collision_distance_m': clearance,
        'semantic_violations': len(violations),
        'first_semantic_violation': violations[0] if violations else None,
        'scope': 'Fixed base, open gripper, no payload; sampled piecewise-linear arm reference, no dynamics or continuous-time certificate'}
    return result, rows


def run(output, config_path=PROJECT_ROOT/'configs/operation_semantics_probe.yaml'):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    probe = load_scene_config(config_path)
    config = load_scene_config(PROJECT_ROOT/probe['scene'])
    base = np.asarray(probe['base_q'])
    start, goal = np.asarray(probe['start_arm_q']), np.asarray(probe['goal_arm_q'])
    protocol = {'probe': probe, 'scene_config': config,
        'comparison': 'Same endpoints, scene, guide generator/seed, collision model and IRIS/GCS. Only semantic policy changes.',
        'task_scope': 'Arm-only reference motion in Room2. No grasp, lift, navigation or contact-execution success claim.',
        'cohort': 'Development diagnostic chosen by inspecting supports; not an independent test set.',
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PROJECT_ROOT/'src').glob('*.py'))},
        'probe_sha256': hashlib.sha256(Path(config_path).read_bytes()).hexdigest()}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    scene = build_scene(config=config, visualize=False)
    direct, direct_rows = audit_path(scene, base, [start, goal], step=probe['audit_step_rad'])
    (root/'direct_connection.json').write_text(json.dumps(direct, indent=2))
    (root/'direct_trace.json').write_text(json.dumps(direct_rows))
    results = []
    for policy in probe['policies']:
        folder = root/policy
        folder.mkdir()
        cfg = copy.deepcopy(config)
        cfg['semantic_policy'] = policy
        current = build_scene(config=cfg, visualize=False)
        oracle = room_oracle(current)
        semantic = policy == 'full_motion'
        began = time.monotonic()
        row = {'policy': policy, 'planner_succeeded': False, 'physical_execution_validated': False}
        try:
            if not all(oracle.query(np.r_[base, q])['satisfied'] for q in (start, goal)):
                raise RuntimeError('Frozen operation endpoint violates room semantics')
            guide = plan_pregrasp(current, base, start, goal,
                state_constraint=(lambda q: oracle.query(np.r_[base, q])['satisfied']) if semantic else None)
            (folder/'guide.json').write_text(json.dumps(guide.tolist()))
            path, cover, dot = solve_operation(current, base, guide, current.gripper_open_positions,
                'operation_probe', semantic=semantic, artifacts=folder/'operation_cover.json')
            # Accept the actual retimed phase before exposing it to callers.
            from .green_room_gcs import append_motion
            from .accepted_motion import accept_cup_phase
            frames = []
            append_motion(frames, np.repeat(base[None, :], len(path), axis=0), path,
                np.repeat(current.gripper_open_positions[None, :], len(path), axis=0), 'approach_cup')
            segment = accept_cup_phase(current, frames, base, start, None,
                start=np.r_[base, start, current.gripper_open_positions],
                end=np.r_[base, goal, current.gripper_open_positions])
            (folder/'accepted_phase.json').write_text(json.dumps(segment.report, indent=2))
            (folder/'reference.json').write_text(json.dumps(segment.frames))
            audit, trace = audit_path(current, base, path, step=probe['audit_step_rad'])
            endpoint_error = float(max(np.max(np.abs(path[0]-start)), np.max(np.abs(path[-1]-goal))))
            row.update(planner_succeeded=True, audit=audit, endpoint_error_rad=endpoint_error,
                method_satisfied=bool(endpoint_error <= 1e-6 and accepts(policy,
                    geometry=audit['geometry_satisfied'], goals=audit['goal_semantics_satisfied'],
                    motion=audit['motion_semantics_satisfied'])),
                full_task_semantics_satisfied=bool(endpoint_error <= 1e-6 and audit['geometry_satisfied']
                    and audit['goal_semantics_satisfied'] and audit['motion_semantics_satisfied']),
                region_count=cover['region_count'], refinement=cover['refinement'],
                phase_accepted=segment.report['policy_satisfied'],
                joint_path_length_rad=float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()),
                gcs_solve_seconds=cover['solve_seconds'])
            (folder/'path.json').write_text(json.dumps(path.tolist()))
            (folder/'trace.json').write_text(json.dumps(trace))
            (folder/'graph.dot').write_text(dot)
        except Exception as exc:
            row.update(reason=str(exc), method_satisfied=False, full_task_semantics_satisfied=False)
        row['wall_seconds'] = time.monotonic()-began
        results.append(row)
        (folder/'result.json').write_text(json.dumps(row, indent=2))
        (root/'comparison.json').write_text(json.dumps(results, indent=2))
        print(policy, row['planner_succeeded'], row['method_satisfied'], row['full_task_semantics_satisfied'], flush=True)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/operation_semantics_probe.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rows = run(args.output, args.config)
    if not all(row['method_satisfied'] for row in rows):
        raise SystemExit(1)
