"""Existing planners as a connector between arbitrary open-gripper pieces.

Only fixed-posture base motion followed by fixed-base arm motion is supported.
Payload/contact-mode changes need an explicit task adapter; they fail closed.
"""
import copy
from pathlib import Path
import numpy as np

from .semantic_plan import configuration, SemanticPlanEvaluator


def audit_open_reference(scene, frames, *, step=.005, max_samples=200000):
    """Independent sampled collision/raw-gripper/limits check, no payload."""
    from .collision_checker import evaluate_collision_status
    from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
    if not frames or not np.isfinite(step) or step <= 0 or max_samples < 1:
        raise ValueError('Nonempty path and positive audit budgets required')
    lower = np.r_[scene.config['robot']['lower_limits'],
                  scene.config['whole_body_planning']['joint_lower_limits']]
    upper = np.r_[scene.config['robot']['upper_limits'],
                  scene.config['whole_body_planning']['joint_upper_limits']]
    count, bad, minimum = 0, 0, None
    for i, frame in enumerate(frames):
        a = configuration(frames[max(0, i-1)])
        b = configuration(frame)
        samples = max(1, int(np.ceil(np.max(np.abs(b-a))/step)))+1
        if count+samples > max_samples:
            raise RuntimeError('Bridge geometry audit sample budget exhausted')
        for alpha in np.linspace(0, 1, samples):
            q = (1-alpha)*a+alpha*b
            collision = evaluate_collision_status(scene, q[:3], arm_q=q[3:10], gripper_q=q[10:])
            raw = summarize_grasp_audit([{'attached': False, 'cup_gripper_audit': measure_cup_gripper(scene)}])
            distance = float(collision.minimum_signed_distance)
            minimum = distance if minimum is None else min(minimum, distance)
            count += 1
            bad += int(not (collision.collision_free and raw['satisfied']
                       and np.all(q[:10] >= lower-1e-8) and np.all(q[:10] <= upper+1e-8)
                       and np.max(np.abs(q[10:]-scene.gripper_open_positions)) <= 1e-8))
    return {'satisfied': bad == 0, 'samples': count, 'violations': bad,
            'minimum_collision_distance_m': minimum,
            'scope': 'Finite interpolated geometry/raw-gripper/joint-limit checks; no payload or continuous-time certificate'}


def connect_open_gripper(scene, previous, target, *, plan, phase, artifact_dir,
                         base_method='rotation_safe_gcs', seed=20261019, bindings=None):
    from .automatic_motion_bridge import plan_bridge
    from .build_scene import build_scene
    from .green_room_gcs import append_motion, solve_operation
    from .safe_pregrasp import plan_pregrasp
    from .accepted_motion import require_endpoint

    a, b = configuration(previous), configuration(target)
    opened = np.asarray(scene.gripper_open_positions)
    if (scene.attached_cup_instance is not None
            or any(f.get('payload_mode', 'free') != 'free' for f in (previous, target))):
        raise ValueError('Generic bridge does not support payload/contact-mode transitions')
    if (np.max(np.abs(a[10:]-opened)) > 1e-8 or np.max(np.abs(b[10:]-opened)) > 1e-8):
        raise ValueError('Generic bridge supports open gripper and no payload only')
    folder = Path(artifact_dir)
    folder.mkdir(parents=True, exist_ok=False)
    # Use a fresh planning context so failed branches cannot change the scene.
    cfg = copy.deepcopy(scene.config)
    cfg.pop('grounding_semantic_plan', None)
    phase_spec = next(p for p in plan['phases'] if p['id'] == phase)
    room_refs = [plan['predicates'][ref] for ref in phase_spec.get('invariants', [])
                 if plan['predicates'][ref]['kind'] == 'whole_robot_in_room']
    if len(room_refs) > 1:
        raise ValueError('Bridge domain supports at most one analytic room; split the supplied phase')
    if room_refs:
        room = room_refs[0]
        bounds = cfg['semantic_regions'][room['region']]
        cfg['green_room'].update(x_bounds=bounds['x_bounds'], y_bounds=bounds['y_bounds'],
                                 margin_m=room.get('margin_m', 0.))
    local = build_scene(config=cfg, visualize=False)
    evaluator = SemanticPlanEvaluator(local, plan, bindings=bindings)
    refs = phase_spec.get('invariants', [])
    def state_valid(base, arm):
        return all(evaluator.evaluate(ref, np.r_[base, arm, opened])['satisfied'] for ref in refs)
    frames, reports = [], {}
    base = a[:3]
    if np.max(np.abs(a[:3]-b[:3])) > 1e-6:
        navigation, report = plan_bridge(local, a[:3], b[:3], a[3:10], base_method,
            folder/'base_bridge.json', seed=seed)
        path = navigation.configurations
        append_motion(frames, path, np.repeat(a[3:10][None, :], len(path), axis=0),
            np.repeat(opened[None, :], len(path), axis=0), phase)
        # The intermediate base endpoint is not the final arm endpoint: apply
        # entry and invariants here, and reserve exit predicates for completion.
        base_phase = dict(phase_spec, exit=[])
        base_audit = SemanticPlanEvaluator(local, dict(plan, phases=[base_phase]),
            bindings=bindings).audit(frames)
        base_geometry = audit_open_reference(local, frames)
        if not base_audit['satisfied'] or not base_geometry['satisfied']:
            raise RuntimeError('Base bridge rejected before arm planning')
        reports['base_acceptance'] = {'semantic': base_audit, 'geometry': base_geometry}
        reports['base'] = report
        base = b[:3]
    if np.max(np.abs(a[3:10]-b[3:10])) > 1e-6:
        guide = plan_pregrasp(local, base, a[3:10], b[3:10],
            state_constraint=lambda q: state_valid(base, q))
        path, report, dot = solve_operation(local, base, guide, opened, phase,
            semantic=bool(room_refs), artifacts=folder/'arm_cover.json')
        append_motion(frames, np.repeat(base[None, :], len(path), axis=0), path,
            np.repeat(opened[None, :], len(path), axis=0), phase)
        (folder/'arm_graph.dot').write_text(dot)
        reports['arm'] = report
    if not frames:
        frames = [dict(previous, phase=phase), dict(target, phase=phase)]
    require_endpoint(frames[0], a, 'Bridge entry')
    require_endpoint(frames[-1], b, 'Bridge exit')
    # Audit every edge and endpoint even when the planner accepted its path.
    single_plan = dict(plan, phases=[phase_spec])
    audit = SemanticPlanEvaluator(local, single_plan, bindings=bindings).audit(frames)
    geometry = audit_open_reference(local, frames)
    if not audit['satisfied'] or not geometry['satisfied']:
        raise RuntimeError('Motion bridge fails independent geometry/semantic audit')
    return frames, {'planners': reports, 'semantic_audit': audit, 'geometry_audit': geometry,
                    'scope': 'Open-gripper reference bridge; official GCS paths; no contact execution'}
