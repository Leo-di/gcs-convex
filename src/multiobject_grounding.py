"""Ground and execute a supplied sequence of pick/lift/replace object actions.

All objects coexist in one continuous free-body contact simulation. Planning
uses explicit held/free modes and payload geometry. No execution pose reset,
attachment, concatenated single-object simulations, or interpolation fallback.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import time
import numpy as np

from .scene_spec import PROJECT_ROOT, load_scene_config
from .build_scene import build_scene, set_robot_configuration
from .semantic_plan import SemanticPlanEvaluator, configuration


class GraspCandidateRejected(RuntimeError):
    def __init__(self, details):
        self.stage = 'closed_grasp_candidate_rejected'
        self.details = details
        super().__init__('Closed grasp lacks valid unfiltered bilateral contact before region generation')


def accept_closed_grasp(scene, base, arm, folder, label):
    """Reject an invalid rigid grasp before spending a region-refinement budget."""
    from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
    set_robot_configuration(scene, base, arm, scene.gripper_closed_positions)
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, scene.cup_initial_pose)
    measured = measure_cup_gripper(scene)
    report = summarize_grasp_audit([{'attached': True, 'cup_gripper_audit': measured}])
    report.update(stage=label, raw_geometry=measured)
    (folder / (label + '_grasp_acceptance.json')).write_text(json.dumps(report, indent=2))
    if not report['satisfied']:
        raise GraspCandidateRejected(report)
    return report


def validate_task(spec):
    if not spec.get('objects') or not spec.get('actions') or 'cup' not in spec['objects']:
        raise ValueError('Multi-object task needs named objects, cup and ordered actions')
    ids = []
    for object_id, center in spec['objects'].items():
        if not isinstance(object_id, str) or not object_id or np.asarray(center).shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError('Named objects need finite position triples')
    for action in spec['actions']:
        if action.get('primitive') != 'pick_lift_replace':
            raise ValueError('Unsupported multi-object primitive; executable adapter required')
        if (not isinstance(action['id'], str) or not action['id'] or any(c in action['id'] for c in '/\\.')
                or action['id'] in ids or action['object_id'] not in spec['objects']):
            raise ValueError('Unique action IDs and known object identities required')
        bases = np.asarray(action['base_candidates'])
        if bases.ndim != 2 or bases.shape[1] != 3 or not len(bases) or not np.all(np.isfinite(bases)):
            raise ValueError('Finite nonempty base candidate triples required')
        ids.append(action['id'])
    if type(spec['max_attempts']) is not int or spec['max_attempts'] < 1 or not np.isfinite(spec['time_scale']) or spec['time_scale'] <= 0:
        raise ValueError('Positive attempt and timing budgets required')


def target_config(config, object_id):
    result = copy.deepcopy(config)
    primary_id = result.get('active_object_id', 'cup')
    objects = {primary_id: result['objects']['cup'], **result['objects'].get('additional_cups', {})}
    result['objects']['cup'] = copy.deepcopy(objects[object_id])
    result['objects']['additional_cups'] = {key: copy.deepcopy(value) for key, value in objects.items() if key != object_id}
    result['active_object_id'] = object_id
    return result


def audit_action(scene, frames, X_GC, *, max_samples=100000):
    from .collision_checker import evaluate_collision_status
    from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
    from .green_room_gcs import room_oracle, cup_room_residual
    oracle = room_oracle(scene)
    from .robot_keep_clear import RobotKeepClearEvaluator
    keep_clear = RobotKeepClearEvaluator(scene)
    from .object_exclusion import ObjectExclusionEvaluator
    exclusions = ObjectExclusionEvaluator(scene)
    from .conditional_permission import ConditionalPermissionEvaluator
    permissions = ConditionalPermissionEvaluator(scene)
    exclusion_min = None
    minimum, room_min, cup_min = None, None, None
    bad, count, examples = 0, 0, []
    for i, f in enumerate(frames):
        a, b = configuration(frames[max(0, i-1)]), configuration(f)
        n = max(1, int(np.ceil(np.max(np.abs(b-a)/np.r_[[.01]*10, [.001]*2]))))
        if count+n+1 > max_samples:
            raise RuntimeError('Multi-object audit budget exhausted')
        for alpha in np.linspace(0, 1, n+1):
            q = (1-alpha)*a+alpha*b
            set_robot_configuration(scene, q[:3], q[3:10], q[10:])
            held = f['mode'] in ('lift', 'hold', 'lower')
            pose = (scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body) @ X_GC
                    if held else scene.cup_initial_pose)
            scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, pose)
            collision = evaluate_collision_status(scene, q[:3], arm_q=q[3:10], gripper_q=q[10:])
            raw = summarize_grasp_audit([{'attached': held, 'cup_gripper_audit': measure_cup_gripper(scene)}])
            permission = permissions.current([scene.config.get('active_object_id', 'cup')]
                                             if held and raw['satisfied'] else [])
            reserved = keep_clear.current()
            value = oracle.query(q[:10], q[10:])
            residual = cup_room_residual(scene, oracle, pose)
            excluded = exclusions.current(scene)
            if excluded['minimum_residual_m'] is not None:
                v = excluded['minimum_residual_m']
                exclusion_min = v if exclusion_min is None else min(exclusion_min, v)
            minimum = collision.minimum_signed_distance if minimum is None else min(minimum, collision.minimum_signed_distance)
            room_min = value['residual_m'] if room_min is None else min(room_min, value['residual_m'])
            cup_min = residual if cup_min is None else min(cup_min, residual)
            valid = (collision.minimum_signed_distance >= -1e-7 and raw['satisfied']
                     and value['satisfied'] and residual >= -1e-8 and excluded['satisfied'] and permission['satisfied']
                     and reserved['satisfied'])
            count += 1
            if not valid:
                bad += 1
                if len(examples) < 5:
                    examples.append({'mode': f['mode'], 'edge': i, 'alpha': float(alpha),
                        'collision_m': collision.minimum_signed_distance, 'raw': raw, 'exclusion':excluded,
                        'permission':permission, 'robot_keep_clear':reserved})
    return {'satisfied': bad == 0, 'samples': count, 'violations': bad, 'counterexamples': examples,
        'minimum_collision_distance_m': minimum, 'minimum_robot_room_residual_m': room_min,
        'minimum_payload_room_residual_m': cup_min,
        'minimum_object_exclusion_residual_m': exclusion_min,
        'scope': 'Finite reference audit; all other objects remain collision-active; target uses explicit held/free geometry'}


def plan_action(config, action, base, previous, folder):
    from .grasp_planner import plan_pick, cartesian_arm_path
    from .safe_pregrasp import plan_pregrasp
    from .green_room_gcs import operation_oracle, guide_edge_validator, solve_operation, append_motion
    from .automatic_motion_bridge import plan_bridge
    scene = build_scene(config=target_config(config, action['object_id']), visualize=False)
    tucked = np.asarray(config['pick_task']['navigation_arm_q'])
    base = np.asarray(base)
    opened, closed = scene.gripper_open_positions, scene.gripper_closed_positions
    oracle = operation_oracle(scene)
    from .conditional_permission import ConditionalPermissionEvaluator
    set_robot_configuration(scene, base, tucked, opened)
    from .robot_keep_clear import RobotKeepClearEvaluator
    reserved = RobotKeepClearEvaluator(scene).current()
    if not reserved['satisfied']:
        from .stateful_transport_grounding import TransportRejected
        (folder/'keep_clear_rejection.json').write_text(json.dumps(reserved, indent=2))
        raise TransportRejected('pickup_blocks_reserved_space', reserved)
    permission = ConditionalPermissionEvaluator(scene).current()
    if not permission['satisfied']:
        from .stateful_transport_grounding import TransportRejected
        (folder/'empty_hand_permission_rejection.json').write_text(json.dumps(permission, indent=2))
        raise TransportRejected('empty_hand_pickup_permission_rejected', permission)
    pick = plan_pick(scene, base)
    grasp_acceptance = {'ik': accept_closed_grasp(scene, base, pick.grasp_q, folder, 'ik')}
    if not all(oracle.query(np.r_[base, q])['satisfied'] for q in (tucked, pick.pregrasp_q, pick.grasp_q, pick.lift_q)):
        raise RuntimeError('Multi-object goal rejected by whole-robot semantics')
    guide_report = {}
    transition = plan_pregrasp(scene, base, tucked, pick.pregrasp_q, diagnostics=guide_report,
        state_constraint=lambda q: oracle.query(np.r_[base, q])['satisfied'])
    approach = cartesian_arm_path(scene, base, pick.pregrasp_q, pick.grasp_q, 25, stage='multi_approach',
        edge_validator=guide_edge_validator(scene, base, pick.pregrasp_q, opened, semantic=True))
    guide = np.vstack([transition, approach[1:]])
    grasp_acceptance['approach'] = accept_closed_grasp(scene, base, guide[-1], folder, 'approach')
    set_robot_configuration(scene, base, guide[-1], closed)
    X_GC = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body).inverse() @ scene.cup_initial_pose
    liftguide = cartesian_arm_path(scene, base, guide[-1], pick.lift_q, 20, stage='multi_lift',
        edge_validator=guide_edge_validator(scene, base, guide[-1], closed, semantic=True, X_GC=X_GC))
    reach, rr, _ = solve_operation(scene, base, guide, opened, action['id']+'_reach', artifacts=folder/'reach.json')
    lift, lr, _ = solve_operation(scene, base, liftguide, closed, action['id']+'_lift', X_GC=X_GC, artifacts=folder/'lift.json')
    # Restore the planning object after payload holdout probes.
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, scene.cup_initial_pose)
    frames, operations = [], []
    phase_audits = {}
    from .accepted_motion import require_endpoint
    repeat = lambda q, n: np.repeat(np.asarray(q)[None, :], n, axis=0)
    def add(arm_path, grips, mode, dwell=0):
        phase = action['id']+'.'+mode
        local = []
        append_motion(local, repeat(base, len(arm_path)), arm_path, grips, phase)
        local.extend([copy.deepcopy(local[-1]) for _ in range(dwell)])
        for f in local:
            f.update(object_id=action['object_id'], mode=mode)
        if frames:
            require_endpoint(local[0], configuration(frames[-1]), phase)
        checked = audit_action(scene, local, X_GC)
        phase_audits[phase] = checked
        (folder/'phase_acceptance.json').write_text(json.dumps(phase_audits, indent=2))
        if not checked['satisfied']:
            raise RuntimeError(f'Multi-object phase {phase} rejected before composition')
        frames.extend(local)
    add(reach, repeat(opened, len(reach)), 'reach')
    add(repeat(reach[-1], 2), np.array([opened, closed]), 'close', dwell=12)
    add(lift, repeat(closed, len(lift)), 'lift')
    add(repeat(lift[-1], 12), repeat(closed, 12), 'hold')
    add(lift[::-1], repeat(closed, len(lift)), 'lower')
    add(repeat(reach[-1], 2), np.array([closed, opened]), 'release', dwell=20)
    add(reach[::-1], repeat(opened, len(reach)), 'retreat')
    operation_audit = audit_action(scene, frames, X_GC)
    if not operation_audit['satisfied']:
        (folder/'rejected_operation_audit.json').write_text(json.dumps(operation_audit, indent=2))
        raise RuntimeError('Multi-object complete operation reference rejected')
    operations = frames
    navframes = []
    nav_start = np.asarray(previous['base_q'])
    if np.max(np.abs(nav_start-base)) > 1e-6:
        nav, navreport = plan_bridge(scene, nav_start, base, tucked, 'rotation_safe_gcs', folder/'bridge.json')
        indices = sorted(set([*range(0, len(nav.configurations), 10), len(nav.configurations)-1]))
        path = nav.configurations[indices]
    else:
        path = repeat(base, 2)
        navreport = {'method': 'shared_complete_configuration'}
    append_motion(navframes, path, repeat(tucked, len(path)), repeat(opened, len(path)), action['id']+'.arrive')
    for f in navframes:
        f.update(object_id=action['object_id'], mode='arrive')
    require_endpoint(navframes[0], configuration(previous), 'Action bridge entry')
    require_endpoint(operations[0], configuration(navframes[-1]), 'Action bridge exit')
    phase_audits[action['id']+'.arrive'] = audit_action(scene, navframes, X_GC)
    (folder/'phase_acceptance.json').write_text(json.dumps(phase_audits, indent=2))
    if not phase_audits[action['id']+'.arrive']['satisfied']:
        raise RuntimeError('Multi-object navigation phase rejected')
    return navframes+operations, {'phase_acceptance': phase_audits, 'operation_audit': operation_audit, 'reach': rr, 'lift': lr,
        'pregrasp_guide': guide_report, 'grasp_acceptance': grasp_acceptance,
        'bridge': navreport, 'goal_grasp_pose': pick.grasp_pose}


def execute_sequence(config, frames, plan, *, visualize=False, grip=True):
    from .execute_delivery_dynamics import execute
    observations = []
    def observe(scene, row):
        observations.append(row)
    physics, scene = execute(config, frames, time_scale=config['execution_dynamics']['reference_time_scale'],
        visualize=visualize, grip=grip, observation_hook=observe)
    semantic = SemanticPlanEvaluator(scene, plan, execution=True).audit(physics['trace'], require_shared_boundaries=False)
    object_checks = {}
    for object_id, initial in scene.object_initial_poses.items():
        active = [r for r in observations if r['object_id'] == object_id]
        held = [r['objects'][object_id] for r in active if r['mode'] == 'hold']
        settled = [r['objects'][object_id] for r in active if r['mode'] == 'release'][-50:]
        lifted = any(r['objects'][object_id]['pose'][2][3] >= initial.translation()[2]+.10
            and min(r['objects'][object_id]['contacts_N'][side] for side in ('left_finger', 'right_finger')) > .05 for r in active)
        bilateral = bool(held and all(min(h['contacts_N'][side] for side in ('left_finger', 'right_finger')) > .05 for h in held))
        stable = bool(len(settled) >= 20 and all(s['contacts_N']['table'] > .05 and s['speed_m_s'] < .01
            and s['angular_speed_rad_s'] < .05 and max(s['contacts_N'][side] for side in ('left_finger', 'right_finger')) < .05 for s in settled))
        slip = max((float(np.linalg.norm(np.asarray(h['relative_translation_m'])-held[0]['relative_translation_m'])) for h in held), default=None)
        final = observations[-1]['objects'][object_id]
        final_error = float(np.linalg.norm(np.array(final['pose'])[:3, 3]-initial.translation()))
        object_checks[object_id] = {'lifted_with_bilateral_contact': lifted, 'bilateral_throughout_hold': bilateral,
            'released_and_settled_on_table': stable, 'hold_slip_m': slip, 'final_position_error_m': final_error,
            'satisfied': bool(lifted and bilateral and stable and slip <= .01 and final_error <= .02)}
    result = {'satisfied': bool(all(c['satisfied'] for c in object_checks.values()) and semantic['satisfied']
        and physics['unexpected_contact_count'] == 0 and physics['actuator_effort_limits_satisfied']),
        'objects': object_checks, 'semantic_audit': semantic,
        'unexpected_contacts': physics['unexpected_contact_count'], 'effort_limits': physics['actuator_effort_limits_satisfied'],
        'cup_attachment': False, 'object_pose_resets_after_initialization': 0,
        'scope': 'One continuous free-body SAP simulation; ideal planar base; sequential pick/lift/replace, finite observation checks'}
    return result, physics, scene


def run(output, config_path=PROJECT_ROOT/'configs/multiobject_grounding.yaml', *, visualize=False):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = load_scene_config(config_path)
    validate_task(spec)
    config = load_scene_config(PROJECT_ROOT/spec['scene'])
    primary = copy.deepcopy(config['objects']['cup'])
    config['objects']['cup']['center'] = spec['objects']['cup']
    config['objects']['additional_cups'] = {key: dict(copy.deepcopy(primary), center=center)
        for key, center in spec['objects'].items() if key != 'cup'}
    config['pick_task']['gripper_rpy'] = spec['grasp_rpy']
    config['execution_dynamics'] = {'hydroelastic_support': True, 'reference_time_scale': spec['time_scale']}
    config['green_room']['operation_refinement'] = {'max_rounds': 2, 'max_regions': 512, 'time_budget_s': 60.}
    for region in config['gcs_base_planning']['regions']:
        region['yaw_bounds'] = [-.4, .4]
    (root/'protocol.json').write_text(json.dumps({'spec': spec, 'config': config,
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PROJECT_ROOT/'src').glob('*.py'))}}, indent=2))
    attempts, began = [], time.monotonic()
    result = {'satisfied': False, 'attempts': attempts}
    for index, bases in enumerate(itertools.product(*[a['base_candidates'] for a in spec['actions']])):
        if index >= spec['max_attempts']:
            break
        folder = root/f'attempt_{index:02d}'
        folder.mkdir()
        row = {'bases': bases, 'satisfied': False}
        attempts.append(row)
        try:
            frames, reports = [], []
            plan = {'version': 1, 'predicates': {
                'inside': {'kind': 'whole_robot_in_room', 'region': 'room2', 'margin_m': .02},
                'tucked': {'kind': 'configuration_equals', 'indices': list(range(3, 10)),
                    'target': config['pick_task']['navigation_arm_q'], 'tolerance': 1e-6, 'execution_tolerance': .12}}, 'phases': []}
            for action, base in zip(spec['actions'], bases):
                dest = folder/action['id']
                dest.mkdir()
                previous = frames[-1] if frames else {'base_q': config['robot']['start']}
                piece, report = plan_action(config, action, base, previous, dest)
                if frames and np.max(np.abs(configuration(frames[-1])-configuration(piece[0]))) > 1e-6:
                    raise RuntimeError('Multi-object task lost complete configuration continuity')
                reports.append(report)
                frames += piece
                dock = action['id']+'_dock'
                plan['predicates'][dock] = {'kind': 'configuration_equals', 'indices': [0, 1, 2],
                    'target': base, 'tolerance': 1e-6, 'execution_tolerance': [.01]*3}
                for phase in dict.fromkeys(f['phase'] for f in piece):
                    mode = phase.split('.')[-1]
                    plan['phases'].append({'id': phase, 'exit': ['inside'],
                        'invariants': ['tucked'] if mode == 'arrive' else ['inside', dock]})
            audit = SemanticPlanEvaluator(build_scene(config=config, visualize=False), plan).audit(frames)
            if not audit['satisfied']:
                raise RuntimeError('Whole multi-object semantic reference failed')
            (folder/'reference.json').write_text(json.dumps(frames))
            (folder/'plan.json').write_text(json.dumps(plan, indent=2))
            (folder/'planning.json').write_text(json.dumps(reports, indent=2))
            (folder/'config.json').write_text(json.dumps(config, indent=2))
            checks, physics, scene = execute_sequence(config, frames, plan, visualize=visualize)
            (folder/'physics.json').write_text(json.dumps(physics))
            (folder/'execution_checks.json').write_text(json.dumps(checks, indent=2))
            if visualize:
                (folder/'scene.html').write_text(scene.meshcat.StaticHtml())
            row.update(satisfied=checks['satisfied'], execution=checks)
            if checks['satisfied']:
                result.update(satisfied=True, selected_attempt=index)
        except Exception as exc:
            row['reason'] = str(exc)
        result['wall_seconds'] = time.monotonic()-began
        (root/'result.json').write_text(json.dumps(result, indent=2))
        print(index, row.get('satisfied'), row.get('reason'), flush=True)
        if result['satisfied']:
            break
    return result


def replay(source, output, *, visualize=False, grip=True):
    """Execute frozen references again, including an open-gripper control."""
    source, root = Path(source).resolve(), Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    names = ('config.json', 'reference.json', 'plan.json')
    config, frames, plan = [json.loads((source/name).read_text()) for name in names]
    (root/'protocol.json').write_text(json.dumps({'source': str(source), 'grip': grip,
        'inputs_sha256': {name: hashlib.sha256((source/name).read_bytes()).hexdigest() for name in names},
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PROJECT_ROOT/'src').glob('*.py'))}}, indent=2))
    checks, physics, scene = execute_sequence(config, frames, plan, visualize=visualize, grip=grip)
    (root/'execution_checks.json').write_text(json.dumps(checks, indent=2))
    (root/'physics.json').write_text(json.dumps(physics))
    if visualize:
        (root/'scene.html').write_text(scene.meshcat.StaticHtml())
    return checks


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/multiobject_grounding.yaml')
    p.add_argument('--html', action='store_true')
    p.add_argument('--replay-from', type=Path, help='Execute a saved attempt without replanning')
    p.add_argument('--no-grip', action='store_true', help='Keep fingers open for a negative control; requires --replay-from')
    args = p.parse_args()
    if args.no_grip and not args.replay_from:
        p.error('--no-grip requires --replay-from')
    result = (replay(args.replay_from, args.output, visualize=args.html, grip=not args.no_grip)
              if args.replay_from else run(args.output, args.config, visualize=args.html))
    if not result['satisfied']:
        raise SystemExit(1)
