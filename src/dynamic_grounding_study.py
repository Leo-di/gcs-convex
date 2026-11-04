"""Schedule an audited GCS cup trajectory around moving customer occupancy.

The customer follows supplied occupancy keyframes; it is a geometric semantic
constraint, not a force-driven human. Robot/cup still use free-body contact
dynamics. A learned room surrogate is exercised with an analytical guard.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np

from .scene_spec import PROJECT_ROOT, load_scene_config
from .build_scene import build_scene
from .semantic_plan import SemanticPlanEvaluator
from .dynamic_semantics import schedule_with_waits
from .learned_semantics import train


def run(source, output, *, config_path=PROJECT_ROOT/'configs/dynamic_customer_grounding.yaml', visualize=False):
    from .execute_delivery_dynamics import execute
    from .green_room_gcs import audit_reference, cup_room_residual, room_oracle
    from .build_scene import set_robot_configuration
    from pydrake.math import RigidTransform
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    source = Path(source)
    cfg = json.loads((source/'scene_config_snapshot.json').read_text())
    frames = json.loads((source/'delivery_trace.json').read_text())
    spec = load_scene_config(config_path)
    cfg['execution_dynamics'] = {'hydroelastic_support': True, 'reference_time_scale': spec['time_scale']}
    scene = build_scene(config=cfg, visualize=False)
    model_path = root/'learned_room.json'
    model = train(scene, model_path)
    plan = copy.deepcopy(cfg['grounding_semantic_plan'])
    plan['predicates']['inside'] = {'kind': 'learned_room', 'region': 'room2', 'margin_m': .02,
        'model_path': str(model_path), 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}
    plan['predicates']['customer'] = spec['customer']
    # Literal targets make the plan self-contained for scheduling/replay.
    result_source = json.loads((source/'result.json').read_text())
    dock, tucked = result_source['dock_q'], result_source['fixed_tucked_arm_q']
    plan['predicates']['docked']['target'] = dock
    plan['predicates']['tucked']['target'] = tucked
    for phase in plan['phases']:
        phase.setdefault('invariants', []).append('customer')
    cfg['grounding_semantic_plan'] = plan
    (root/'protocol.json').write_text(json.dumps({'spec': spec, 'config': cfg,
        'source_reference_sha256': hashlib.sha256((source/'delivery_trace.json').read_bytes()).hexdigest(),
        'source': str(source.resolve()), 'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((PROJECT_ROOT/'src').glob('*.py'))}}, indent=2))
    scheduled = schedule_with_waits(frames, plan, scene, candidate_delays_s=spec['candidate_delays_s'],
        frame_period_s=.05*spec['time_scale'])
    (root/'schedule.json').write_text(json.dumps({k: v for k, v in scheduled.items() if k not in ('frames', 'timed_frames')}, indent=2))
    result = {'satisfied': False, 'learned_model_metrics': model['metrics'], 'schedule_satisfied': scheduled['satisfied']}
    if not scheduled['satisfied']:
        (root/'result.json').write_text(json.dumps(result, indent=2))
        return result
    (root/'reference.json').write_text(json.dumps(scheduled['frames']))
    (root/'config.json').write_text(json.dumps(cfg, indent=2))
    grasp = next(f for f in scheduled['frames'] if f['phase'] == 'close_gripper')
    set_robot_configuration(scene, grasp['base_q'], grasp['arm_q'], scene.gripper_closed_positions)
    X_GC = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body).inverse() @ scene.cup_initial_pose
    reference_audit, _ = audit_reference(scene, scheduled['frames'], np.array(dock), np.array(tucked), X_GC)
    if not reference_audit['satisfied']:
        raise RuntimeError('Scheduled reference fails full geometry/semantic audit')
    # Use a separate planning context for online semantic checks, so the
    # evaluator cannot write any state into the running physical simulation.
    monitor = SemanticPlanEvaluator(build_scene(config=cfg, visualize=False), plan)
    online = {'samples': 0, 'violations': 0, 'minimum_customer_residual_m': None}
    def observe(dynamic_scene, row):
        q = np.concatenate([dynamic_scene.plant.GetPositions(dynamic_scene.plant_context, instance)
                  for instance in (dynamic_scene.base_instance, dynamic_scene.arm_instance, dynamic_scene.gripper_instance)])
        value = monitor.evaluate('customer', q, time_s=row['time_s'])
        online['samples'] += 1
        online['violations'] += int(not value['satisfied'])
        online['minimum_customer_residual_m'] = value['residual'] if online['minimum_customer_residual_m'] is None else min(online['minimum_customer_residual_m'], value['residual'])
        if not value['satisfied']:
            raise RuntimeError('Online moving customer constraint rejected execution')
    physics, dynamic_scene = execute(cfg, scheduled['frames'], time_scale=spec['time_scale'],
        visualize=visualize, observation_hook=observe)
    actual = SemanticPlanEvaluator(dynamic_scene, plan, execution=True).audit(physics['trace'], require_shared_boundaries=False)
    oracle = room_oracle(dynamic_scene)
    cup_residual = min(cup_room_residual(dynamic_scene, oracle, RigidTransform(np.array(f['cup_pose']))) for f in physics['trace'])
    checks = {'plan': actual['satisfied'], 'online_customer': online['violations'] == 0,
        'bilateral_lift': physics['lifted_with_bilateral_contact'], 'bilateral_hold': physics['held_bilateral_contact_fraction'] == 1.,
        'unexpected_contacts': physics['unexpected_contact_count'] == 0, 'effort_limits': physics['actuator_effort_limits_satisfied'],
        'slip': physics['carry_slip_satisfied'], 'cup_room': cup_residual >= -1e-8}
    result.update(satisfied=all(checks.values()), checks=checks, selected_delay_s=scheduled['delay_s'],
        online=online, actual_plan_audit=actual, minimum_cup_room_residual_m=cup_residual,
        cup_attachment=False, cup_pose_resets_after_initialization=0,
        scope='Scripted moving customer occupancy + force-driven robot/free-cup execution; finite checks, not crowd dynamics')
    (root/'physics.json').write_text(json.dumps(physics))
    (root/'result.json').write_text(json.dumps(result, indent=2))
    if visualize:
        (root/'scene.html').write_text(dynamic_scene.meshcat.StaticHtml())
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True, help='Completed cup candidate folder with scene and reference snapshots')
    p.add_argument('--output', required=True)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/dynamic_customer_grounding.yaml')
    p.add_argument('--html', action='store_true')
    args = p.parse_args()
    result = run(args.source, args.output, config_path=args.config, visualize=args.html)
    print({k: result.get(k) for k in ('satisfied', 'selected_delay_s', 'checks')})
    if not result['satisfied']:
        raise SystemExit(1)
