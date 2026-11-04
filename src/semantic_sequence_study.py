"""Real IRIS/GCS pieces plus a planned bridge, with whole-plan auditing.

Arm-only reference experiment. Initial states/goals are supplied configurations;
grasp pose sampling and contact execution are tested by professor_grounding_study.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

from .build_scene import build_scene
from .scene_spec import load_scene_config, PROJECT_ROOT
from .semantic_motion_bridge import connect_open_gripper, audit_open_reference
from .semantic_plan import SemanticPlanEvaluator, configuration, ground_sequence


def run(output, config_path=PROJECT_ROOT/'configs/semantic_sequence_probe.yaml'):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = load_scene_config(config_path)
    cfg = load_scene_config(PROJECT_ROOT/spec['scene'])
    scene = build_scene(config=cfg, visualize=False)
    supplied = spec['semantic_plan']
    (root/'protocol.json').write_text(json.dumps({'spec': spec, 'scene_config': cfg,
        'scope': 'Two disconnected arm reference pieces and a planned constrained bridge; no payload or physical execution',
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PROJECT_ROOT/'src').glob('*.py'))}}, indent=2))
    began = time.monotonic()
    reports, stages = {}, []
    def frame(q, phase):
        return {'base_q': spec['base_q'], 'arm_q': q,
                'gripper_q': scene.gripper_open_positions.tolist(), 'phase': phase}
    result = {'satisfied': False, 'physical_execution_validated': False}
    try:
        for piece in spec['pieces']:
            phase = piece['id']
            frames, report = connect_open_gripper(scene, frame(piece['start_arm_q'], phase),
                frame(piece['goal_arm_q'], phase), plan=supplied, phase=phase, artifact_dir=root/phase)
            reports[phase] = report
            stages.append([frames])
        def connect(a, b, depth):
            phase = spec['bridge_phase']
            frames, report = connect_open_gripper(scene, a, b, plan=supplied, phase=phase,
                artifact_dir=root/f'bridge_{depth}')
            reports[phase] = report
            return frames
        def validate(frames, depth):
            phases = list(dict.fromkeys(f['phase'] for f in frames))
            prefix = dict(supplied, phases=supplied['phases'][:len(phases)])
            return (SemanticPlanEvaluator(scene, prefix).audit(frames)['satisfied']
                    and audit_open_reference(scene, frames)['satisfied'])
        grounded = ground_sequence(stages, stages[0][0][0], connect=connect, validate=validate,
            max_attempts=8, time_budget_s=240.)
        result.update({k: v for k, v in grounded.items() if k != 'frames'})
        result['piece_gap_before_bridge_rad'] = float(max(abs(configuration(stages[1][0][0])-configuration(stages[0][0][-1]))))
        if grounded['satisfied']:
            frames = grounded['frames']
            result['semantic_audit'] = SemanticPlanEvaluator(scene, supplied).audit(frames)
            result['geometry_audit'] = audit_open_reference(scene, frames)
            result['satisfied'] = bool(result['semantic_audit']['satisfied'] and result['geometry_audit']['satisfied'])
            (root/'grounded_reference.json').write_text(json.dumps(frames))
        result['reports'] = reports
    except Exception as exc:
        result.update(satisfied=False, reason=str(exc), reports=reports)
    result['wall_seconds'] = time.monotonic()-began
    (root/'result.json').write_text(json.dumps(result, indent=2))
    print({k: result.get(k) for k in ('satisfied', 'reason', 'wall_seconds')}, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/semantic_sequence_probe.yaml')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if not run(args.output, args.config)['satisfied']:
        raise SystemExit(1)
