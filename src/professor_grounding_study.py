"""Reproducible supplied-plan / SE(2) and grasp-pose grounding experiment."""
import argparse
import hashlib
import json
from pathlib import Path

from .scene_spec import PROJECT_ROOT, load_scene_config
from .semantic_refinement import generate_candidates
from .operation_refinement_study import prepare


def run(output, config_path=PROJECT_ROOT/'configs/professor_grounding.yaml', *, rotated_only=False):
    from .joint_grounding_study import search, validate_bridge_yaws
    spec = load_scene_config(config_path)
    cfg, protocol = prepare(load_scene_config(PROJECT_ROOT/spec['scene']), spec['layout'],
        budget_seconds=spec['total_search_seconds'], candidate_seconds=spec['per_candidate_seconds'])
    cfg['semantic_refinement'].update(spec['sampling'])
    if rotated_only:
        cfg['semantic_refinement']['base_yaw_candidates'] = [
            yaw for yaw in cfg['semantic_refinement']['base_yaw_candidates'] if abs(yaw) > 1e-8]
    cfg['grounding_semantic_plan'] = spec['semantic_plan']
    cfg['green_room']['operation_refinement'] = {'max_rounds': 2, 'max_regions': 512, 'time_budget_s': 60.}
    protocol['candidates'] = generate_candidates(cfg)
    validate_bridge_yaws(cfg, protocol['candidates'], protocol['bridge_method'])
    protocol.update(tasks=[{'id': 'supplied_plan'}], rotated_only=rotated_only,
        comparison='Single development run, no comparative performance claim',
        scope='Supplied semantic plan; finite base SE(2) and grasp position/orientation library; task ends after free-cup lift/hold. Numerical audits, no continuous-time proof.')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir()
    (root/'configs/supplied_plan.json').write_text(json.dumps(cfg, indent=2))
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    manifest = {}
    for path in sorted((PROJECT_ROOT/'src').glob('*.py')):
        dest = root/'source_snapshot'/path.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        manifest[str(path.relative_to(PROJECT_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    result = search(root, 'supplied_plan', 'operation_first')
    (root/'result.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/professor_grounding.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--rotated-only', action='store_true', help='Freeze candidate library to nonzero base yaw before planning')
    args = p.parse_args()
    if not run(args.output, args.config, rotated_only=args.rotated_only)['success']:
        raise SystemExit(1)
