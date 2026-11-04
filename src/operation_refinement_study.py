"""Self-contained matched fixed-cover/local-repair study; no historical outputs.

Both methods have identical goals, candidate order, IRIS settings, audit gates,
navigation and contact execution. Only the bounded operation repair differs.
These are development layouts, not held-out generalization trials.
"""
import argparse
import copy
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time

from .scene_spec import load_scene_config, PROJECT_ROOT


def prepare(config, layout, *, budget_seconds=600., candidate_seconds=300.):
    if budget_seconds <= 0 or candidate_seconds <= 0:
        raise ValueError('Positive search and candidate budgets required')
    widths = {'narrow': .15, 'medium': .30, 'wide': .45}
    if layout not in widths:
        raise ValueError('Unknown frozen screen layout')
    cfg = copy.deepcopy(config)
    cfg['semantic_policy'] = 'full_motion'
    cfg['world']['walls'].append({'name': 'operation_table_screen',
        'center': [.72, .55, 1.05], 'size': [widths[layout], .04, .60]})
    cfg['green_room']['region_method'] = 'iris_np2'
    cfg['green_room']['require_clean_region_holdout'] = True
    for box in cfg['gcs_base_planning']['regions']:
        if box['name'] in cfg['green_room']['navigation_regions']:
            box['yaw_bounds'] = [-.4, .4]
    candidates = []
    for x, y, (j, angle) in itertools.product([.45, .65, .85, 1.05], [-.15, 0., .15],
                                              enumerate([0., -.2, .2])):
        candidates.append({'base_q': [x, y, 0.], 'grasp_rpy': [0., 0., angle],
            'orientation_index': j,
            'ranking_score': float(((x-cfg['robot']['start'][0])**2+(y-cfg['robot']['start'][1])**2)**.5)})
    candidates.sort(key=lambda c: (round(c['ranking_score'], 10), c['orientation_index'], c['base_q']))
    for i, candidate in enumerate(candidates):
        candidate['id'] = f'joint_{i:03d}'
    protocol = {'tasks': [], 'candidates': candidates, 'methods': ['operation_first'],
        'seed': 20261019, 'bridge_method': 'rotation_safe_gcs',
        'total_search_seconds': budget_seconds, 'per_candidate_seconds': candidate_seconds,
        'candidate_order': 'Navigation distance, predefined grasp orientation index, base coordinates',
        'comparison': 'Only operation_refinement.max_rounds changes (0 vs 2). Same two audit gates, scene, candidates, algorithms, budgets and physical criteria.',
        'scope': 'Prespecified existing development layout, task ends after free-cup lift/hold. Finite audits; no generalization, completeness or continuous safety claim.'}
    return cfg, protocol


def run(output, *, config_path=PROJECT_ROOT/'configs/scene_semantic_choice_v6.yaml',
        layout='narrow', budget_seconds=600., candidate_seconds=300., methods=('fixed', 'refined')):
    from .joint_grounding_study import search
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir()
    cfg, protocol = prepare(load_scene_config(config_path), layout,
                            budget_seconds=budget_seconds, candidate_seconds=candidate_seconds)
    protocol['runtime'] = {'python': sys.version, 'executable': sys.executable}
    protocol['tasks'] = [{'id': method} for method in methods]
    protocol['layout'] = layout
    for method in methods:
        local = copy.deepcopy(cfg)
        local['green_room']['operation_refinement'] = {
            'max_rounds': 0 if method == 'fixed' else 2, 'max_regions': 512, 'time_budget_s': 60.}
        (root/'configs'/f'{method}.json').write_text(json.dumps(local, indent=2))
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    manifest = {}
    for path in sorted((PROJECT_ROOT/'src').glob('*.py')):
        relative = path.relative_to(PROJECT_ROOT)
        dest = root/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for method in methods:
        started = time.monotonic()
        result = search(root, method, 'operation_first')
        rows.append(dict(result, repair_method=method, wall_seconds=time.monotonic()-started))
        (root/'comparison.json').write_text(json.dumps(rows, indent=2))
        print(method, result['success'], result.get('selected'), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/scene_semantic_choice_v6.yaml')
    parser.add_argument('--layout', choices=['narrow', 'medium', 'wide'], default='narrow')
    parser.add_argument('--methods', nargs='+', choices=['fixed', 'refined'], default=['fixed', 'refined'])
    parser.add_argument('--budget-seconds', type=float, default=600.)
    parser.add_argument('--candidate-seconds', type=float, default=300.)
    args = parser.parse_args()
    rows = run(args.output, config_path=args.config, layout=args.layout,
               budget_seconds=args.budget_seconds, candidate_seconds=args.candidate_seconds,
               methods=args.methods)
    if not all(row['success'] for row in rows):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
