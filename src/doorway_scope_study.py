"""Matched free-candidate search: geometry, base containment, whole body.

Base containment uses actual base collision volumes, not the center point.
Its fixed-base manipulation domain makes it invariant under arm motion;
phase-local checks still evaluate the emitted path and measured execution.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from .scene_spec import PROJECT_ROOT, load_scene_config

METHODS = ('geometry_only', 'base_only', 'full_motion')


def method_config(original, method):
    from .semantic_plan import cup_semantic_plan
    if method not in METHODS:
        raise ValueError(f'Unknown scope method: {method}')
    cfg = copy.deepcopy(original)
    cfg['semantic_policy'] = 'full_motion' if method == 'full_motion' else 'geometry_only'
    cfg['comparison_method'] = method
    cfg.pop('grounding_semantic_plan', None)
    if method != 'geometry_only':
        cfg['grounding_semantic_plan'] = cup_semantic_plan(cfg)
        if method == 'base_only':
            cfg['grounding_semantic_plan']['predicates']['inside']['kind'] = 'base_in_room'
    return cfg


def run(output, budget_seconds=900., candidate_seconds=240.):
    from .semantic_refinement import generate_candidates
    from .joint_grounding_study import search
    os.environ.setdefault('LCM_DEFAULT_URL', 'memq://')
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir()
    original = load_scene_config(PROJECT_ROOT/'configs/scene_doorway_table.yaml')
    candidates = generate_candidates(original)
    protocol = {'methods': list(METHODS), 'candidates': candidates, 'seed': 20261025,
                'total_search_seconds': budget_seconds, 'per_candidate_seconds': candidate_seconds,
                'bridge_method': 'rotation_safe_gcs',
                'candidate_order': 'Identical workspace lattice sorted by navigation distance',
                'base_only_definition': 'All base collision volumes inside room; arm/gripper unrestricted by room',
                'scope': 'One deterministic development scene; independent docking selection, not matched endpoints or a success-rate estimate.'}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    for method in METHODS:
        (root/'configs'/f'{method}.json').write_text(json.dumps(method_config(original, method), indent=2))
    manifest = {}
    for file in sorted((PROJECT_ROOT/'src').glob('*.py')):
        rel = file.relative_to(PROJECT_ROOT); dest = root/'source_snapshot'/rel
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(file.read_bytes())
        manifest[str(rel)] = hashlib.sha256(file.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for method in METHODS:
        rows.append(search(root, method, 'operation_first'))
        (root/'comparison.json').write_text(json.dumps(rows, indent=2))
        print(method, rows[-1]['success'], rows[-1].get('selected'), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget-seconds', type=float, default=900.)
    parser.add_argument('--candidate-seconds', type=float, default=240.)
    args = parser.parse_args()
    run(args.output, args.budget_seconds, args.candidate_seconds)
