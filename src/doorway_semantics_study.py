"""Matched doorway-table development experiment, including contact execution.

The baseline can choose either side of the door. Both methods use identical
candidate order, geometry, budgets and planners; only room semantics differ.
This small development experiment does not estimate a failure probability.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

from .scene_spec import PROJECT_ROOT, load_scene_config


def run(output, *, budget_seconds=900., candidate_seconds=240.):
    from .joint_grounding_study import search
    from .semantic_refinement import generate_candidates
    from .semantic_plan import cup_semantic_plan
    os.environ.setdefault('LCM_DEFAULT_URL', 'memq://')
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'configs').mkdir()
    cfg = load_scene_config(PROJECT_ROOT/'configs/scene_doorway_table.yaml')
    candidates = generate_candidates(cfg)
    protocol = {'candidates': candidates, 'seed': 20261025,
        'lcm_url': os.environ['LCM_DEFAULT_URL'],
        'bridge_method': 'rotation_safe_gcs',
        'total_search_seconds': budget_seconds, 'per_candidate_seconds': candidate_seconds,
        'candidate_order': 'Same deterministic workspace lattice ordered by navigation distance',
        'comparison': 'Only room constraints differ. Geometry baseline has no supplied room plan. Both enforce tucked navigation, fixed-base operation, contacts, joint limits and complete boundaries.',
        'scope': 'One development layout, two deterministic searches; no failure-frequency or generalization claim'}
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    for policy in ('geometry_only', 'full_motion'):
        local = copy.deepcopy(cfg)
        local['semantic_policy'] = policy
        if policy == 'full_motion':
            local['grounding_semantic_plan'] = cup_semantic_plan(local)
        else:
            local.pop('grounding_semantic_plan', None)
        (root/'configs'/f'{policy}.json').write_text(json.dumps(local, indent=2))
    manifest = {}
    for path in sorted((PROJECT_ROOT/'src').glob('*.py')):
        relative = path.relative_to(PROJECT_ROOT)
        dest = root/'source_snapshot'/relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
        manifest[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    rows = []
    for policy in ('geometry_only', 'full_motion'):
        result = search(root, policy, 'operation_first')
        rows.append(result)
        (root/'comparison.json').write_text(json.dumps(rows, indent=2))
        print(policy, result['success'], result.get('selected'), flush=True)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--budget-seconds', type=float, default=900.)
    parser.add_argument('--candidate-seconds', type=float, default=240.)
    args = parser.parse_args()
    run(args.output, budget_seconds=args.budget_seconds, candidate_seconds=args.candidate_seconds)
