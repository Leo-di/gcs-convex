"""Factorial ablation and conflict controls for composed object semantics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .scene_spec import PROJECT_ROOT, load_scene_config
from .semantic_ablation_matrix import bounded


METHODS = ('room_only', 'room_corridor', 'room_exclusion', 'all_constraints')


def method_spec(spec, method):
    if method not in METHODS:
        raise ValueError('Unknown combined semantic policy')
    local = copy.deepcopy(spec)
    local['object_exclusion_display'] = copy.deepcopy(spec['object_exclusions'])
    local['object_corridor_display'] = copy.deepcopy(spec['object_corridors'])
    if method not in ('room_exclusion', 'all_constraints'):
        local['object_exclusions'] = []
    if method not in ('room_corridor', 'all_constraints'):
        local['object_corridors'] = []
    return local


def shadow_trace(spec, trace):
    from .stateful_transport_grounding import make_config
    from .build_scene import build_scene
    from .object_exclusion import ObjectExclusionEvaluator
    from .object_corridor import ObjectCorridorEvaluator
    scene = build_scene(config=make_config(spec), visualize=False)
    exclusion, corridor = ObjectExclusionEvaluator(scene), ObjectCorridorEvaluator(scene)
    samples = []
    for f in trace:
        poses = {k: v['pose'] for k, v in f['objects'].items()}
        samples.append({'time_s': f['time_s'], 'phase': f['phase'],
            'exclusion': exclusion.evaluate(poses), 'corridor': corridor.evaluate(poses, f['phase'])})
    summary = {}
    for key in ('exclusion', 'corridor'):
        active = [r[key] for r in samples if r[key]['minimum_residual_m'] is not None]
        summary[key] = {'samples': len(active), 'violations': sum(not r['satisfied'] for r in active),
            'minimum_residual_m': min((r['minimum_residual_m'] for r in active), default=None)}
    summary['all_object_constraints_satisfied'] = all(r['violations'] == 0 for r in summary.values())
    return samples, summary


def run(output, study_path=PROJECT_ROOT/'configs/combined_semantics_study.yaml'):
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    study = load_scene_config(study_path)
    manifest = {}
    for p in sorted((PROJECT_ROOT/'src').glob('*.py')):
        dest = root/'source_snapshot'/p.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
        manifest[str(p.relative_to(PROJECT_ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    protocol = {'study': study, 'source_sha256': manifest,
        'comparison': 'All methods retain whole-body room and geometry constraints; only corridor/exclusion activation differs.',
        'scope': 'Robot starts inside Room 2. Fixed-arm/yaw transport, finite development scenes; not doorway entry or unconstrained geometry-only planning.'}
    jobs = []
    for case in study['cases']:
        folder = root/case['id']; folder.mkdir()
        spec = load_scene_config(PROJECT_ROOT/case['task'])
        (folder/'spec.json').write_text(json.dumps(spec, indent=2))
        for method in case['methods']:
            file = folder/f'{method}.json'; file.write_text(json.dumps(method_spec(spec, method), indent=2))
            jobs.append((case, method, spec, file))
    (root/'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = []
    for case, method, spec, file in jobs:
        current = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in (PROJECT_ROOT/'src').glob('*.py')}
        if current != manifest:
            raise RuntimeError('Source changed during frozen combined study')
        folder = root/case['id']/method
        command = [sys.executable, '-m', 'src.stateful_transport_grounding', '--config', str(file), '--output', str(folder/'run')]
        if case['html']: command.append('--html')
        code, wall = bounded(command, folder, study['worker_timeout_s'])
        row = {'case': case['id'], 'method': method, 'exit': code, 'wall_seconds': wall, 'accepted': False}
        file_result = folder/'run/result.json'
        if file_result.exists():
            row['result'] = json.loads(file_result.read_text())
            row['accepted'] = code == 0 and row['result']['satisfied']
        file_search = folder/'run/search.json'
        if file_search.exists():
            search = json.loads(file_search.read_text())
            row['planning_satisfied'] = search['satisfied']
            row['failures'] = [b for b in search['branches'] if not b['local_satisfied']]
            if search['satisfied']:
                row['payload_bridge'] = search['selected'][0]['report']['payload_bridge']
        physics = folder/'run/physics.json'
        if physics.exists():
            samples, summary = shadow_trace(spec, json.loads(physics.read_text())['trace'])
            (folder/'shadow_trace.json').write_text(json.dumps(samples))
            row['shadow_all_constraints'] = summary
        rows.append(row); (root/'comparison.json').write_text(json.dumps(rows, indent=2))
        print(case['id'], method, row['accepted'], row.get('shadow_all_constraints'), flush=True)
    from .semantic_plan import configuration
    matches = []
    for case in study['cases']:
        paths = {}
        for method in case['methods']:
            file = root/case['id']/method/'run/reference.json'
            if file.exists():
                paths[method] = [f for f in json.loads(file.read_text()) if f['mode'] == 'carry']
        if paths:
            first = next(iter(paths.values()))
            for method, path in paths.items():
                qerror = max(float(np.max(abs(configuration(first[i])-configuration(path[i])))) for i in (0, -1))
                perror = max(float(np.max(abs(np.array(first[i]['world_objects']['cup'])-np.array(path[i]['world_objects']['cup'])))) for i in (0, -1))
                matches.append(dict(case=case['id'], method=method, configuration_error=qerror,
                    object_pose_error=perror, matched=qerror < 1e-6 and perror < 1e-6))
    (root/'endpoint_matching.json').write_text(json.dumps(matches, indent=2))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--output', required=True)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/combined_semantics_study.yaml')
    a = p.parse_args(); run(a.output, a.config)
