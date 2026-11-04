"""Frozen local ablations with bounded subprocesses and retained failures."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
from .scene_spec import PROJECT_ROOT, load_scene_config


def doorway_config(original, layout):
    cfg = copy.deepcopy(original)
    cfg['objects']['table']['center'][0] += layout['table_dx']
    cfg['objects']['cup']['center'][0] += layout['cup_dx']
    cfg['task_zones']['pickup']['center'][0] += layout['table_dx']
    half = float(layout['door_half_width'])
    if not 0 < half < 1.4:
        raise ValueError('Door half-width must be in (0,1.4)')
    for wall in cfg['world']['walls']:
        if wall['name'] in ('shared_wall_upper', 'shared_wall_lower'):
            wall['size'][1] = 3.-half
            wall['center'][1] = (3.+half)/2 * (1 if wall['name'].endswith('upper') else -1)
        elif wall['name'] == 'doorway_header':
            wall['size'][1] = 2*half
    return cfg


def bounded(command, folder, seconds):
    folder.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    env = dict(os.environ, LCM_DEFAULT_URL='memq://', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (folder/'run.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(); code = 'timeout'
    return code, time.monotonic()-began


def run(output, spec_path=PROJECT_ROOT/'configs/semantic_ablation_study.yaml'):
    from .semantic_plan import cup_semantic_plan
    root = Path(output).resolve(); root.mkdir(parents=True, exist_ok=False)
    spec = load_scene_config(spec_path)
    (root/'protocol.json').write_text(json.dumps(spec, indent=2))
    manifest = {}
    for p in sorted((PROJECT_ROOT/'src').glob('*.py')):
        dest = root/'source_snapshot'/p.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(p.read_bytes())
        manifest[str(p.relative_to(PROJECT_ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    jobs = []
    # Prepare every scene, method and fixed endpoint before running any solver.
    for x in spec['arm_probe']['base_x']:
        for policy in spec['policies']:
            folder = root/'arm'/f'x_{x:.2f}'/policy; folder.mkdir(parents=True)
            probe = {k: v for k, v in spec['arm_probe'].items() if k != 'base_x'}
            probe.update(base_q=[x,0.,0.], policies=[policy])
            file = folder/'probe.json'; file.write_text(json.dumps(probe, indent=2))
            jobs.append(('arm', f'x_{x:.2f}', policy, folder,
                [sys.executable, '-m', 'src.operation_semantics_study', '--output', str(folder/'run'), '--config', str(file)]))
    original = load_scene_config(PROJECT_ROOT/spec['doorway']['scene'])
    for layout in spec['doorway']['layouts']:
        for policy in spec['policies']:
            folder = root/'doorway'/layout['id']/policy; folder.mkdir(parents=True)
            cfg = doorway_config(original, layout); cfg['semantic_policy'] = policy
            if policy == 'full_motion':
                cfg['grounding_semantic_plan'] = cup_semantic_plan(cfg)
            else:
                cfg.pop('grounding_semantic_plan', None)
            file = folder/'config.json'; file.write_text(json.dumps(cfg, indent=2))
            protocol = {'config': str(file), 'candidates': [spec['doorway']['frozen_candidate']],
                'endpoint_first': True, 'operation_first': True, 'bridge_method': 'rotation_safe_gcs', 'seed': 20261025}
            (folder/'protocol.json').write_text(json.dumps(protocol, indent=2))
            jobs.append(('doorway', layout['id'], policy, folder,
                [sys.executable, '-m', 'src.semantic_grounding_candidates', '--output', str(folder), '--worker', '0']))
    rows = []
    for suite, case, policy, folder, command in jobs:
        # Record the exact checkout for each launched worker, in addition to
        # the initial snapshot. This exposes any edits during a long batch.
        worker_hashes = {str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in (PROJECT_ROOT/'src').glob('*.py')}
        (folder/'worker_source_manifest.json').write_text(json.dumps(worker_hashes,indent=2))
        code, elapsed = bounded(command, folder, spec['candidate_seconds'])
        row = {'suite':suite, 'case':case, 'policy':policy, 'exit':code, 'wall_seconds':elapsed,
               'artifact':str(folder.relative_to(root)), 'accepted':False}
        file = (folder/'run'/policy/'result.json' if suite == 'arm' else folder/'fixed_dock/candidate_result.json')
        if file.exists():
            result = json.loads(file.read_text()); row['result'] = result
            if suite == 'arm':
                row['accepted'] = code == 0 and result.get('method_satisfied', False)
                row['failure_stage'] = None if row['accepted'] else 'arm_planning_or_phase_acceptance'
            else:
                row['accepted'] = code == 0 and all(result.get(k,{}).get('accepted') for k in ('operation','bridge','physical'))
                row['failure_stage'] = None if row['accepted'] else result.get('failure_stage', result.get('stage'))
                if not row['accepted'] and result.get('operation',{}).get('accepted'):
                    row['failure_stage'] = ('bridge' if not result.get('bridge',{}).get('accepted') else 'physical_execution')
                row['failure_reason'] = result.get('failure_reason',result.get('bridge',{}).get('reason'))
        else:
            row['failure_stage'] = 'worker_error'
        if code == 'timeout': row['failure_stage'] = 'timeout'
        rows.append(row); (root/'comparison.json').write_text(json.dumps(rows, indent=2))
        print(suite, case, policy, row['accepted'], row['failure_stage'], flush=True)
    matching = []
    for layout in spec['doorway']['layouts']:
        endpoints = {}
        for policy in spec['policies']:
            file=root/'doorway'/layout['id']/policy/'fixed_dock/delivery_trace.json'
            if file.exists():
                frames=json.loads(file.read_text())
                endpoints[policy]=np.array([f['base_q']+f['arm_q']+f['gripper_q'] for phase in
                    ('approach_cup','lift_cup') for f in ([v for v in frames if v['phase']==phase][0],
                                                       [v for v in frames if v['phase']==phase][-1])])
        matching.append({'case':layout['id'], 'available':list(endpoints),
            'all_three_match':len(endpoints)==3 and all(np.max(abs(q-endpoints['full_motion'])) <= 1e-6 for q in endpoints.values()),
            'endpoints':{k:v.tolist() for k,v in endpoints.items()}})
    (root/'endpoint_matching.json').write_text(json.dumps(matching, indent=2))
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.output)
