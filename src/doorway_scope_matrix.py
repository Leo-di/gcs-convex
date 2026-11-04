"""Freeze and run a local scene matrix with matched candidate searches.

Every case/method is retained, including timeouts and failed candidates.
Wall time includes screening, planning, physics and replay generation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from .doorway_scope_study import METHODS, method_config


def case_config(original, case, sampling):
    cfg = copy.deepcopy(original)
    values = np.array([case['table_dx'], case['cup_dx'], case['cup_dy'], *case['start_base']], float)
    if values.shape != (6,) or not np.all(np.isfinite(values)):
        raise ValueError('Finite offsets and a three-dimensional base pose required')
    cfg['objects']['table']['center'][0] += float(case['table_dx'])
    cfg['task_zones']['pickup']['center'][0] += float(case['table_dx'])
    cfg['objects']['cup']['center'][0] += float(case['cup_dx'])
    cfg['objects']['cup']['center'][1] += float(case['cup_dy'])
    cfg['robot']['start'] = list(case['start_base'])
    cfg['semantic_refinement'].update(copy.deepcopy(sampling))
    # Prevent an accidental unsupported initial cup from becoming a semantic result.
    table, cup = cfg['objects']['table'], cfg['objects']['cup']
    for axis in (0, 1):
        if abs(cup['center'][axis]-table['center'][axis])+cup['radius'] > table['size'][axis]/2:
            raise ValueError('Cup footprint must lie on the source tabletop')
    return cfg


def snapshot(root):
    manifest = {}
    for file in sorted((PROJECT_ROOT/'src').glob('*.py')):
        rel = file.relative_to(PROJECT_ROOT); data = file.read_bytes()
        dest = root/'source_snapshot'/rel; dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(data)
        manifest[str(rel)] = hashlib.sha256(data).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def freeze(root, spec):
    from .semantic_refinement import generate_candidates
    from .joint_grounding_study import validate_bridge_yaws
    ids = [c['id'] for c in spec['cases']]
    if (not ids or len(ids) != len(set(ids)) or
            any(not i or any(ch not in 'abcdefghijklmnopqrstuvwxyz0123456789_' for ch in i) for i in ids)):
        raise ValueError('Distinct safe nonempty case IDs required')
    if not 1 <= spec['parallel_workers'] <= 2 or any(not np.isfinite(spec[k]) or spec[k] <= 0 for k in
            ('total_search_seconds','per_candidate_seconds','outer_timeout_slack_seconds')):
        raise ValueError('Positive finite budgets and one or two workers required')
    original = load_scene_config(PROJECT_ROOT/spec['scene'])
    prepared = []
    for case in spec['cases']:
        cfg = case_config(original, case, spec['sampling'])
        candidates = generate_candidates(cfg)
        validate_bridge_yaws(cfg, candidates, 'rotation_safe_gcs')
        if not candidates: raise ValueError('Empty candidate set')
        prepared.append((case, cfg, candidates))
    root.mkdir(parents=True, exist_ok=False)
    (root/'protocol.json').write_text(json.dumps(dict(spec=spec,methods=list(METHODS),
        scope='Six deterministic local development perturbations; all inputs frozen before any solve; no statistical generalization claim.'),indent=2))
    manifest=snapshot(root)
    for case,cfg,candidates in prepared:
        folder=root/case['id'];(folder/'configs').mkdir(parents=True)
        protocol=dict(candidates=candidates,seed=spec['seed'],bridge_method='rotation_safe_gcs',
            total_search_seconds=spec['total_search_seconds'],per_candidate_seconds=spec['per_candidate_seconds'],
            methods=list(METHODS),case=case,candidate_order='Navigation distance, grasp RPY, base coordinates; identical within case',
            candidate_count=len(candidates),base_only_definition='Full base collision volume containment, not base center')
        (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
        (folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
        for method in METHODS:
            (folder/'configs'/f'{method}.json').write_text(json.dumps(method_config(cfg,method),indent=2))
    return manifest


def terminate_tree(proc):
    # Candidate workers start their own sessions; terminate descendants too.
    listing=subprocess.check_output(['ps','-axo','pid=,ppid='],text=True)
    pairs=[tuple(map(int,line.split())) for line in listing.splitlines()]
    descendants={proc.pid}
    while True:
        expanded=descendants|{pid for pid,ppid in pairs if ppid in descendants}
        if expanded==descendants:break
        descendants=expanded
    for pid in reversed(sorted(descendants)):
        try:os.kill(pid,signal.SIGTERM)
        except ProcessLookupError:pass
    try:proc.wait(timeout=5)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    for pid in descendants:
        try:os.kill(pid,signal.SIGKILL)
        except ProcessLookupError:pass


def run(output, config):
    root=Path(output).resolve();spec=load_scene_config(config);manifest=freeze(root,spec)
    env=dict(os.environ,LCM_DEFAULT_URL='memq://',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    def job(case,method):
        folder=root/case['id']
        assert all(hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()==v for p,v in manifest.items()),'Source changed during frozen run'
        began=time.monotonic()
        with (folder/f'{method}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.joint_grounding_study','--output',str(folder),
                '--worker',method,'operation_first'],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=proc.wait(timeout=spec['total_search_seconds']+spec['outer_timeout_slack_seconds'])
            except subprocess.TimeoutExpired:terminate_tree(proc);code='timeout'
        file=folder/method/'operation_first/search_result.json'
        search=json.loads(file.read_text()) if file.exists() else {}
        return dict(case=case['id'],method=method,process_exit=code,wall_seconds=time.monotonic()-began,
                    success=bool(code==0 and search.get('success')),search=search,
                    failure_kind=None if code==0 and search.get('success') else ('outer_timeout' if code=='timeout' else 'search_failed'))
    rows=[]
    with ThreadPoolExecutor(max_workers=spec['parallel_workers']) as pool:
        futures=[pool.submit(job,case,method) for case in spec['cases'] for method in METHODS]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            rows.sort(key=lambda r:([c['id'] for c in spec['cases']].index(r['case']),METHODS.index(r['method'])))
            (root/'comparison.json').write_text(json.dumps(rows,indent=2))
            print(row['case'],row['method'],row['success'],row['search'].get('selected'),row['failure_kind'],flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(PROJECT_ROOT/'configs/doorway_scope_matrix.yaml'))
    a=p.parse_args();run(a.output,a.config)
