"""Frozen paired transport experiments with independent planning and dynamics.

Every complete plan is physically executed. All predefined cases remain in the
denominator; hard stage deadlines, planner rejection and execution rejection
are recorded separately. No cross-run motion cache is used.
"""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback

from .scene_spec import PROJECT_ROOT, load_scene_config

METHODS = ('greedy', 'backtrack')
TERMINAL = {'success', 'planning_rejected', 'execution_rejected', 'planning_timeout',
            'execution_timeout', 'planning_error', 'execution_error'}


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(base):
    """Deterministic perturbations specified before any outcome is observed."""
    layouts = {'replication': copy.deepcopy(base), 'source_right_20mm': copy.deepcopy(base),
               'destination_left_40mm': copy.deepcopy(base)}
    moved = layouts['source_right_20mm']
    for center in moved['objects'].values():
        center[0] += .02
    moved['start_base'][0] += .02
    for action in moved['actions']:
        action['pickup_base'][0] += .02
    moved = layouts['destination_left_40mm']
    moved['destination_table']['center'][0] -= .04
    for action in moved['actions']:
        for position in action['placements']:
            position[0] -= .04
    cells = []
    for layout, spec in layouts.items():
        for order in ('conflict_first', 'free_first'):
            task = copy.deepcopy(spec)
            if order == 'free_first':
                task['actions'][0]['placements'].reverse()
            cells.append({'id': layout+'__'+order, 'layout': layout, 'cohort': 'transport',
                          'order': order, 'task': task})
    narrow = copy.deepcopy(base)
    narrow['carry']['lower_xy'][1] = -.04
    cells.append({'id': 'restricted_cover', 'layout': 'replication', 'cohort': 'representation_control',
                  'order': 'conflict_first', 'task': narrow})
    for i, cell in enumerate(cells):
        cell['methods'] = list(METHODS if i % 2 == 0 else reversed(METHODS))
    return cells


def verify(root):
    protocol = read(root/'protocol.json')
    for name, expected in protocol['input_sha256'].items():
        if file_hash(PROJECT_ROOT/name) != expected:
            raise RuntimeError('Frozen input changed: '+name)
    for cell in protocol['cells']:
        path = root/'inputs'/(cell['id']+'.json')
        if file_hash(path) != protocol['task_sha256'][cell['id']]:
            raise RuntimeError('Frozen task changed: '+cell['id'])
    return protocol


def worker(root, cell_id, method, stage):
    protocol = verify(root)
    if method not in METHODS or cell_id not in {c['id'] for c in protocol['cells']}:
        raise ValueError('Unknown frozen cell or method')
    folder = root/cell_id/method
    record_path = folder/'record.json'
    record = read(record_path)
    began = time.monotonic()
    try:
        task = read(root/'inputs'/(cell_id+'.json'))
        if stage == 'planning':
            from .stateful_transport_grounding import (make_config, search, ground_action,
                                                       semantic_plan, SemanticPlanEvaluator, build_scene)
            cfg = make_config(task)
            result = search(cfg, task, folder, strategy=method,
                            ground=partial(ground_action, pickup_cache={}))
            write(folder/'search.json', {k: v for k, v in result.items() if k != 'frames'})
            accepted = result['satisfied']
            if accepted:
                frames = result['frames']
                plan = semantic_plan(cfg, frames)
                audit = SemanticPlanEvaluator(build_scene(config=cfg, visualize=False), plan).audit(frames)
                write(folder/'assembled_semantic_audit.json', audit)
                accepted = audit['satisfied']
                if accepted:
                    write(folder/'reference.json', frames)
                    write(folder/'plan.json', plan)
                    write(folder/'config.json', cfg)
            record.update(planning_success=accepted, status='planned' if accepted else 'planning_rejected',
                          expansions=result['expansions'], events=result['events'],
                          branch_failures=[{k: b[k] for k in ('index', 'action', 'failure_stage', 'reason') if k in b}
                                           for b in result['branches'] if not b['local_satisfied']])
            if not accepted:
                record['failure'] = ('bounded_search_exhausted' if not result['satisfied']
                                     else 'assembled_semantic_audit_rejected')
        else:
            from .stateful_transport_grounding import execute_reference
            if not record['planning_success']:
                raise ValueError('Execution requires a fully audited plan')
            cfg, frames, plan, search = [read(folder/name) for name in
                                        ('config.json', 'reference.json', 'plan.json', 'search.json')]
            result, physics, _ = execute_reference(cfg, task, frames, plan, search['selected'])
            write(folder/'result.json', result)
            write(folder/'physics.json', physics)
            record.update(status='success' if result['satisfied'] else 'execution_rejected',
                          end_to_end_success=result['satisfied'], actions=result['actions'],
                          online_samples=result['online_samples'],
                          semantic_samples=result['semantic_audit']['samples'],
                          unexpected_contacts=result['unexpected_contacts'])
            if not result['satisfied']:
                record['failure'] = 'contact_or_semantic_acceptance_rejected'
        verify(root)
    except Exception as exc:
        rejected = stage == 'execution' and hasattr(exc, 'stage')
        record.update(status=stage+('_rejected' if rejected else '_error'),
                      failure=getattr(exc, 'stage', repr(exc)),
                      details=getattr(exc, 'details', {}), traceback=traceback.format_exc())
        traceback.print_exc()
    record[stage+'_worker_seconds'] = time.monotonic()-began
    write(record_path, record)


def run_stage(root, cell, method, stage, budget):
    folder = root/cell/method
    began = time.monotonic()
    env = {**os.environ, 'LCM_DEFAULT_URL': 'memq://', 'OPENBLAS_NUM_THREADS': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1'}
    command = [sys.executable, '-m', 'src.benchmark_stateful_transport', '--output', str(root),
               '--worker', stage, '--cell', cell, '--method', method]
    with (folder/(stage+'.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        try:
            process.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    record = read(folder/'record.json')
    if timed_out or process.returncode:
        record.update(status=stage+('_timeout' if timed_out else '_error'),
                      failure='hard_stage_deadline' if timed_out else 'worker_process_error',
                      end_to_end_success=False)
        if stage == 'planning':
            record['planning_success'] = False
    record[stage+'_wall_seconds'] = time.monotonic()-began
    record[stage+'_return_code'] = process.returncode
    write(folder/'record.json', record)
    return record


def summarize(records, cells):
    expected = {(c['id'], m) for c in cells for m in METHODS}
    observed = [(r['cell'], r['method']) for r in records]
    if len(set(observed)) != len(observed) or any(key not in expected for key in observed):
        raise ValueError('Duplicate or unknown result cell')
    lookup = {(r['cell'], r['method']): r for r in records}
    result = {'expected_runs': len(expected), 'recorded_runs': len(records),
              'complete': set(observed) == expected and all(r['status'] in TERMINAL for r in records),
              'cohorts': {}, 'records': records}
    for cohort in dict.fromkeys(c['cohort'] for c in cells):
        subset = [c for c in cells if c['cohort'] == cohort]
        data = {'cells': len(subset), 'methods': {}, 'paired': {'both_success': 0, 'backtrack_only': 0,
                                                             'greedy_only': 0, 'both_failure': 0, 'incomplete': 0}}
        for method in METHODS:
            rows = [lookup[(c['id'], method)] for c in subset if (c['id'], method) in lookup]
            counts = {status: sum(r['status'] == status for r in rows) for status in sorted(TERMINAL)}
            data['methods'][method] = {'denominator': len(subset), 'recorded': len(rows),
                'planning_successes': sum(r['planning_success'] for r in rows),
                'end_to_end_successes': sum(r['end_to_end_success'] for r in rows), 'outcomes': counts}
        for c in subset:
            pair = [lookup.get((c['id'], m)) for m in METHODS]
            if any(r is None or r['status'] not in TERMINAL for r in pair):
                key = 'incomplete'
            else:
                g, b = [r['end_to_end_success'] for r in pair]
                key = 'both_success' if g and b else 'greedy_only' if g else 'backtrack_only' if b else 'both_failure'
            data['paired'][key] += 1
        result['cohorts'][cohort] = data
    return result


def run(root, *, workers=2, planning_budget=600., execution_budget=400.):
    if workers < 1 or any(not 0 < v < float('inf') for v in (planning_budget, execution_budget)):
        raise ValueError('Positive finite stage budgets and workers required')
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'inputs').mkdir()
    cells = design(load_scene_config(PROJECT_ROOT/'configs/stateful_transport_grounding.yaml'))
    for cell in cells:
        # Same soft search budget and expansion allowance for both methods.
        cell['task']['time_budget_s'] = planning_budget
        write(root/'inputs'/(cell['id']+'.json'), cell['task'])
    source_files = sorted((PROJECT_ROOT/'src').glob('*.py'))+sorted((PROJECT_ROOT/'configs').glob('*.yaml'))
    for source in source_files:
        dest = root/'source_snapshot'/source.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    protocol = {'version': 1, 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'cells': [{k: v for k, v in c.items() if k != 'task'} for c in cells],
        'methods': list(METHODS), 'workers': workers,
        'planning_budget_s': planning_budget, 'execution_budget_s': execution_budget,
        'input_sha256': {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in source_files},
        'task_sha256': {c['id']: file_hash(root/'inputs'/(c['id']+'.json')) for c in cells},
        'environment': {'python': sys.version, 'drake': version('drake'), 'ompl': version('ompl')},
        'primary_metric': 'End-to-end acceptance among ALL predefined transport cells; every complete plan executed',
        'scope': 'Small predefined diagnostic cohort, one run per cell/method; orders within a layout are correlated; no statistical generalization or timing advantage claim',
        'control': 'Restricted carrying cover is a representation control, not proof of physical infeasibility; separate denominator',
        'fairness': 'Independent fresh planners and local caches; identical geometry, audit, execution, candidate list and hard budgets; only search strategy differs. Greedy tries first option only.',
        'timeouts': 'Separate hard wall deadlines include process startup and loading for each stage; process group terminated on timeout; no outcome filtering'}
    write(root/'protocol.json', protocol)
    protocol_hash = file_hash(root/'protocol.json')
    lock, finished, active = threading.Lock(), [], []
    def progress():
        write(root/'progress.json', {'finished': len(finished), 'expected': len(cells)*len(METHODS),
                                    'active': list(active), 'records': list(finished)})
    def cell_run(cell):
        for method in cell['methods']:
            key = cell['id']+'/'+method
            folder = root/cell['id']/method
            folder.mkdir(parents=True)
            write(folder/'record.json', {'cell': cell['id'], 'method': method, 'cohort': cell['cohort'],
                'status': 'planning', 'planning_success': False, 'end_to_end_success': False})
            with lock:
                active.append(key)
                progress()
            record = run_stage(root, cell['id'], method, 'planning', planning_budget)
            if record['status'] == 'planned':
                record['status'] = 'execution'
                write(folder/'record.json', record)
                record = run_stage(root, cell['id'], method, 'execution', execution_budget)
            with lock:
                active.remove(key)
                finished.append(record)
                progress()
                print(key, record['status'], flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(cell_run, cells))
    verify(root)
    if file_hash(root/'protocol.json') != protocol_hash:
        raise RuntimeError('Protocol changed during experiment')
    result = summarize(finished, protocol['cells'])
    result['protocol_sha256'] = protocol_hash
    write(root/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', choices=['planning', 'execution'])
    parser.add_argument('--cell')
    parser.add_argument('--method', choices=METHODS)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--planning-budget', type=float, default=600.)
    parser.add_argument('--execution-budget', type=float, default=400.)
    args = parser.parse_args()
    if args.worker:
        worker(args.output.resolve(), args.cell, args.method, args.worker)
    else:
        run(args.output, workers=args.workers, planning_budget=args.planning_budget,
            execution_budget=args.execution_budget)
