"""Freeze and run paired full-transport tasks with retained failures.

This is a local perturbation study, with a fixed task-specific docking pose
per scene, not a general candidate search or arbitrary-scene benchmark.
"""
import argparse
import concurrent.futures
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

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from src.scene_spec import load_scene_config
from src.planner_randomness import seeded_config
from scripts.object_facing_grasps import facing_yaw

METHODS = ('room_only', 'all_constraints')
SCOPE_ZONE = dict(id='cup_arm_access_zone', object_id='cup',
                  lower_xyz=[.72, .25, .95], upper_xyz=[1.08, .50, 1.30], margin_m=.02)


def scenarios(count=6, seed=2026102601):
    rng = np.random.default_rng(seed)
    base = load_scene_config(PROJECT / 'configs/room_entry_transport.yaml')
    scene = load_scene_config(PROJECT / base['scene'])
    rows = []
    for i in range(count):
        task, cfg = copy.deepcopy(base), copy.deepcopy(scene)
        cup = [float(rng.uniform(.44, .49)), float(rng.uniform(.82, .88)), .84]
        dock = [float(rng.uniform(.44, .50)), float(rng.uniform(-.025, .025)), 0.]
        target = [float(rng.uniform(1.39, 1.46)), float(rng.uniform(.82, .88)), .84]
        task['objects']['cup'] = cup
        task['actions'][0]['pickup_base'] = dock
        task['actions'][0]['placements'] = [target]
        task['entry_navigation']['handoff_base'] = dock
        task['start_base'] = [float(rng.uniform(-1.6, -1.1)), float(rng.uniform(-.22, .22)), float(rng.uniform(-.15, .15))]
        task['grasp_rpy'] = [0., 0., facing_yaw(dock[:2], cup[:2]) - .2]
        task['object_exclusions'][0]['lower_xyz'][0] = float(rng.uniform(.84, .88))
        task['object_exclusions'].append(copy.deepcopy(SCOPE_ZONE))
        floor = float(rng.uniform(.56, .59))
        for cell in task['object_corridors'][0]['cells']:
            cell['lower_xyz'][1] = floor
        half = float(rng.uniform(.60, .72))
        for wall in cfg['world']['walls']:
            if wall['name'] in ('shared_wall_upper', 'shared_wall_lower'):
                wall['size'][1] = 3. - half
                wall['center'][1] = (3. + half) / 2 * (1 if wall['name'].endswith('upper') else -1)
            elif wall['name'] == 'doorway_header':
                wall['size'][1] = 2 * half
        cfg = seeded_config(cfg, 31001)
        rows.append(dict(id=f'scene_{i:02d}', task=task, config=cfg,
                         parameters=dict(cup=cup, dock=dock, target=target, door_half_width_m=half,
                                         start=task['start_base'], corridor_floor_y_m=floor)))
    return rows


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root, count, seed, timeout):
    root.mkdir(parents=True, exist_ok=False)
    generated = scenarios(count, seed)
    jobs, inputs = [], {}
    for row in generated:
        folder = root / row['id']; folder.mkdir()
        scene_path = folder / 'scene.json'
        scene_path.write_text(json.dumps(row['config'], indent=2))
        inputs[str(scene_path.relative_to(root))] = sha(scene_path)
        for method in METHODS:
            spec = copy.deepcopy(row['task']); spec['scene'] = str(scene_path)
            spec['object_exclusion_display'] = copy.deepcopy(spec['object_exclusions'])
            spec['object_corridor_display'] = copy.deepcopy(spec['object_corridors'])
            if method == 'room_only':
                spec['object_exclusions'] = []; spec['object_corridors'] = []
            config = folder / (method + '.json'); config.write_text(json.dumps(spec, indent=2))
            inputs[str(config.relative_to(root))] = sha(config)
            jobs.append(dict(scene=row['id'], method=method, config=str(config.relative_to(root))))
    sources = {}
    for path in [*sorted((PROJECT / 'src').glob('*.py')), Path(__file__).resolve(),
                 PROJECT / 'scripts/object_facing_grasps.py']:
        rel = path.relative_to(PROJECT); dest = root / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(path.read_bytes())
        sources[str(rel)] = sha(path)
    protocol = dict(seed=seed, independent_scenes=count, planner_seed=31001, methods=METHODS,
        worker_timeout_s=timeout, workers=2, jobs=jobs, source_sha256=sources, input_sha256=inputs,
        scenes=[dict(id=r['id'], parameters=r['parameters']) for r in generated],
        distributions=dict(cup_x=[.44,.49], cup_y=[.82,.88], dock_x=[.44,.50], dock_y=[-.025,.025],
            target_x=[1.39,1.46], target_y=[.82,.88], start_x=[-1.6,-1.1], start_y=[-.22,.22],
            start_yaw=[-.15,.15], exclusion_lower_x=[.84,.88], corridor_lower_y=[.56,.59], door_half_width=[.60,.72]),
        scope='IID local perturbations of a developed two-room full pick-transport-place task; one fixed planner seed, one docking candidate per scene. No arbitrary-layout or general grasp-search claim.',
        scope_zone_purpose='Developed arm-access volume: cup excluded, arm permitted. This zone tests entity scope and need not change the cup route; the separate original exclusion tests routing.',
        development='Original room-entry tasks were inspected for arm witnesses: none in the original exclusion. New arm-access volume designed from link locations in that development task, then fixed before this cohort. New scene outcomes were not used to tune it.')
    (root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    return protocol


def run_job(root, protocol, job):
    for rel, value in protocol['source_sha256'].items():
        if sha(PROJECT / rel) != value:
            raise RuntimeError(f'Source changed after freeze: {rel}')
    for rel, value in protocol['input_sha256'].items():
        if sha(root / rel) != value:
            raise RuntimeError(f'Input changed after freeze: {rel}')
    folder = root / job['scene'] / job['method']; folder.mkdir()
    env = dict(os.environ, LCM_DEFAULT_URL='memq://', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    started = time.monotonic()
    with (folder / 'run.log').open('w') as log:
        p = subprocess.Popen([sys.executable, '-m', 'src.stateful_transport_grounding',
            '--config', str(root / job['config']), '--output', str(folder / 'run'), '--html'],
            cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = p.wait(timeout=protocol['worker_timeout_s'])
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL); p.wait()
            code = 'timeout'
    result_path = folder / 'run/result.json'
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    row = dict(**job, exit=code, wall_seconds=time.monotonic()-started,
               accepted=code == 0 and result.get('satisfied', False), result=result)
    (folder / 'condition.json').write_text(json.dumps(row, indent=2))
    print(job['scene'], job['method'], code, row['accepted'], flush=True)
    return row


def run(root, protocol):
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=protocol['workers']) as pool:
        futures = [pool.submit(run_job, root, protocol, job) for job in protocol['jobs']]
        for future in concurrent.futures.as_completed(futures):
            rows.append(future.result())
            (root / 'comparison.json').write_text(json.dumps(sorted(rows, key=lambda r:(r['scene'],r['method'])), indent=2))
    (root / 'complete.json').write_text(json.dumps(dict(conditions=len(rows), expected=len(protocol['jobs']))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True); p.add_argument('--count', type=int, default=6)
    p.add_argument('--seed', type=int, default=2026102601); p.add_argument('--timeout', type=float, default=360)
    args = p.parse_args()
    if args.count < 1 or args.timeout <= 0:
        p.error('Positive scene count and timeout required')
    root = Path(args.output).resolve()
    protocol = freeze(root, args.count, args.seed, args.timeout)
    run(root, protocol)
