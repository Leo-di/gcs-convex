"""Bounded candidate grounding around the existing staged Drake GCS planner.

The scene is immutable. Only docking and grasp orientation are searched.
Failure is scoped to this finite candidate set / planner budget, never a
certificate of configuration-space infeasibility. No reference-path fallback.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .build_scene import build_scene
from .collision_checker import evaluate_collision_status
from .green_room_gcs import room_oracle
from .scene_spec import load_scene_config
from .semantic_policy import policy_from_config


def generate_candidates(config):
    spec = config['semantic_refinement']
    step = float(spec['xy_step_m'])
    if not np.isfinite(step) or step <= 0:
        raise ValueError('Positive finite candidate spacing required')
    bounds = np.asarray(spec['base_xy_bounds'], float)
    if bounds.shape != (2, 2) or not np.all(np.isfinite(bounds)) or np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError('Expected ordered finite x/y bounds')
    # Global lattice, independent of any manually supplied docking point.
    axes = [np.arange(np.ceil(lo / step), np.floor(hi / step) + 1) * step for lo, hi in bounds]
    cup = np.asarray(config['objects']['cup']['center'][:2])
    yaws = np.asarray(spec.get('base_yaw_candidates', [config['robot']['start'][2]]), float)
    if yaws.ndim != 1 or not len(yaws) or not np.all(np.isfinite(yaws)) or len(np.unique(yaws)) != len(yaws):
        raise ValueError('Distinct finite base yaw candidates required')
    poses = []
    for x, y, yaw in itertools.product(*axes, yaws):
        for rpy in spec['grasp_rpy_candidates']:
            if np.asarray(rpy).shape != (3,) or not np.all(np.isfinite(rpy)):
                raise ValueError('Finite grasp RPY triples required')
            score = abs(np.linalg.norm(np.array([x, y]) - cup) - spec['preferred_standoff_m'])
            distance = float(np.linalg.norm(np.array([x,y]) - cup))
            if 'standoff_bounds_m' in spec and not spec['standoff_bounds_m'][0] <= distance <= spec['standoff_bounds_m'][1]:
                continue
            if spec.get('ranking', 'standoff') == 'navigation_distance':
                score = float(np.linalg.norm(np.array([x,y]) - np.asarray(config['robot']['start'][:2])))
            poses.append({'base_q': [float(x), float(y), yaw], 'grasp_rpy': list(rpy),
                          'ranking_score': float(score)})
    poses.sort(key=lambda p: (round(p['ranking_score'], 10), p['grasp_rpy'], p['base_q']))
    for i, pose in enumerate(poses):
        pose['id'] = f'candidate_{i:03d}'
    return poses


def screen_candidates(config, candidates):
    scene = build_scene(config=config, visualize=False)
    oracle = room_oracle(scene)
    tucked = np.asarray(config['pick_task']['navigation_arm_q'])
    rows = []
    for candidate in candidates:
        base = candidate['base_q']
        inside = oracle.query(np.r_[base, tucked])
        collision = evaluate_collision_status(scene, base, arm_q=tucked,
                                              gripper_q=scene.gripper_open_positions)
        row = dict(candidate, dock_room_residual_m=inside['residual_m'],
                   dock_collision_distance_m=float(collision.minimum_signed_distance))
        if policy_from_config(config) != 'geometry_only' and not inside['satisfied']:
            row.update(status='rejected', reason='whole_robot_goal_outside_room')
        elif not collision.collision_free:
            row.update(status='rejected', reason='docking_collision')
        else:
            row.update(status='eligible', reason='goal_only_not_yet_connected')
        rows.append(row)
    return rows


def make_candidate_config(config, candidate):
    result = copy.deepcopy(config)
    result['pick_task']['candidates'] = [{'name': candidate['id'], 'base_q': candidate['base_q']}]
    result['pick_task']['candidate_source'] = 'deterministic_workspace_lattice_then_full_robot_screen'
    result['pick_task']['gripper_rpy'] = candidate['grasp_rpy']
    return result


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def accepted_result(result, require_dynamics):
    """A completed report must explicitly confirm all requested validations."""
    return bool(result and result.get('satisfied') and result.get('reference_audit', {}).get('policy_satisfied', result.get('reference_audit', {}).get('satisfied'))
                and (not require_dynamics or
                     (result.get('dynamics_executed') and result.get('physics', {}).get('policy_satisfied', result.get('physics', {}).get('satisfied')))))


def write_review(output, summary):
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(r.get(key, "")))}</td>' for key in
                   ['id', 'base_q', 'status', 'reason', 'failure_stage']) + '</tr>'
                   for r in summary['candidates'])
    chosen = summary.get('selected_candidate')
    viewer = ''
    metrics = ''
    result = summary.get('selected_result', {})
    method = result.get('reach', {}).get('region_method', summary.get('operation_region_method', 'reference_corridor'))
    region_description = ('Drake IRIS-NP2 区域，按本方法的语义规则构造域；独立碰撞抽样必须通过'
                          if method == 'iris_np2' else '引导轨迹周围的凸盒，非 IRIS')
    if result:
        physics = result.get('physics', {})
        metrics = '<h2>实际执行与独立审计</h2><pre>' + html.escape(json.dumps({
            'contact_execution': physics,
            'convex_region_holdout_NOT_a_certificate': result.get('region_holdout_audits'),
            'reference_audit_scope': result.get('reference_audit', {}).get('scope'),
        }, ensure_ascii=False, indent=2)) + '</pre>'
    if chosen and (output / chosen / 'scene.html').exists():
        viewer = f'<p><a href="{chosen}/scene.html">打开完整 Drake 接触动力学动画</a></p><iframe src="{chosen}/scene.html"></iframe>'
    text = f'''<!doctype html><meta charset="utf-8"><title>Semantic refinement</title>
<style>body{{font:16px system-ui;margin:30px;color:#18354a}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd;padding:8px}}iframe{{width:100%;height:650px;border:0}}code{{white-space:pre-wrap}}</style>
<h1>两房间抓取：{summary.get('semantic_policy', 'full_motion')}</h1>
<p>场景固定；搜索底盘停靠点和抓取姿态。连接运动与操作均调用已有 Drake GCS。</p>
<p>状态：<b>{html.escape(summary['status'])}</b>；候选数：{len(summary['candidates'])}；完整规划尝试：{summary['attempt_count']}</p>
<p>统一任务语义通过：{result.get('semantic_task_satisfied', result.get('reference_audit', {}).get('satisfied', '未完成'))}</p>
<p>本方法验收规则下，参考轨迹通过：{summary.get('reference_validated', False)}；实际接触执行通过：{summary.get('physical_execution_validated', False)}</p>
<p>底盘朝向固定、导航时手臂收拢、操作时底盘固定。当前手臂区域：{region_description}；碰撞采用数值审计，不是连续时间证明。</p>
{viewer}{metrics}<h2>全部候选与失败记录</h2><table><tr><th>ID</th><th>底盘配置</th><th>状态</th><th>原因</th><th>失败阶段</th></tr>{rows}</table>
<p><a href="summary.json">完整结果</a> · <a href="protocol.json">冻结实验协议</a></p>'''
    (output / 'index.html').write_text(text, encoding='utf-8')


def run(config_path, output, *, dynamics=True, visualize=True):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = load_scene_config(config_path)
    spec = config['semantic_refinement']
    if int(spec['max_planning_attempts']) < 1 or float(spec['candidate_timeout_s']) <= 0:
        raise ValueError('Positive search budgets required')
    candidates = generate_candidates(config)
    (output / 'scene_config_snapshot.json').write_text(json.dumps(config, indent=2))
    protocol = {'semantic_policy': policy_from_config(config), 'candidate_list': candidates, 'search': spec, 'require_dynamics': dynamics,
                'world_sha256': hashlib.sha256(json.dumps({k: config[k] for k in
                    ['world', 'objects', 'robot', 'arm', 'gripper', 'green_room']}, sort_keys=True).encode()).hexdigest(),
                'scope': 'Finite lattice, fixed navigation yaw and tucked arm; candidate ranking is not global task optimization',
                'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                    [Path(__file__), Path(__file__).with_name('green_room_gcs.py'),
                     Path(__file__).with_name('whole_robot_room.py'), Path(__file__).with_name('green_room_iris.py')]}}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    rows = screen_candidates(config, candidates)
    summary = {'status': 'searching', 'semantic_policy': policy_from_config(config), 'candidates': rows, 'attempt_count': 0,
               'selected_candidate': None, 'reference_validated': False,
               'physical_execution_validated': False, 'scene_changed_during_search': False,
               'operation_region_method': config['green_room'].get('region_method', 'reference_corridor')}
    def save():
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        write_review(output, summary)
    save()
    began = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    for row in rows:
        if row['status'] != 'eligible':
            continue
        if summary['attempt_count'] >= int(spec['max_planning_attempts']):
            break
        summary['attempt_count'] += 1
        row.update(status='planning', reason='checking_bridge_and_operation')
        candidate_config = make_candidate_config(config, row)
        path = output / f"{row['id']}.json"
        path.write_text(json.dumps(candidate_config, indent=2))
        destination = output / row['id']
        command = [sys.executable, '-m', 'src.green_room_gcs', '--config', str(path), '--output', str(destination)]
        if not dynamics:
            command.append('--no-dynamics')
        if not visualize:
            command.append('--no-visualize')
        save()
        print(f"Trying {row['id']}: base={row['base_q']} grasp={row['grasp_rpy']}", flush=True)
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', LCM_DEFAULT_URL='memq://')
        started = time.perf_counter()
        with (output / f"{row['id']}.log").open('w') as log:
            try:
                process = subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         timeout=float(spec['candidate_timeout_s']), check=False)
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = None
        row['elapsed_s'] = time.perf_counter() - started
        result = read_json(destination / 'result.json', {})
        row['reference_validated'] = bool(result.get('reference_audit', {}).get('policy_satisfied', result.get('reference_audit', {}).get('satisfied')))
        row['region_holdout_audits'] = result.get('region_holdout_audits')
        row['exit_code'] = code
        if code == 0 and accepted_result(result, dynamics):
            row.update(status='accepted', reason='all_requested_checks_passed')
            summary.update(status='physical_execution_passed' if dynamics else 'reference_only_passed',
                           selected_candidate=row['id'], reference_validated=True,
                           physical_execution_validated=bool(dynamics), selected_result=result)
            break
        progress = read_json(destination / 'progress.json', {})
        row.update(status='rejected', failure_stage=progress.get('stage', 'unknown'),
                   reason='candidate_time_budget_exhausted' if code is None else 'planning_or_validation_failed')
        row['log_tail'] = (output / f"{row['id']}.log").read_text()[-4000:]
        save()
    if summary['selected_candidate'] is None:
        summary['status'] = 'finite_search_exhausted_not_infeasibility_proof'
    summary['elapsed_s'] = time.perf_counter() - began
    save()
    print(json.dumps({k: summary[k] for k in ['status', 'selected_candidate', 'attempt_count', 'elapsed_s']}, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/scene_green_room_iris.yaml')
    parser.add_argument('--output', required=True)
    parser.add_argument('--no-dynamics', action='store_true')
    parser.add_argument('--no-visualize', action='store_true')
    args = parser.parse_args()
    result = run(args.config, args.output, dynamics=not args.no_dynamics, visualize=not args.no_visualize)
    if result['selected_candidate'] is None:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
