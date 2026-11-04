"""Publish saved operation experiments; never manufacture planner successes."""
import argparse
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def publish(repair, semantics, output):
    repair, semantics, root = (Path(p).resolve() for p in (repair, semantics, output))
    root.mkdir(parents=True, exist_ok=False)
    methods = json.loads((repair/'comparison.json').read_text())
    policies = json.loads((semantics/'comparison.json').read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), layout='constrained')
    for row, color in zip(policies, ['#be4b36', '#167a70']):
        path_file = semantics/row['policy']/'path.json'
        trace_file = semantics/row['policy']/'trace.json'
        if not row['planner_succeeded'] or not path_file.exists() or not trace_file.exists():
            continue
        trace = json.loads(trace_file.read_text())
        q = np.array([v['arm_q'] for v in trace])
        progress = np.linspace(0, 1, len(trace))
        axes[0].plot(progress, [100*v['room_residual_m'] for v in trace],
                     label=row['policy'], color=color, lw=2)
        axes[1].plot(q[:, 0], q[:, 1], label=row['policy'], color=color, lw=2)
        axes[1].scatter(q[[0, -1], 0], q[[0, -1], 1], s=25, color=color)
    axes[0].axhline(0, color='#555555', ls='--', lw=1)
    axes[0].set(xlabel='Normalized reference sample index (not time)',
                ylabel='Whole-robot room residual (cm)', title='Room containment along GCS paths')
    axes[1].set(xlabel='Joint 1 (rad)', ylabel='Joint 2 (rad)',
                title='Joint projection; other five joints also planned')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(root/'semantic_comparison.png', dpi=180)
    fig.savefig(root/'semantic_comparison.svg')
    plt.close(fig)
    evidence = {'operation_runs': [], 'semantic_runs': [],
        'limitations': ['One development screen layout and one arm-only development probe.',
                        'No statistical speed/success advantage or continuous-time safety claim.',
                        'Arm-only semantic probe has no grasp/contact execution.']}
    physical_rows = []
    for row in methods:
        selected = row.get('selected')
        detail = None
        if selected:
            folder = repair/row['task']/'operation_first'/selected
            detail = json.loads((folder/'candidate_result.json').read_text())
            animation = folder/'execution/execution_grounded_candidate/scene.html'
            link = html.escape(os.path.relpath(animation, root), quote=True)
            viewer = f'<a href="{link}">实际接触动画</a>' if animation.exists() else '无动画'
            physics_file = animation.with_name('physics.json')
            physics = json.loads(physics_file.read_text()) if physics_file.exists() else {}
            refinement = detail['operation'].get('refinement', {})
            evidence['operation_runs'].append({
                'method': row['repair_method'], 'selected': selected, 'success': row['success'],
                'attempts': row['attempts'], 'wall_seconds': row['total_wall_seconds'],
                'base_q': detail['base_q'], 'physics': detail['physical'],
                'cup_attachment': physics.get('cup_attachment'),
                'cup_pose_resets_after_initialization': physics.get('cup_pose_resets_after_initialization'),
                'refinement': {stage: {'repaired_regions': value['repaired_regions'],
                    'rounds': [{'region_count': r['region_count'],
                               'discovery_samples': r['discovery']['samples'],
                               'discovery_collisions': r['discovery']['collisions'],
                               'counterexamples': r['discovery']['counterexamples'],
                               'validation_samples': r.get('validation', {}).get('samples'),
                               'validation_collisions': r.get('validation', {}).get('collisions')}
                              for r in value['rounds']]}
                    for stage, value in refinement.items() if value is not None}})
        else:
            viewer = '无通过验收的执行'
            evidence['operation_runs'].append(dict(row))
        physical_rows.append(f'<tr><td>{html.escape(row["repair_method"])}</td>'
            f'<td>{len(row["attempts"])}</td><td>{html.escape(str(selected))}</td>'
            f'<td>{row["success"]}</td><td>{row["total_wall_seconds"]:.2f}</td><td>{viewer}</td></tr>')
    semantic_rows = []
    for row in policies:
        audit = row.get('audit', {})
        evidence['semantic_runs'].append({k: row.get(k) for k in
            ['policy', 'planner_succeeded', 'method_satisfied', 'full_task_semantics_satisfied',
             'physical_execution_validated', 'audit', 'region_count', 'wall_seconds']})
        value = audit.get('minimum_room_residual_m')
        residual = f'{100*value:.3f}' if value is not None else '无轨迹'
        semantic_rows.append(f'<tr><td>{html.escape(row["policy"])}</td>'
            f'<td>{audit.get("goal_semantics_satisfied")}</td><td>{audit.get("geometry_satisfied")}</td>'
            f'<td>{residual}</td><td>{row["full_task_semantics_satisfied"]}</td></tr>')
    (root/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2))
    (root/'index.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>整机语义与局部区域修复</title>
<style>body{font:17px system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#223646;line-height:1.7}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #cdd6da}img{width:100%;margin:20px 0}a{color:#126d74}</style>
<h1>整机语义与操作区域局部修复</h1>
<p>2026-10-19 · 当前结果来自保存的实际规划与执行文件。任务为入室、抓杯、抬升和保持。</p>
<h2>区域审计发现碰撞后，先修复当前候选</h2>
<p>仅替换失败区域：按反例位置细分引导区间、缩小 padding、重新调用官方 IRIS，再以新种子审计。
成功后以真实相交关系建立 GCS 图。底盘目标、抓取目标、场景和安全余量保持相同。</p>
<table><tr><th>方法</th><th>尝试候选</th><th>最终候选</th><th>物理通过</th><th>搜索秒数</th><th>查看</th></tr>'''
        + ''.join(physical_rows) + '''</table>
<p>两组均使用相同候选、预算、审计与接触验收。单次开发场景结果，耗时不构成统计性能结论。</p>
<h2>端点合法仍可能在途中违反房间语义</h2>
<p>独立的固定底盘、空手机械臂试验。两组使用同一引导生成器和 IRIS/GCS，只改变语义作用范围。
以下均为参考轨迹检查，没有抓取或动力学执行。</p>
<table><tr><th>语义策略</th><th>端点入室</th><th>无碰撞</th><th>最小房间余量 cm</th><th>全过程语义通过</th></tr>'''
        + ''.join(semantic_rows) + '''</table><img alt="两组 GCS 参考路径的房间余量与关节投影" src="semantic_comparison.png">
<p>图中余量已扣除房间配置的 2 cm margin。采样索引归一化仅用于观察过程，不表示相同运动时长。</p>
<p>采用模型几何与有限数值采样，不构成连续时间安全证书。仍为给定两阶段任务，未实现通用缺失动作发现。</p>
<p><a href="evidence.json">结果摘要与碰撞反例</a></p></html>''', encoding='utf-8')
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repair', required=True)
    parser.add_argument('--semantics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    publish(args.repair, args.semantics, args.output)
