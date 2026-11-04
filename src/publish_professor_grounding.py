"""Compact evidence and local review from completed, saved experiments."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

from .scene_spec import PROJECT_ROOT


def read(path):
    return json.loads(path.read_text())


def publish(cup_roots, sequence_root, output, evidence, figure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cups, links = [], []
    for root in map(Path, cup_roots):
        search = read(root/'result.json')
        if not search['success']:
            raise ValueError(f'Cup search incomplete: {root}')
        folder = root/'supplied_plan/operation_first'/search['selected']
        result = read(folder/'candidate_result.json')
        if not all(result[stage]['accepted'] for stage in ('bridge', 'operation', 'physical')):
            raise ValueError('Cannot publish unverified cup success')
        selected = next(c for c in read(root/'protocol.json')['candidates'] if c['id'] == search['selected'])
        physics = read(folder/'execution/execution_grounded_candidate/physics.json')
        cups.append({'root': str(root), 'search': search, 'candidate': selected,
            'sampled_grasp_pose': result['operation']['sampled_grasp_pose'],
            'reference_plan_audit': result['assembled_reference_audit']['semantic_plan_audit'],
            'physical': result['physical'],
            'cup_attachment': physics['cup_attachment'],
            'cup_pose_resets_after_initialization': physics['cup_pose_resets_after_initialization'],
            'source_manifest_sha256': hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest()})
        movie = folder/'execution/execution_grounded_candidate/scene.html'
        links.append(f'<li><a href="{html.escape(os.path.relpath(movie.resolve(), output))}">{html.escape(root.name)}：实际接触仿真动画</a></li>')
    sequence_root = Path(sequence_root)
    seq = read(sequence_root/'result.json')
    if not seq['satisfied']:
        raise ValueError('Sequence grounding incomplete')
    compact = {k: v for k, v in seq.items() if k != 'reports'}
    compact['pieces_and_bridge'] = {name: {'semantic_audit': report['semantic_audit'],
        'geometry_audit': report['geometry_audit'],
        'arm_region_count': report['planners'].get('arm', {}).get('region_count')}
        for name, report in seq['reports'].items()}
    data = {'cup_runs': cups, 'sequence': compact,
            'scope': 'Development examples, finite numerical audits; no statistical, continuous-time or arbitrary multi-object success claim'}
    evidence = Path(evidence)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(data, indent=2))
    frames = read(sequence_root/'grounded_reference.json')
    colors = ['#246b85', '#cf6c24', '#34846a']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for phase, color in zip(dict.fromkeys(f['phase'] for f in frames), colors):
        qs = np.array([f['arm_q'] for f in frames if f['phase'] == phase])
        axes[0].plot(qs[:, 0], qs[:, 1], color=color, label=phase.replace('_', ' '), lw=2)
    pieces = read(sequence_root/'protocol.json')['spec']['pieces']
    a, b = np.array(pieces[0]['goal_arm_q']), np.array(pieces[1]['start_arm_q'])
    axes[0].scatter([a[0], b[0]], [a[1], b[1]], color='black', s=20, zorder=4)
    axes[0].set(xlabel='Joint 1 (rad)', ylabel='Joint 2 (rad)', title='Two pieces joined by a planned GCS bridge')
    axes[0].legend(fontsize=8)
    for i, cup in enumerate(cups):
        audit = cup['physical']['semantic_plan_audit']
        values = [p['predicates']['inside']['minimum_residual']*1000 for p in audit['phases']]
        axes[1].plot(range(5), values, marker='o', label=f"Base yaw {cup['candidate']['base_q'][2]:.2f} rad")
    axes[1].axhline(0, color='black', ls='--', lw=1)
    axes[1].set(xticks=range(5), xticklabels=['arrive', 'reach', 'close', 'lift', 'hold'],
        ylabel='Minimum sampled room residual (mm)', title='Measured execution, after 20 mm margin')
    axes[1].legend(fontsize=8)
    figure = Path(figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    (output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>教授思路：实现验证</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;color:#203c49}}img{{width:100%}}pre{{white-space:pre-wrap}}</style>
<h1>语义计划 → 目标选择 → 连续轨迹</h1>
<p>两次自由杯抓取/抬升/保持执行通过；另有两段断开的手臂轨迹，经官方 IRIS/GCS motion bridge 连成连续参考轨迹。</p>
<p>解析配置检查与采样轨迹检查分开记录。所有结果来自开发场景；没有连续时间安全或任意多对象任务成功的声明。</p>
<img src="{html.escape(os.path.relpath(figure.resolve(), output))}"><ul>{''.join(links)}</ul>
<p><a href="{html.escape(os.path.relpath(evidence.resolve(), output))}">精简原始证据</a></p>
<pre>{html.escape(json.dumps({'cup_runs': [{'candidate': c['candidate'], 'success': c['search']['success'], 'physical_plan_violations': c['physical']['semantic_plan_audit']['violations']} for c in cups], 'sequence': {k: compact[k] for k in ['satisfied', 'piece_gap_before_bridge_rad', 'geometry_audit']}}, ensure_ascii=False, indent=2))}</pre>''', encoding='utf-8')
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cup-roots', nargs='+', type=Path, required=True)
    p.add_argument('--sequence-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--figure', type=Path, required=True)
    args = p.parse_args()
    publish(args.cup_roots, args.sequence_root, args.output, args.evidence, args.figure)
