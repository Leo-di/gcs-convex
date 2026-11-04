"""Publish compact evidence from completed multi-object/dynamic/learned runs."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
from .scene_spec import PROJECT_ROOT


def read(path):
    return json.loads(Path(path).read_text())


def publish(multiobject, dynamic, negative, output, evidence, figure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .build_scene import build_scene
    from .dynamic_semantics import MovingKeepClear
    from .semantic_plan import configuration
    from .learned_semantics import predict

    multiobject, dynamic, negative = map(lambda p: Path(p).resolve(), (multiobject, dynamic, negative))
    output, evidence, figure = map(lambda p: Path(p).resolve(), (output, evidence, figure))
    checks = read(multiobject/'execution_checks.json')
    moving = read(dynamic/'result.json')
    control_file = negative/'execution_checks.json'
    if not control_file.exists():
        control_file = negative/'result.json'
    control = read(control_file)
    if not checks['satisfied'] or not moving['satisfied'] or control['satisfied']:
        raise ValueError('Expected two successful executions and a rejected negative control')
    if any(c['lifted_with_bilateral_contact'] for c in control['objects'].values()):
        raise ValueError('No-grip control unexpectedly lifted an object')
    protocol = read(dynamic/'protocol.json')
    schedule = read(dynamic/'schedule.json')
    sources = {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((PROJECT_ROOT/'src').glob('*.py'))}
    data = {'multiobject': checks, 'dynamic': moving,
        'wait_attempts': [{k: row[k] for k in ('delay_s', 'satisfied', 'reason') if k in row}
                          | {'samples': row.get('audit', {}).get('samples'),
                             'violations': row.get('audit', {}).get('violations')}
                          for row in schedule['attempts']],
        'no_grip_control': control,
        'raw_sources': [str(p.resolve()) for p in (multiobject, dynamic, negative)],
        'frozen_dynamic_source_reference_sha256': protocol['source_reference_sha256'],
        'artifacts_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (multiobject/'physics.json', dynamic/'physics.json', dynamic/'learned_room.json', control_file)},
        'supplementary_source_sha256': sources,
        'scope': 'Development examples; sampled validation; prescribed customer occupancy; geometric learned labels with mandatory analytical guard'}
    for path in (output, evidence.parent, figure.parent):
        path.mkdir(parents=True, exist_ok=True)
    evidence.write_text(json.dumps(data, indent=2))

    multi_physics, dynamic_physics = [read(p/'physics.json') for p in (multiobject, dynamic)]
    model = read(dynamic/'learned_room.json')
    dataset = np.load(dynamic/'learned_room.data.npz')
    actual, prediction = dataset['test_residual'], predict(model, dataset['test_q'])
    # These are fresh geometry queries of saved measured states, not a new
    # dynamics run. Execution's separate 10 ms monitor remains in the evidence.
    config = read(dynamic/'config.json')
    scene = build_scene(config=config, visualize=False)
    oracle = MovingKeepClear(scene, config['grounding_semantic_plan']['predicates']['customer'])
    trace = dynamic_physics['trace']
    residual = [oracle.evaluate(configuration(f), f['time_s'])['residual'] for f in trace]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    trace_multi = multi_physics['trace']
    for name in checks['objects']:
        z0 = trace_multi[0]['objects'][name]['pose'][2][3]
        axes[0].plot([r['time_s'] for r in trace_multi],
            [100*(r['objects'][name]['pose'][2][3]-z0) for r in trace_multi], label=name, lw=1.8)
    axes[0].axhline(10, color='gray', ls='--', lw=1)
    axes[0].set(xlabel='Measured time (s)', ylabel='Cup height above start (cm)',
                title='Two objects, one continuous contact simulation')
    axes[0].legend()
    axes[1].plot([r['time_s'] for r in trace], np.array(residual)*100, color='#246b85')
    axes[1].axhline(0, color='black', ls='--', lw=1)
    axes[1].axvspan(0, moving['selected_delay_s'], color='#edc17e', alpha=.3, label='Selected 25 s wait')
    axes[1].set(xlabel='Measured time (s)', ylabel='Customer separation after margin (cm)',
                title='Prescribed occupancy; 10 ms online checks passed')
    axes[1].legend(fontsize=8)
    axes[2].scatter(actual, prediction, s=7, alpha=.45, color='#34846a')
    wrong = (actual < 0) & (prediction > model['abstention_margin_m'])
    axes[2].scatter(actual[wrong], prediction[wrong], s=65, marker='x', color='#c54132',
                    label='Confident false acceptance')
    limits = [min(actual.min(), prediction.min()), max(actual.max(), prediction.max())]
    axes[2].plot(limits, limits, color='gray', ls='--', lw=1)
    axes[2].axhline(0, color='black', lw=.6)
    axes[2].axvline(0, color='black', lw=.6)
    axes[2].set(xlabel='Analytical room residual (m)', ylabel='Learned prediction (m)',
                title='600 held-out samples; analytical guard required')
    axes[2].legend(fontsize=8)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    def link(path):
        return html.escape(os.path.relpath(Path(path).resolve(), output))
    (output/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>教授思路：多对象、动态占用与学习评估</title>
<style>body{{font:16px system-ui;max-width:1280px;margin:32px auto;padding:0 20px;color:#203c49;line-height:1.7}}img{{width:100%}}a{{color:#246b85}}pre{{white-space:pre-wrap}}</style>
<h1>多对象连续任务、动态顾客占用与学习评估</h1>
<p>两个自由杯子在同一个接触仿真中依次抓取、抬升、放回；夹爪始终张开的负对照无法完成任一杯子的双指抬升。另一个动态占用实验选择等待25秒后出发，8250次在线检查无违规。</p>
<p>橙色方柱表示预先给定的顾客占用范围。它是随时间移动的几何语义约束。机器人和杯子的运动由力驱动仿真产生。</p>
<img src="{link(figure)}" alt="双杯高度、动态占用间距与学习评估测试集">
<ul><li><a href="{link(multiobject/'scene.html')}">双杯连续接触执行动画</a></li>
<li><a href="{link(dynamic/'scene.html')}">等待与移动顾客占用动画</a></li>
<li><a href="{link(evidence)}">精简原始证据及来源哈希</a></li></ul>
<p>学习器使用几何标签训练，独立测试集符号准确率97.33%，仍有1次高置信度误接受；最终验收始终保留解析几何检查。这些开发案例不构成连续时间证明、任意长任务成功率或人类偏好学习结果。</p>
</html>''', encoding='utf-8')
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('multiobject', 'dynamic', 'negative', 'output', 'evidence', 'figure'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    publish(**vars(args))
