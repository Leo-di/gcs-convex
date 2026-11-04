"""Publish measured stateful transport traces and compact provenance."""
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


def publish(source, greedy, output, evidence, figure, equivalence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source, greedy, output, evidence, figure = [Path(p).resolve() for p in
                                               (source, greedy, output, evidence, figure)]
    result, search, physics = [read(source/name) for name in ('result.json', 'search.json', 'physics.json')]
    baseline = read(greedy/'search.json')
    if not result['satisfied'] or not search['satisfied'] or baseline['satisfied']:
        raise ValueError('Expected completed backtracking execution and rejected greedy search')
    if not any(e['event'] == 'rollback_after_downstream_failure' for e in search['events']):
        raise ValueError('Missing actual downstream rollback')
    for p in (output, evidence.parent, figure.parent):
        p.mkdir(parents=True, exist_ok=True)
    branches = {str(r['index']): read(source/f"branch_{r['index']:03d}"/'result.json')
                for r in search['branches'] if r['local_satisfied']}
    paths = [source/n for n in ('protocol.json', 'search.json', 'result.json', 'physics.json', 'reference.json')]
    paths += [greedy/n for n in ('protocol.json', 'search.json', 'result.json')]
    equivalence = Path(equivalence).resolve()
    if not read(equivalence)['satisfied']:
        raise ValueError('Overlap graph equivalence was rejected')
    paths.append(equivalence)
    data = {'execution': result, 'search': {k: search[k] for k in ('strategy', 'expansions', 'branches', 'events', 'wall_seconds', 'termination')},
        'greedy': {k: baseline[k] for k in ('strategy', 'satisfied', 'expansions', 'branches', 'events', 'wall_seconds')},
        'validated_branches': {k: {name: v[name] for name in ('input_world_sha256', 'output_world_sha256', 'reference_audit', 'payload_bridge')}
                               for k, v in branches.items()},
        'duration_s': physics['duration_s'], 'overlap_graph_equivalence': read(equivalence),
        'frozen_source_sha256': {'backtrack': read(source/'protocol.json')['source_sha256'],
                                 'greedy': read(greedy/'protocol.json')['source_sha256']},
        'supplementary_source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted((PROJECT_ROOT/'src').glob('*.py'))},
        'artifacts_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'scope': 'One development layout; equal-height tables; fixed arm/yaw payload translation; planning rollback before execution; finite checks; ideal planar base'}
    evidence.write_text(json.dumps(data, indent=2))
    trace = physics['trace']
    times = [r['time_s'] for r in trace]
    colors = {'cup': '#247f9a', 'cup_b': '#c16a38'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), layout='constrained')
    for name, color in colors.items():
        positions = np.array([r['objects'][name]['pose'] for r in trace])[:, :3, 3]
        axes[0].plot(positions[:, 0], positions[:, 1], color=color, label=name)
        axes[0].scatter(*positions[0, :2], marker='o', color=color)
        axes[0].scatter(*positions[-1, :2], marker='x', s=70, color=color)
        axes[1].plot(times, 100*(positions[:, 2]-positions[0, 2]), color=color, label=name)
        carry = [r for r in trace if r['mode'] == 'carry' and r['object_id'] == name]
        axes[2].plot([r['time_s'] for r in carry],
            [min(r['objects'][name]['contacts_N'][f] for f in ('left_finger', 'right_finger')) for r in carry],
            color=color, label=name+' min finger')
        axes[2].plot([r['time_s'] for r in carry],
            [max(r['objects'][name]['contacts_N'][f] for f in ('table', 'dropoff_table')) for r in carry],
            color=color, ls='--', label=name+' table support')
    axes[0].set(xlabel='Measured cup x (m)', ylabel='Measured cup y (m)', title='Cross-table transport; start o / finish x')
    axes[1].set(xlabel='Simulation time (s)', ylabel='Height above start (cm)', title='Two free objects, one continuous simulation')
    axes[2].set(xlabel='Simulation time (s)', ylabel='Contact force (N)', title='Carry: bilateral contact, no table support')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    def link(p):
        return html.escape(os.path.relpath(p, output))
    samples = [[round(r['time_s'], 2), r['phase'], r['base_q'],
                *[np.array(r['objects'][k]['pose'])[:3, 3].tolist() for k in colors]] for r in trace[::4]]
    samples.append([trace[-1]['time_s'], trace[-1]['phase'], trace[-1]['base_q'],
                    *[np.array(trace[-1]['objects'][k]['pose'])[:3, 3].tolist() for k in colors]])
    rows = ''.join(f"<tr><td>{r['index']}</td><td>{html.escape(r['action'])}</td><td>{r['placement'][0]:.2f} m</td><td>{'局部通过' if r['local_satisfied'] else '前一杯占用目标，拒绝'}</td></tr>" for r in search['branches'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>双杯跨桌搬运与规划回溯</title>
<style>body{font:16px system-ui;max-width:1250px;margin:28px auto;padding:0 20px;color:#203c49;line-height:1.7}img{width:100%}canvas{width:100%;background:#f6f8fa;border:1px solid #ddd}a{color:#247f9a}table{border-collapse:collapse}td,th{padding:6px 20px;border-bottom:1px solid #ddd}input{width:75%}</style>
<h1>双杯跨桌搬运与规划回溯</h1><p>SUMMARY</p>
<p>第一杯最初选择了第二杯需要的位置。第二步被真实几何占用检查拒绝，规划恢复先前世界状态，改选第一杯放置位置后完成整段计划。贪心基线在相同冲突处失败。回溯在物理执行前完成。</p>
<table><tr><th>分支</th><th>动作</th><th>放置x</th><th>结果</th></tr>ROWS</table>
<h2>实测轨迹俯视回放</h2><p>蓝色为cup，橙色为cup_b，灰色为底盘。仅显示实测位置与高度；完整机器人姿态请打开三维动画。</p>
<canvas id="view" width="1100" height="390"></canvas><button id="play">播放 / 暂停</button> <input id="time" type="range" min="0" value="0" step="1"><p id="label"></p>
<img src="FIGURE" alt="实测位置、高度与携带接触力"><p><a href="SCENE">完整三维接触执行动画（239 MB）</a> · <a href="EVIDENCE">精简证据及来源哈希</a></p>
<p>CHECKS 当前只验证一个开发布局、同高度桌面和固定手臂/朝向的携带平移；不是任意场景成功率或连续时间安全证明。</p>
<script>const data=DATA;const c=document.getElementById('view'),x=c.getContext('2d'),s=document.getElementById('time');s.max=data.length-1;let playing=false;const px=v=>80+(v+.15)*430,py=v=>345-v*270;
function draw(){let r=data[+s.value];x.clearRect(0,0,c.width,c.height);for(const t of [[.33,.3,.7,.8],[1.33,.3,.9,.8]]){x.fillStyle='#e2e7eb';x.fillRect(px(t[0]),py(t[1]+t[3]),t[2]*430,t[3]*270)}x.fillStyle='#71808b';x.beginPath();x.arc(px(r[2][0]),py(r[2][1]),23,0,7);x.fill();for(let i=3;i<5;i++){x.fillStyle=i===3?'#247f9a':'#c16a38';x.beginPath();x.arc(px(r[i][0]),py(r[i][1]),13,0,7);x.fill();x.fillText('z='+r[i][2].toFixed(3)+' m',px(r[i][0])-30,py(r[i][1])-20)}document.getElementById('label').textContent=r[0].toFixed(2)+' s · '+r[1]}
s.oninput=draw;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing){s.value=(+s.value+1)%data.length;draw()}},40);draw();</script></html>'''
    metrics = list(result['actions'].values())
    summary = (f"同一次{physics['duration_s']:.2f}秒接触仿真完成两个杯子的抓取、携带、放置。"
               f"移动距离{metrics[0]['object_displacement_m']*100:.2f}厘米与{metrics[1]['object_displacement_m']*100:.2f}厘米；"
               f"最终位置误差{metrics[0]['final_position_error_m']*1000:.2f}毫米与{metrics[1]['final_position_error_m']*1000:.2f}毫米。"
               '物体初始化一次，此后自由运动。')
    checks = (f"执行期间{result['online_samples']}次在线房间检查、{result['semantic_audit']['samples']}个实测语义插值样本均通过；"
              f"非预期接触{result['unexpected_contacts']}。")
    for a, b in [('SUMMARY', summary), ('CHECKS', checks), ('ROWS', rows), ('FIGURE', link(figure)), ('SCENE', link(source/'scene.html')),
                 ('EVIDENCE', link(evidence)), ('DATA', json.dumps(samples, separators=(',', ':')))]:
        page = page.replace(a, b)
    (output/'index.html').write_text(page)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'greedy', 'output', 'evidence', 'figure', 'equivalence'):
        parser.add_argument('--'+name, type=Path, required=True)
    publish(**vars(parser.parse_args()))
