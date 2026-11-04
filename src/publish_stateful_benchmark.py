"""Publish every cell of a completed frozen transport comparison."""
import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path

import numpy as np
from .benchmark_stateful_transport import read, write, file_hash, summarize, verify
from .scene_spec import PROJECT_ROOT


def publish(source, output, evidence, figure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    source, output, evidence, figure = [Path(p).resolve() for p in (source, output, evidence, figure)]
    protocol = verify(source)
    result = read(source/'summary.json')
    actual_records = [read(source/c['id']/m/'record.json') for c in protocol['cells'] for m in protocol['methods']]
    actual_records.sort(key=lambda r: (r['cell'], r['method']))
    expected_records = sorted(result['records'], key=lambda r: (r['cell'], r['method']))
    if actual_records != expected_records or not result['complete']:
        raise ValueError('Incomplete or changed benchmark records')
    if any(r['end_to_end_success'] != (r['status'] == 'success')
           or (r['end_to_end_success'] and not r['planning_success']) for r in actual_records):
        raise ValueError('Outcome flags disagree with acceptance status')
    if result['cohorts'] != summarize(actual_records, protocol['cells'])['cohorts']:
        raise ValueError('Summary disagrees with every frozen cell')
    if result['protocol_sha256'] != file_hash(source/'protocol.json'):
        raise ValueError('Protocol changed after experiment')
    for row in actual_records:
        directory = source/row['cell']/row['method']
        if row['planning_success']:
            search = read(directory/'search.json')
            audit = read(directory/'assembled_semantic_audit.json')
            if not search['satisfied'] or not audit['satisfied']:
                raise ValueError('Planning flag lacks complete audited reference')
        if row['end_to_end_success']:
            checks, physics = [read(directory/name) for name in ('result.json', 'physics.json')]
            if (not checks['satisfied'] or not checks['semantic_audit']['satisfied']
                    or checks['unexpected_contacts'] != 0 or not checks['effort_limits']
                    or checks['object_pose_resets_after_initialization'] != 0
                    or checks['cup_attachment'] or not physics['trace']):
                raise ValueError('Success lacks matching physical and semantic evidence')
    for p in (output, evidence.parent, figure.parent):
        p.mkdir(parents=True, exist_ok=True)
    files = [source/'protocol.json', source/'summary.json']
    for record in actual_records:
        directory = source/record['cell']/record['method']
        files.extend(p for p in directory.glob('*.json') if p.name in
                     ('record.json', 'result.json', 'search.json', 'reference.json', 'physics.json', 'assembled_semantic_audit.json'))
    data = {'protocol': protocol, 'summary': result,
        'artifact_sha256': {str(p.relative_to(PROJECT_ROOT)): file_hash(p) for p in files},
        'publication_source_sha256': {str(Path(__file__).resolve().relative_to(PROJECT_ROOT)): file_hash(__file__)},
        'scope': 'All preregistered cells retained; one run per method/cell; correlated candidate orders; control separate; no independent-sample significance or timing claim'}
    write(evidence, data)
    cells, methods = protocol['cells'], protocol['methods']
    lookup = {(r['cell'], r['method']): r for r in actual_records}
    positives = [c for c in cells if c['cohort'] == 'transport']
    statuses = ['success', 'planning_rejected', 'execution_rejected', 'planning_timeout',
                'execution_timeout', 'planning_error', 'execution_error']
    palette = ['#318572', '#b5bdc7', '#ca714a', '#d6bb62', '#d29444', '#8b65a6', '#a04469']
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), layout='constrained', gridspec_kw={'width_ratios': [1, 1.7, 1.5]})
    for x, method in enumerate(methods):
        rows = [lookup[(c['id'], method)] for c in positives]
        planned = sum(r['planning_success'] for r in rows)
        complete = sum(r['end_to_end_success'] for r in rows)
        axes[0].bar(x-.16, planned, .3, color='#7faaba', label='Audited plan' if x == 0 else None)
        axes[0].bar(x+.16, complete, .3, color='#318572', label='Contact execution' if x == 0 else None)
    axes[0].set(xticks=range(len(methods)), xticklabels=methods, ylim=(0, len(positives)+.6),
                ylabel='Accepted / all 6 predefined cells', title='Transport cohort (control excluded)')
    axes[0].legend(fontsize=8)
    grid = np.array([[statuses.index(lookup[(c['id'], m)]['status']) for m in methods] for c in cells])
    axes[1].imshow(grid, cmap=ListedColormap(palette), vmin=-.5, vmax=len(statuses)-.5, aspect='auto')
    labels = [c['layout'].replace('replication', 'replica').replace('_', ' ')+' / '+c['order'].split('_')[0]
              if c['cohort'] == 'transport' else 'restricted cover (control)' for c in cells]
    axes[1].set(xticks=range(len(methods)), xticklabels=methods, yticks=range(len(cells)),
                yticklabels=labels, title='All frozen cells; no omitted failures')
    for i, c in enumerate(cells):
        for j, m in enumerate(methods):
            axes[1].text(j, i, lookup[(c['id'], m)]['status'].replace('_', '\n'),
                         ha='center', va='center', fontsize=8, color='white' if grid[i, j] == 0 else '#253447')
    for j, method in enumerate(methods):
        axes[2].scatter([], [], color=['#247f9a', '#c16a38'][j], label=method)
        for i, c in enumerate(positives):
            row = lookup[(c['id'], method)]
            for k, action in enumerate(row.get('actions', {}).values()):
                axes[2].scatter(i+(j-.5)*.18+(k-.5)*.045, action['final_position_error_m']*1000,
                                color=['#247f9a', '#c16a38'][j], marker=['o', 'x'][k])
    axes[2].axhline(20, color='gray', ls='--', label='20 mm limit')
    axes[2].set(xticks=range(len(positives)), xticklabels=range(1, len(positives)+1),
                xlabel='Frozen transport cell', ylabel='Final cup position error (mm)',
                title='Measured placements for executed plans')
    axes[2].legend(fontsize=8)
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    def link(path):
        return html.escape(os.path.relpath(path, output))
    rows, options, animation, trace_analysis = [], [], {}, {}
    for i, c in enumerate(cells):
        for method in methods:
            row = lookup[(c['id'], method)]
            directory = source/c['id']/method
            failure = row.get('failure', '')
            if row.get('branch_failures'):
                prefix = failure+': ' if failure else '已回溯的分支拒绝：'
                failure = prefix+row['branch_failures'][-1].get('reason', '')
            layout_label = c['layout'] if c['cohort'] == 'transport' else c['layout']+'（覆盖受限对照）'
            values = [html.escape(str(v)) for v in
                      (i+1, layout_label, c['order'], method, row['status'], row.get('expansions', '—'), failure)]
            detail = directory/('result.json' if (directory/'result.json').exists() else 'record.json')
            values[4] = '<a href="'+link(detail)+'">'+values[4]+'</a>'
            rows.append('<tr>'+''.join('<td>'+v+'</td>' for v in values)+'</tr>')
            if (directory/'physics.json').exists():
                physics = read(directory/'physics.json')
                trace = physics['trace']
                sampled = trace[::4]
                if sampled[-1] is not trace[-1]:
                    sampled.append(trace[-1])
                name = c['id']+'__'+method
                phase_times, previous_time = {}, 0.
                for frame in trace:
                    phase_times[frame['phase']] = phase_times.get(frame['phase'], 0.)+frame['time_s']-previous_time
                    previous_time = frame['time_s']
                trace_analysis[name] = {'duration_s': physics['duration_s'],
                    'phase_duration_s_approximately_50ms_resolution': phase_times,
                    'acceptance': row['status']}
                animation[name] = {'task': read(source/'inputs'/(c['id']+'.json')),
                    'samples': [[round(r['time_s'], 2), r['phase'], r['base_q'],
                        *[np.array(r['objects'][key]['pose'])[:3, 3].tolist() for key in ('cup', 'cup_b')]] for r in sampled]}
                options.append('<option value="'+name+'">'+html.escape(c['id']+' / '+method+' / '+row['status'])+'</option>')
    stats = result['cohorts']['transport']
    intro = '；'.join(f"{m}：完整规划{stats['methods'][m]['planning_successes']}/6，物理执行成功{stats['methods'][m]['end_to_end_successes']}/6" for m in methods)
    counts = Counter(r['status'] for r in actual_records)
    quality = ''
    original = trace_analysis.get('replication__free_first__backtrack')
    shifted = trace_analysis.get('source_right_20mm__free_first__backtrack')
    if original and shifted:
        phase = 'phase_duration_s_approximately_50ms_resolution'
        old, new = original[phase].get('move_second_cup.reach'), shifted[phase].get('move_second_cup.reach')
        if old is not None and new is not None:
            quality = (f'仍需改进：源杯仅向右移动20毫米，第二杯的实测伸臂阶段从约{old:.2f}秒增加到{new:.2f}秒，'
                       f'总仿真时长从{original["duration_s"]:.2f}秒增加到{shifted["duration_s"]:.2f}秒。'
                       '通过任务验收并不代表当前手臂路径效率已经足够。')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>冻结双杯跨桌搬运对照</title>
<style>body{font:16px system-ui;max-width:1400px;margin:28px auto;padding:0 20px;color:#203c49;line-height:1.7}img,canvas{width:100%}canvas{background:#f7f9fa}table{border-collapse:collapse;font-size:13px}td,th{padding:7px;border-bottom:1px solid #ddd}td:last-child{max-width:380px;overflow-wrap:anywhere}input{width:75%}a{color:#247f9a}</style>
<h1>冻结双杯跨桌搬运对照</h1><p>INTRO。另有1组覆盖受限负对照，单独统计。所有14次方法运行均保留。</p>
<p>三种物理布局：原布局复现、源杯与取杯底盘向右移动20毫米、目标桌与放置点向左移动40毫米。每种分别让冲突候选或可行候选排在首位。运行前已写入全部任务、源码哈希和每阶段时限；每个完整计划均执行接触仿真。</p>
<p>greedy是固定首候选基线；backtrack允许在下游失败后回溯。回溯只发生在规划阶段，成功参考形成后才执行；物理仿真不会倒带或重置杯子。</p>
<img src="FIGURE" alt="完整计划与物理成功数、全部实验结果和实测放置误差">
<p>QUALITY</p>
<h2>全部运行</h2><p>COUNTS</p><table><tr><th>组</th><th>布局</th><th>候选顺序</th><th>方法</th><th>状态</th><th>展开</th><th>搜索记录</th></tr>ROWS</table>
<h2>已执行计划的实测回放</h2><p>选择器列出所有保存的实际轨迹。灰色为底盘，蓝色/橙色为杯子；仅显示实际位置与高度。</p>
<select id="run">OPTIONS</select><canvas id="view" width="1100" height="390"></canvas><button id="play">播放 / 暂停</button> <input id="time" type="range" min="0" value="0" step="1"><p id="label"></p>
<p><a href="EVIDENCE">协议、全部结果与原始产物哈希</a></p><p>每个方法/任务只运行一次；同一布局的两种候选顺序相关。这是预先指定的小规模诊断，不能解释为总体成功率、独立样本显著性结论或速度优势。覆盖受限失败也不能证明物理任务不可行。验收仍为有限数值检查。</p>
<script>const all=DATA;const canvas=document.getElementById('view'),x=canvas.getContext('2d'),select=document.getElementById('run'),s=document.getElementById('time');let playing=false;const px=v=>80+(v+.15)*420,py=v=>345-v*270;
function draw(){let d=all[select.value];if(!d)return;let r=d.samples[+s.value];x.clearRect(0,0,1100,390);let table=d.task.destination_table;for(let t of [[.33,.3,.7,.8],[table.center[0]-table.size[0]/2,table.center[1]-table.size[1]/2,table.size[0],table.size[1]]]){x.fillStyle='#e2e7eb';x.fillRect(px(t[0]),py(t[1]+t[3]),t[2]*420,t[3]*270)}x.fillStyle='#71808b';x.beginPath();x.arc(px(r[2][0]),py(r[2][1]),23,0,7);x.fill();for(let i=3;i<5;i++){x.fillStyle=i===3?'#247f9a':'#c16a38';x.beginPath();x.arc(px(r[i][0]),py(r[i][1]),13,0,7);x.fill();x.fillText('z='+r[i][2].toFixed(3)+' m',px(r[i][0])-30,py(r[i][1])-20)}document.getElementById('label').textContent=r[0].toFixed(2)+' s · '+r[1]}
function choose(){s.value=0;s.max=all[select.value]?.samples.length-1||0;draw()}select.onchange=choose;s.oninput=draw;document.getElementById('play').onclick=()=>playing=!playing;setInterval(()=>{if(playing&&all[select.value]){s.value=(+s.value+1)%all[select.value].samples.length;draw()}},40);choose();</script></html>'''
    for a, b in [('INTRO', html.escape(intro)), ('QUALITY', html.escape(quality)), ('FIGURE', link(figure)), ('ROWS', ''.join(rows)),
                 ('COUNTS', html.escape(str(dict(counts)))), ('OPTIONS', ''.join(options)),
                 ('EVIDENCE', link(evidence)), ('DATA', json.dumps(animation, separators=(',', ':')))]:
        page = page.replace(a, b)
    (output/'index.html').write_text(page)
    data['measured_trace_analysis'] = trace_analysis
    write(evidence, data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'output', 'evidence', 'figure'):
        parser.add_argument('--'+name, type=Path, required=True)
    publish(**vars(parser.parse_args()))
