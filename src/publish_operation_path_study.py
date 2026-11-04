"""Compare a new frozen path study with the preserved report-80 experiment."""
import argparse
import html
import json
import os
from pathlib import Path

import numpy as np

from .benchmark_stateful_transport import read, write, file_hash
from .publish_stateful_benchmark import publish
from .scene_spec import PROJECT_ROOT


def verify_archive(root, evidence):
    """Verify old code against its snapshot, rather than today's working tree."""
    protocol = read(root/'protocol.json')
    if read(root/'summary.json')['protocol_sha256'] != file_hash(root/'protocol.json'):
        raise ValueError('Archived protocol changed')
    for name, expected in protocol['input_sha256'].items():
        if file_hash(root/'source_snapshot'/name) != expected:
            raise ValueError('Archived source snapshot changed: '+name)
    for cell, expected in protocol['task_sha256'].items():
        if file_hash(root/'inputs'/(cell+'.json')) != expected:
            raise ValueError('Archived task changed: '+cell)
    data = read(evidence)
    if data['protocol'] != protocol or data['summary'] != read(root/'summary.json'):
        raise ValueError('Archived evidence disagrees with protocol or summary')
    for name, expected in data['artifact_sha256'].items():
        if file_hash(PROJECT_ROOT/name) != expected:
            raise ValueError('Archived artifact changed: '+name)
    return data


def path_metrics(frames):
    result = {}
    for phase in dict.fromkeys(f['phase'] for f in frames):
        if not phase.endswith(('.reach', '.retreat')):
            continue
        q = np.array([f['arm_q'] for f in frames if f['phase'] == phase])
        delta = np.diff(q, axis=0)
        result[phase] = {'frames': len(q),
            'joint_length_rad': float(np.linalg.norm(delta, axis=1).sum()),
            'joint_linf_length_rad': float(np.max(abs(delta), axis=1).sum()),
            'nearly_stationary_edges_below_1e_8_rad': int(np.sum(np.max(abs(delta), axis=1) < 1e-8)),
            'maximum_joint_step_rad': float(np.max(abs(delta)))}
    return result


def build(source, baseline, baseline_evidence, output, evidence, figure):
    source, baseline, baseline_evidence, output, evidence, figure = [Path(p).resolve() for p in
        (source, baseline, baseline_evidence, output, evidence, figure)]
    old = verify_archive(baseline, baseline_evidence)
    new_protocol = read(source/'protocol.json')
    for key in ('task_sha256', 'cells', 'methods', 'planning_budget_s', 'execution_budget_s'):
        if new_protocol[key] != old['protocol'][key]:
            raise ValueError('Comparison requires identical frozen '+key)
    cohort_figure = figure.with_name(figure.stem+'_cohort.png')
    new = publish(source, output, evidence, cohort_figure)
    comparisons, rows, hashes = [], [], {}
    for record in new['summary']['records']:
        cell, method = record['cell'], record['method']
        prior = next(r for r in old['summary']['records'] if r['cell'] == cell and r['method'] == method)
        row = {'cell': cell, 'method': method, 'old_status': prior['status'], 'new_status': record['status']}
        for label, root, data, status in [('old', baseline, old, prior), ('new', source, new, record)]:
            if not status['planning_success']:
                continue
            reference_path = root/cell/method/'reference.json'
            row[label+'_paths'] = path_metrics(read(reference_path))
            hashes[str(reference_path.relative_to(PROJECT_ROOT))] = file_hash(reference_path)
            measured = data['measured_trace_analysis'].get(cell+'__'+method)
            if measured:
                row[label+'_measured'] = measured
        comparisons.append(row)
        if 'old_measured' in row and 'new_measured' in row:
            values = [cell, method]
            values += [f"{row[label+'_measured']['duration_s']:.2f}" for label in ('old', 'new')]
            values += [f"{row[label+'_measured']['phase_duration_s_approximately_50ms_resolution']['move_second_cup.reach']:.2f}"
                       for label in ('old', 'new')]
            rows.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in values)+'</tr>')
    comparisons.sort(key=lambda r: (r['cell'], r['method']))
    new['paired_path_comparison'] = comparisons
    new['baseline'] = {'directory': str(baseline.relative_to(PROJECT_ROOT)),
        'protocol_sha256': file_hash(baseline/'protocol.json'),
        'evidence_sha256': file_hash(baseline_evidence),
        'archive_snapshot_and_artifacts_verified': True, 'all_task_hashes_identical': True}
    new['comparison_artifact_sha256'] = hashes
    for artifact in source.glob('*/*/branch_*/pickup/*_path.json'):
        new['comparison_artifact_sha256'][str(artifact.relative_to(PROJECT_ROOT))] = file_hash(artifact)
    new['publication_source_sha256'][str(Path(__file__).resolve().relative_to(PROJECT_ROOT))] = file_hash(__file__)
    physical = [read(source/r['cell']/r['method']/'result.json') for r in new['summary']['records']
                if (source/r['cell']/r['method']/'physics.json').exists()]
    actions = [action for result in physical for action in result['actions'].values()]
    new['physical_quality_summary'] = {
        'saved_continuous_executions': len(physical),
        'online_samples': sum(r['online_samples'] for r in physical),
        'measured_semantic_samples': sum(r['semantic_audit']['samples'] for r in physical),
        'maximum_final_position_error_m': max((a['final_position_error_m'] for a in actions), default=None),
        'maximum_hold_carry_slip_m': max((a['hold_carry_slip_m'] for a in actions), default=None),
        'minimum_robot_and_objects_room_residual_m': min((r['minimum_robot_and_objects_room_residual_m'] for r in physical), default=None),
        'unexpected_contacts': sum(r['unexpected_contacts'] for r in physical),
        'all_effort_limits_satisfied': all(r['effort_limits'] for r in physical),
        'all_measured_action_preconditions_satisfied': all(g['satisfied'] for r in physical for g in r['measured_action_preconditions']),
        'pose_resets_after_initialization': sum(r['object_pose_resets_after_initialization'] for r in physical),
        'any_cup_attachment': any(r['cup_attachment'] for r in physical)}
    new['comparison_scope'] = ('Same frozen tasks, budgets and execution speed caps; independent fresh planning and dynamics. '
        'Joint changes combine guide and post-GCS simplification; no isolated component attribution, statistical claim or hardware timing claim. '
        'An earlier interrupted planning pilot is preserved at outputs/results/stateful_transport_path_v2_20261020/aborted_diagnostic.json; '
        'it is not a completed experiment and is excluded from before/after outcome claims.')
    write(evidence, new)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected = [next(r for r in comparisons if r['cell'] == layout+'__free_first' and r['method'] == 'backtrack')
                for layout in ('replication', 'source_right_20mm', 'destination_left_40mm')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7), layout='constrained')
    for ax, field, title, ylabel in zip(axes, ('duration', 'reach', 'frames'),
        ('Continuous contact execution', 'Second cup: measured reach phase', 'Second cup: reach reference'),
        ('Simulated seconds', 'Simulated seconds', 'Uniform reference frames')):
        for offset, label, color in [(-.18, 'old', '#b8a18a'), (.18, 'new', '#287f92')]:
            values = []
            for row in selected:
                if label+'_measured' not in row:
                    value = np.nan
                elif field == 'frames':
                    value = row[label+'_paths']['move_second_cup.reach']['frames']
                elif field == 'duration':
                    value = row[label+'_measured']['duration_s']
                else:
                    value = row[label+'_measured']['phase_duration_s_approximately_50ms_resolution']['move_second_cup.reach']
                values.append(value)
            bars = ax.bar(np.arange(3)+offset, values, .34, label=label, color=color)
            ax.bar_label(bars, fmt='%d' if field == 'frames' else '%.2f', fontsize=8)
        ax.set(xticks=range(3), xticklabels=['replica', 'source +20 mm', 'target -40 mm'],
               title=title, ylabel=ylabel)
        ax.set_ylim(0, ax.get_ylim()[1]*1.12)
        ax.legend()
    fig.savefig(figure, dpi=160)
    plt.close(fig)
    rel = html.escape(os.path.relpath(figure, output))
    intro = ('<h2>同一冻结任务下的新旧路径对照</h2><p>全部任务JSON哈希与报告80逐一相同；旧源码按原始快照核验。'
             '保留同样的速度上限、未过滤抓取检查、整段语义和连续自由物体接触验收。'
             '图取各布局可行优先的回溯运行；下表列出全部双方完成物理执行的配对。</p>'
             f'<img src="{rel}" alt="三个冻结布局的总时长、第二杯伸臂时间与帧数改进对比">'
             '<table><tr><th>任务</th><th>方法</th><th>旧总秒</th><th>新总秒</th><th>旧伸臂秒</th><th>新伸臂秒</th></tr>'
             +''.join(rows)+'</table>')
    page = output/'index.html'
    page.write_text(page.read_text().replace('<h1>冻结双杯跨桌搬运对照</h1>',
        '<h1>报告81：手臂路径简化与冻结回归</h1>'+intro).replace(
            '<title>冻结双杯跨桌搬运对照</title>', '<title>手臂路径简化与冻结回归</title>'))
    new['publication_artifact_sha256'] = {str(p.relative_to(PROJECT_ROOT)): file_hash(p)
                                         for p in (figure, cohort_figure, page)}
    write(evidence, new)
    return new


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'baseline', 'baseline-evidence', 'output', 'evidence', 'figure'):
        parser.add_argument('--'+name, type=Path, required=True)
    build(**vars(parser.parse_args()))
