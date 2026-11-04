import copy
import subprocess
import sys

import pytest

from src.benchmark_stateful_transport import design, summarize, verify, write, file_hash
from src.scene_spec import load_scene_config


def cells():
    return design(load_scene_config('configs/stateful_transport_grounding.yaml'))


def test_frozen_design_retains_order_pairs_and_separate_representation_control():
    base = load_scene_config('configs/stateful_transport_grounding.yaml')
    before = copy.deepcopy(base)
    cohort = design(base)
    assert base == before and len(cohort) == 7
    positive = [c for c in cohort if c['cohort'] == 'transport']
    assert len(positive) == 6
    for a, b in zip(positive[::2], positive[1::2]):
        assert a['layout'] == b['layout']
        reversed_task = copy.deepcopy(a['task'])
        reversed_task['actions'][0]['placements'].reverse()
        assert b['task'] == reversed_task
    assert cohort[-1]['cohort'] == 'representation_control'
    assert cohort[-1]['task']['carry']['lower_xy'][1] == -.04


def test_summary_keeps_all_denominators_and_stage_failures():
    cohort = cells()
    records = []
    for i, c in enumerate(cohort):
        for m in ('greedy', 'backtrack'):
            status = 'success' if m == 'backtrack' else 'planning_rejected'
            if i == 1:
                status = 'execution_timeout' if m == 'backtrack' else 'execution_rejected'
            if c['cohort'] != 'transport':
                status = 'planning_timeout'
            records.append({'cell': c['id'], 'method': m, 'status': status,
                            'planning_success': status.startswith('execution') or status == 'success',
                            'end_to_end_success': status == 'success'})
    result = summarize(records, cohort)
    assert result['complete'] and result['expected_runs'] == 14
    main = result['cohorts']['transport']
    assert main['methods']['backtrack']['denominator'] == 6
    assert main['methods']['backtrack']['end_to_end_successes'] == 5
    assert main['methods']['backtrack']['outcomes']['execution_timeout'] == 1
    assert main['paired']['backtrack_only'] == 5 and main['paired']['both_failure'] == 1
    assert result['cohorts']['representation_control']['methods']['greedy']['denominator'] == 1
    missing = summarize(records[:-1], cohort)
    assert not missing['complete']
    assert missing['cohorts']['representation_control']['paired']['incomplete'] == 1
    with pytest.raises(ValueError, match='Duplicate'):
        summarize(records+[records[0]], cohort)


def test_frozen_task_mutation_is_rejected(tmp_path):
    (tmp_path/'inputs').mkdir()
    task = tmp_path/'inputs'/'sample.json'
    write(task, {'objects': ['cup']})
    write(tmp_path/'protocol.json', {'input_sha256': {}, 'cells': [{'id': 'sample'}],
                                     'task_sha256': {'sample': file_hash(task)}})
    verify(tmp_path)
    write(task, {'objects': ['cup', 'cup_b']})
    with pytest.raises(RuntimeError, match='Frozen task changed'):
        verify(tmp_path)


def test_hard_deadline_terminates_worker_and_preserves_failure_denominator(tmp_path, monkeypatch):
    from src import benchmark_stateful_transport as benchmark
    directory = tmp_path/'cell'/'backtrack'
    directory.mkdir(parents=True)
    write(directory/'record.json', {'cell': 'cell', 'method': 'backtrack', 'status': 'execution',
                                    'planning_success': True, 'end_to_end_success': False})
    original = subprocess.Popen
    processes = []
    def sleeping_worker(command, **kwargs):
        process = original([sys.executable, '-c', 'import time; time.sleep(20)'], **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(benchmark.subprocess, 'Popen', sleeping_worker)
    row = benchmark.run_stage(tmp_path, 'cell', 'backtrack', 'execution', .05)
    assert row['status'] == 'execution_timeout'
    assert row['planning_success'] and not row['end_to_end_success']
    assert processes[0].poll() is not None
