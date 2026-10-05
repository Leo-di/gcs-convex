"""Independently verify the complete query pairing and report every outcome."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .joint_cspace_research import write_json


def timing(values):
    values=np.asarray(values,dtype=float)
    if not len(values):return None
    return dict(count=len(values),mean=float(values.mean()),median=float(np.median(values)),p90=float(np.quantile(values,.9)),maximum=float(values.max()),sum=float(values.sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    root=p.parse_args().source
    assert json.loads((root/'status.json').read_text())['stage']=='completed'
    summary=json.loads((root/'summary.json').read_text());protocol=summary['protocol'];source=Path(protocol['source'])
    rows=summary['rows'];queries=protocol['queries'];assert queries==list(range(1,101)) and len(rows)==200
    assert len({(r['query'],r['method']) for r in rows})==200
    saved=np.load(root/'query_points.npy');np.testing.assert_array_equal(saved,np.load(source/'points.npz')['query_points'])
    for f,h in json.loads((root/'source_sha256.json').read_text()).items():
        assert hashlib.sha256((root/'source_snapshot'/f).read_bytes()).hexdigest()==h
    methods={};pairs={};rejections=[];checked=0;samples=0
    for method in ['uniform','vcc']:
        selected=sorted([r for r in rows if r['method']==method],key=lambda r:r['query']);assert [r['query'] for r in selected]==queries
        offline_records=json.loads((source/f'{method}_regions.json').read_text());new_regions=[];baseline_disagreements=[]
        for row in selected:
            q=row['query'];folder=root/f'case_{q:03}';records=json.loads((folder/f'{method}_regions.json').read_text())
            assert records[:len(offline_records)]==offline_records
            added=records[len(offline_records):];assert len(added)==row['attachment']['attempts']<=2
            assert len({r['online_endpoint'] for r in added})==len(added)
            for record in added:
                i=0 if record['online_endpoint']=='start' else 1
                np.testing.assert_array_equal(record['seed'],saved[2*(q-1)+i]);assert record['candidate_index']==700000+10*q+i
                if not record['accepted']:
                    reasons=[]
                    if 'error' in record:reasons.append('iris_error')
                    if record.get('audit',{}).get('collisions',0):reasons.append('physical_collision')
                    if record.get('audit',{}).get('semantic_violations',0):reasons.append('semantic_violation')
                    if record.get('seed_contained') is False:reasons.append('endpoint_excluded')
                    if record.get('inscribed_radius_rad',1)<=1e-6:reasons.append('no_full_dimensional_interior')
                    rejections.append(dict(query=q,method=method,endpoint=record['online_endpoint'],candidate_index=record['candidate_index'],reasons=reasons or ['other'],error=record.get('error'),audit=record.get('audit')))
            new_regions+=added
            previous=json.loads((source/f'{method}_query_{q:02}.json').read_text())
            if previous['accepted']!=row['offline']['accepted']:baseline_disagreements.append(q)
            for variant,label in [('offline',method+'_offline'),('online',method)]:
                plan=json.loads((folder/f'{label}_plan.json').read_text())
                np.testing.assert_array_equal(plan['start'],saved[2*(q-1)]);np.testing.assert_array_equal(plan['goal'],saved[2*(q-1)+1])
                assert row[variant]['accepted']==plan['accepted']
                audit=json.loads((folder/f'{label}_coefficient_audit.json').read_text())
                if audit['checked']:
                    checked+=1;assert audit['passed']
                if plan['accepted']:
                    assert plan['audit']['collisions']==0 and plan['audit']['semantic_violations']==0
                    samples+=plan['audit']['samples']
            np.testing.assert_allclose(row['online_total_wall_seconds'],row['attachment']['attachment_wall_seconds']+row['online']['planning_and_audit_wall_seconds'])
        methods[method]=dict(queries=100,offline_successes=sum(r['offline']['accepted'] for r in selected),online_successes=sum(r['online']['accepted'] for r in selected),
            recovered=sum(not r['offline']['accepted'] and r['online']['accepted'] for r in selected),lost=sum(r['offline']['accepted'] and not r['online']['accepted'] for r in selected),
            unchanged_successes=sum(r['offline']['accepted'] and r['online']['accepted'] for r in selected),both_failed=sum(not r['offline']['accepted'] and not r['online']['accepted'] for r in selected),
            failure_categories=dict(Counter(r['online']['failure_category'] for r in selected)),calls_per_condition=dict(Counter(r['attachment']['attempts'] for r in selected)),iris_calls=len(new_regions),accepted_new_regions=sum(r['accepted'] for r in new_regions),
            offline_time=timing([r['offline']['planning_and_audit_wall_seconds'] for r in selected]),
            online_planning_time=timing([r['online']['planning_and_audit_wall_seconds'] for r in selected]),
            online_solver_time=timing([r['online'].get('solve_seconds',0) for r in selected]),
            attachment_time=timing([r['attachment']['attachment_wall_seconds'] for r in selected]),
            attachment_time_when_called=timing([r['attachment']['attachment_wall_seconds'] for r in selected if r['attachment']['attempts']]),
            online_core_time=timing([r['online_total_wall_seconds'] for r in selected]),
            successful_online_core_time=timing([r['online_total_wall_seconds'] for r in selected if r['online']['accepted']]),
            baseline_disagreements_with_prior_run=baseline_disagreements)
        pairs[method]={r['query']:r['online']['accepted'] for r in selected}
    paired=dict(both=sum(pairs['uniform'][q] and pairs['vcc'][q] for q in queries),uniform_only=sum(pairs['uniform'][q] and not pairs['vcc'][q] for q in queries),vcc_only=sum(pairs['vcc'][q] and not pairs['uniform'][q] for q in queries),neither=sum(not pairs['uniform'][q] and not pairs['vcc'][q] for q in queries))
    u,v=methods['uniform'],methods['vcc']
    takeaway=f"随机覆盖成功 {u['offline_successes']}→{u['online_successes']}/100；VCC 覆盖 {v['offline_successes']}→{v['online_successes']}/100。在线核心耗时中位数分别为 {u['online_core_time']['median']:.1f} s、{v['online_core_time']['median']:.1f} s（包含失败和无需接入的查询）。这只是构图种子 1701 的结果。"
    write_json(root/'analysis.json',dict(methods=methods,paired=paired,coefficient_checks=checked,accepted_trajectory_samples=samples,takeaway=takeaway,scope=protocol['scope'],timing_scope='Core IRIS + GCS build/solve and dense audits; excludes independent coefficient postprocessing, connectivity diagnostics and file I/O. Concurrent construction, serial GCS.'))
    write_json(root/'online_region_rejections.json',rejections)
    from .diagnose_online_iris_trajectory_failures import diagnose
    diagnose(root)
    print(takeaway,flush=True)

if __name__=='__main__':main()
