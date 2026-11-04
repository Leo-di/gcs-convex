"""Scene-clustered holdout statistics; all budgets and failures stay in denominator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.stats import beta
from scripts.verify_doorway_scope import verify
from scripts.certify_room_reference import certify_candidate
from src.doorway_scope_study import METHODS


def exact_binomial_interval(successes,total,alpha=.05):
    if not 0<=successes<=total or total<1:raise ValueError('Valid nonempty binomial count required')
    return [0. if successes==0 else float(beta.ppf(alpha/2,successes,total-successes+1)),
            1. if successes==total else float(beta.ppf(1-alpha/2,successes+1,total-successes))]


def summarize_scene_scores(scores,bootstrap_seed=8025,bootstrap_samples=10000,alpha=.05):
    values=np.asarray(scores,float)
    if values.ndim!=2 or not len(values) or np.any((values<0)|(values>1)):raise ValueError('Scene by repeat scores in [0,1] required')
    per_scene=values.mean(axis=1);n=len(values);rng=np.random.default_rng(bootstrap_seed)
    resampled=per_scene[rng.integers(0,n,(bootstrap_samples,n))].mean(axis=1)
    robust=int(np.all(values==1,axis=1).sum());mean=float(per_scene.mean());eps=float(np.sqrt(np.log(2/alpha)/(2*n)))
    return dict(independent_scenes=n,repeats=values.shape[1],successful_runs=int(values.sum()),total_runs=values.size,
        scene_averaged_success=mean,scene_cluster_bootstrap_interval=np.quantile(resampled,[alpha/2,1-alpha/2]).tolist(),
        both_repeats_successful_scenes=robust,both_repeats_success_interval=exact_binomial_interval(robust,n,alpha),
        bounded_scene_mean_hoeffding_interval=[max(0.,mean-eps),min(1.,mean+eps)],
        note='Bootstrap resamples scenes, not individual repeats. Degenerate all-success bootstrap intervals do not establish population certainty; exact both-repeat and distribution-free intervals are also shown.')


def analyze(root):
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());runs=json.loads((root/'comparison.json').read_text())
    expected={(r['scene'],r['trial'],r['method']) for r in protocol['runs']}
    assert len(runs)==len(expected) and {(r['scene'],r['trial'],r['method']) for r in runs}==expected
    rows=[];attempts=[]
    for scene in protocol['scenes']:
        for trial in range(len(protocol['spec']['planner_seeds'])):
            folder=root/scene['id']/f'trial_{trial}'
            completed={r['method'] for r in runs if r['scene']==scene['id'] and r['trial']==trial and r['success']}
            available=[m for m in METHODS if m in completed]
            evaluated=verify(folder,methods=available)['rows'] if available else []
            evaluated.extend(dict(method=m,success=False,selected=None,attempts=0) for m in METHODS if m not in available)
            for row in evaluated:
                run=next(r for r in runs if (r['scene'],r['trial'],r['method'])==(scene['id'],trial,row['method']))
                row.update(scene=scene['id'],family=scene['family'],trial=trial,worker_exit=run['process_exit'],worker_wall_seconds=run['wall_seconds'])
                row['attempts']=len(run['search'].get('attempts',[]))
                if not run['success']:
                    row['failure_reason']=run['search'].get('failure_reason',run['search'].get('termination','Worker failed; see retained worker log'))
                row['success']=bool(row['success'] and run['success'])
                row['joint_success']=bool(row['success'] and row['minimum_room_residual_m']>=-1e-8)
                row['raw_joint_success']=bool(row['success'] and row['minimum_raw_room_residual_m']>=-1e-8)
                if row['success']:
                    candidate=folder/row['method']/'operation_first'/row['selected']
                    row['reference_room_certificate']=certify_candidate(candidate)
                    row['replay']=str(Path(scene['id'])/f'trial_{trial}'/row['replay'])
                for a in run['search'].get('attempts',[]):
                    path=folder/row['method']/'operation_first'/a['id']/'candidate_result.json'
                    detail=json.loads(path.read_text()) if path.exists() else {}
                    category='accepted' if a['complete'] else ('candidate_timeout' if detail.get('process_exit')=='timeout' else detail.get('failure_stage',a.get('failure_stage','unknown')))
                    reason=a.get('failure_reason') or ''
                    if not a['complete'] and category!='candidate_timeout':
                        if reason.startswith('IK failed at '):category='ik_'+reason.split()[3]
                        elif reason.startswith('Manipulation endpoint violates whole-robot room semantics'):category='semantic_endpoint_rejection'
                    attempts.append(dict(scene=scene['id'],trial=trial,method=row['method'],candidate=a['id'],complete=a['complete'],category=category or 'unknown',failure_stage=a.get('failure_stage'),reason=a.get('failure_reason')))
                rows.append(row)
    stats=[];settings=protocol['spec']['analysis']
    for method in METHODS:
        group=[r for r in rows if r['method']==method]
        score=[[int(next(r for r in group if r['scene']==s['id'] and r['trial']==trial)['joint_success'])
                for trial in range(len(protocol['spec']['planner_seeds']))] for s in protocol['scenes']]
        stats.append(dict(method=method,**summarize_scene_scores(score,settings['bootstrap_seed'],settings['bootstrap_samples']),
            physical_completions=sum(r['success'] for r in group),raw_containment_completions=sum(r['raw_joint_success'] for r in group),
            completed_but_margin_violating=sum(r['success'] and not r['joint_success'] for r in group),
            completed_with_actual_outside_shape=sum(r['success'] and not r['raw_joint_success'] for r in group),
            reference_certificate_counts=dict(Counter(r.get('reference_room_certificate',{}).get('status','no_completed_execution') for r in group)),
            attempt_categories=dict(Counter(a['category'] for a in attempts if a['method']==method))))
    out=dict(rows=rows,statistics=stats,attempts=attempts,conditions=len(rows),independent_scenes=len(protocol['scenes']),
        analysis_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Inference only for declared IID synthetic scene distribution and these two fixed planner RNG streams; repeated runs are clustered by scene; no arbitrary-environment success guarantee.')
    (root/'analysis.json').write_text(json.dumps(out,indent=2));print(json.dumps(stats,indent=2));return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);analyze(p.parse_args().root)
