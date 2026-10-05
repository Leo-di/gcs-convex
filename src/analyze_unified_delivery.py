"""Paired descriptive and preregistered primary analyses of complete delivery."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
from .benchmark_semantic_baselines import atomic_json

OLD=Path('outputs/results/full_task_evaluation_40_v1')
REG=Path('outputs/results/unified_delivery_regression_v2')
NEW=Path('outputs/results/unified_delivery_fresh40_v1')


def compare(a,b):
    assert set(a)==set(b)
    wins=sum(b[k]['end_to_end_success'] and not a[k]['end_to_end_success'] for k in a)
    losses=sum(a[k]['end_to_end_success'] and not b[k]['end_to_end_success'] for k in a)
    return {'n':len(a),'baseline_successes':sum(r['end_to_end_success'] for r in a.values()),
            'unified_successes':sum(r['end_to_end_success'] for r in b.values()),
            'unified_only':wins,'baseline_only':losses,'difference_percentage_points':100*(wins-losses)/len(a),
            'exact_two_sided_mcnemar_p':float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.,
            'lost_cases':[k for k in a if a[k]['end_to_end_success'] and not b[k]['end_to_end_success']],
            'recovered_cases':[k for k in a if b[k]['end_to_end_success'] and not a[k]['end_to_end_success']]}


def method_summary(rows):
    success=sum(r['end_to_end_success'] for r in rows);ci=binomtest(success,len(rows)).proportion_ci()
    calls=[]
    for r in rows:
        stages=r.get('stages',[{'online':r.get('online',{}),'bridges':{}}])
        for s in stages:
            for kind in ['online','bridges']:
                for info in s.get(kind,{}).values():calls.extend((kind,a) for a in info['attempts'])
    return {'n':len(rows),'planning_successes':sum(r['planning_success'] for r in rows),'successes':success,
            'success_rate':success/len(rows),'exact_binomial_95_interval':[ci.low,ci.high],
            'failure_counts':dict(Counter(r.get('failure','unfinished') for r in rows if not r['end_to_end_success'])),
            'accepted_height_counts':dict(Counter(str(r.get('accepted_height',.22)) for r in rows if r['planning_success'])),
            'accepted_bridge_stages':sum(r.get('accepted_bridge_stage',False) for r in rows),
            'endpoint_iris_calls':sum(k=='online' for k,a in calls),'bridge_iris_calls':sum(k=='bridges' for k,a in calls),
            'planning_seconds_all_cases':{'median':float(np.median([r['planning_wall_seconds'] for r in rows])),
                                          'p90':float(np.percentile([r['planning_wall_seconds'] for r in rows],90))}}


def main():
    a={r['cell']:r for r in json.loads((OLD/'summary.json').read_text())['records'] if r['method']=='product_online_iris'}
    reg=json.loads((REG/'summary.json').read_text())['records'];b={r['cell']:r for r in reg}
    result={'regression':{'comparison':compare(a,b),'unified':method_summary(reg),'development_only':True}}
    # Compare actual reference trajectories, not just boolean success.
    equal=[]
    for k in a:
        if a[k]['end_to_end_success'] and b[k]['planning_success']:
            old=json.loads((Path(a[k]['reference'])/'delivery_trace.json').read_text())
            new=json.loads((Path(b[k]['reference'])/'delivery_trace.json').read_text())
            equal.append({'cell':k,'exact_trace_equal':old==new})
    result['regression']['nominal_success_trace_preservation']=equal
    if (NEW/'complete.json').exists():
        rows=json.loads((NEW/'summary.json').read_text())['records'];methods={m:[r for r in rows if r['method']==m] for m in ['product_online_iris','unified']}
        fresh=json.loads((NEW/'fresh_design.json').read_text());predrawn=json.loads((REG/'fresh_design.json').read_text())
        assert fresh==predrawn,'New tasks differ from pre-drawn design'
        paired=compare(*[{r['cell']:r for r in methods[m]} for m in methods])
        bundles=[json.loads(p.read_text()) for p in sorted((NEW/'inputs').glob('*.json'))]
        common=[x['cell']['id'] for x in bundles if any(n in x['candidates'] for n in x['cell']['allowed'])]
        result['fresh']={'comparison':paired,'methods':{m:method_summary(rs) for m,rs in methods.items()},
                         'predrawn_design_unchanged':True,'nominal_primitive_available_cases':len(common),
                         'common_nominal_valid_planning_seconds':{},'primary_alpha':.05,'primary_test_single_comparison':True}
        for m,rs in methods.items():
            t=[r['planning_wall_seconds'] for r in rs if r['cell'] in common]
            result['fresh']['common_nominal_valid_planning_seconds'][m]={'n':len(t),'median':float(np.median(t)),'p90':float(np.percentile(t,90))}
        result['fresh']['batch_completion']=json.loads((NEW/'complete.json').read_text())
    atomic_json(REG/'analysis.json',result)
    if 'fresh' in result:atomic_json(NEW/'analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
