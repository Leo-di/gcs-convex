"""Complete-case paired accounting, including common preparation failures."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import numpy as np


def exact_discordance_test(wins, losses):
    n=wins+losses
    return min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.


def analyze(root):
    protocol=json.loads((root/'protocol.json').read_text())
    design=json.loads((root/'design.json').read_text());cases=design['evaluation_cases']
    methods=protocol['methods'];rows={}
    viable=[];missing=[];coverage_missing=[]
    for cell in cases:
        bundle=json.loads((root/'inputs'/(cell['id']+'.json')).read_text())
        if any(n in bundle['candidates'] for n in cell['allowed']):viable.append(cell['id'])
        if any(not v['goal_covered'] for n,v in bundle['endpoint_coverage'].items() if n in cell['allowed']):coverage_missing.append(cell['id'])
        for m in methods:
            f=root/cell['id']/m/'record.json'
            if not f.exists():missing.append(str(f));continue
            r=json.loads(f.read_text())
            if r['status'] in ('planning','planned'):missing.append(str(f)+' still running')
            rows[cell['id'],m]=r
    if missing:raise RuntimeError('Incomplete evaluation: '+str(missing))
    pairs=[]
    for a,b in itertools.combinations(methods,2):
        both=neither=awins=bwins=0
        for cell in cases:
            x,y=(bool(rows[cell['id'],m]['end_to_end_success']) for m in [a,b])
            both+=x and y;neither+=not x and not y;awins+=x and not y;bwins+=y and not x
        pairs.append({'a':a,'b':b,'both_success':both,'both_failure':neither,'a_only':awins,'b_only':bwins,
                      'b_minus_a_percentage_points':100*(bwins-awins)/len(cases),
                      'exact_mcnemar_two_sided_p':exact_discordance_test(awins,bwins)})
    maximum=0.
    for i,pair in enumerate(sorted(pairs,key=lambda p:p['exact_mcnemar_two_sided_p'])):
        maximum=max(maximum,min(1.,(len(pairs)-i)*pair['exact_mcnemar_two_sided_p']))
        pair['holm_adjusted_p']=maximum
    totals={}
    for m in methods:
        r=[rows[c['id'],m] for c in cases];v=[rows[c,m] for c in viable]
        physics=[json.loads((root/c['cell']/m/'physics.json').read_text())
                 for c in r if (root/c['cell']/m/'physics.json').exists()]
        calls=[a for c in r for report in c.get('online',{}).values() for a in report['attempts']]
        totals[m]={'tasks':len(cases),'planning_successes':sum(x['planning_success'] for x in r),
            'full_task_successes':sum(x['end_to_end_success'] for x in r),'physical_runs':len(physics),
            'online_calls':len(calls),'accepted_online_regions':sum(a['accepted'] for a in calls),
            'successful_tasks_using_online_regions':sum(x['end_to_end_success'] and bool(x.get('used_online_regions')) for x in r),
            'planning_time_common_primitive_valid_subset':{'n':len(v),'median_s':float(np.median([x['planning_wall_seconds'] for x in v])),
                                                        'p90_s':float(np.quantile([x['planning_wall_seconds'] for x in v],.9))},
            'failures':{reason:sum(x.get('failure')==reason for x in r) for reason in sorted({x['failure'] for x in r if 'failure' in x})}}
    hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in protocol['hashes'].items()}
    assert all(hashes.values()),[f for f,ok in hashes.items() if not ok]
    result={'tasks':len(cases),'primitive_valid_tasks':len(viable),'primitive_invalid_tasks':len(cases)-len(viable),
            'valid_tasks_with_uncovered_goal':coverage_missing,'methods':totals,'paired':pairs,'all_frozen_inputs_unchanged':True,
            'interpretation':'One fixed scene, independently sampled continuous task parameters with four balanced policies. Exact paired discordance tests and Holm correction across all three method comparisons are descriptive within this task distribution; no equivalence or cross-scene claim. Forty tasks are not 120 independent tasks.',
            'sample_size_note':'With five discordant cases all favoring one method, two-sided exact p=0.0625; six gives 0.03125 before multiple-comparison adjustment. More total tasks with no disagreements do not create evidence for a method difference.'}
    (root/'analysis.json').write_text(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.root),indent=2))
