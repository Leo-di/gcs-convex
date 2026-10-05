"""Aggregate audited baseline results; retain failures and censored runs."""
import argparse
import json
from pathlib import Path
import statistics


METHODS=['B1_first_valid','B2_enumerate','joint_gcs']


def med(values):return statistics.median(values) if values else None


def summarize(records,budget=300):
    by_method={}
    positive=[r for r in records if r.get('allowed') or ('allowed' not in r and '__none__' not in r['cell'])]
    negative=[r for r in records if r not in positive]
    for method in METHODS:
        rows=[r for r in positive if r['method']==method]
        valid=[r for r in rows if r.get('best') and r['best']['discovered_seconds']<=budget]
        history=lambda r,t:[h for h in r.get('incumbent_history',[]) if h['seconds']<=t]
        by_method[method]={'positive_runs':len(rows),'audited_successes':len(valid),
          'no_valid_plan':len(rows)-len(valid),'timeouts':sum(r.get('timed_out',False) for r in rows),
          'errors':sum(r['status']=='error' for r in rows),
          'invalid_attempts':sum(a.get('solver_status')=='solved' and not a.get('accepted',False) for r in rows for a in r['attempts']),
          'median_wall_seconds_all_runs':med([min(r['parent_wall_seconds'],budget) for r in rows]),
          'median_first_valid_seconds_successes':med([r['incumbent_history'][0]['seconds'] for r in valid]),
          'median_cost_successes':med([r['best']['cost'] for r in valid]),
          'fully_completed_enumerations':sum(r.get('enumeration_complete',False) for r in rows) if method=='B2_enumerate' else None,
          'by_budget':{str(t):{'successes':sum(bool(history(r,t)) for r in rows),
                              'median_cost_successes':med([min(h['cost'] for h in history(r,t)) for r in rows if history(r,t)])}
                       for t in [10,30,60,120,300]},
          'negative_runs':sum(r['method']==method for r in negative),
          'negative_rejected':sum(r['method']==method and r['status']=='unreachable_in_compiled_graph' and not r.get('best') for r in negative)}
    cells={}
    for r in positive:cells.setdefault(r['cell'],{})[r['method']]=r
    pairs=[]
    for cell,group in cells.items():
        if not all(m in group and group[m].get('best') and group[m]['best']['discovered_seconds']<=budget for m in METHODS):continue
        costs={m:group[m]['best']['cost'] for m in METHODS}
        pairs.append({'cell':cell,'costs':costs,
                      'joint_vs_enumerate_relative':costs['joint_gcs']/costs['B2_enumerate']-1,
                      'joint_vs_first_relative':costs['joint_gcs']/costs['B1_first_valid']-1})
    return {'methods':by_method,'paired_all_success_count':len(pairs),'paired_costs':pairs,
            'joint_same_candidate_as_first':sum(
                group.get('joint_gcs',{}).get('best') is not None and group.get('B1_first_valid',{}).get('best') is not None
                and group['joint_gcs']['best'].get('candidate') is not None
                and group['joint_gcs']['best'].get('candidate')==group['B1_first_valid']['best'].get('candidate')
                for group in cells.values()),
            'median_joint_vs_enumerate_relative':med([p['joint_vs_enumerate_relative'] for p in pairs]),
            'median_joint_vs_first_relative':med([p['joint_vs_first_relative'] for p in pairs]),
            'joint_better_than_enumerate':sum(p['joint_vs_enumerate_relative']<-.001 for p in pairs),
            'joint_worse_than_enumerate':sum(p['joint_vs_enumerate_relative']>.001 for p in pairs),
            'joint_tied_with_enumerate':sum(abs(p['joint_vs_enumerate_relative'])<=.001 for p in pairs),
            'scope':'Repeated rounding seeds on nine positive task conditions in one fixed physical scene; no population success-rate claim. B2 is best observed enumeration, not globally optimal truth.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/results/semantic_baselines_long'));args=p.parse_args()
    root=args.root;spec=json.loads((root/'protocol.json').read_text())
    progress=json.loads((root/'progress.json').read_text())
    data=summarize(progress['runs'],spec['budget_seconds_per_method'])
    data.update(completed=progress['completed'],total=progress['total'],planning_complete=(root/'complete.json').exists(),
                protocol=spec,planning_elapsed_seconds=progress['elapsed_seconds'])
    if (root/'contacts.json').exists():data['contacts']=json.loads((root/'contacts.json').read_text())
    (root/'summary.json').write_text(json.dumps(data,indent=2))
    print(json.dumps({k:v for k,v in data.items() if k not in ('protocol','paired_costs','contacts')},indent=2))


if __name__=='__main__':main()
