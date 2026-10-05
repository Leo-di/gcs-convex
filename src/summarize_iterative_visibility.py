"""Summarize a completed continuation without changing any planning result."""
import argparse
import json
from pathlib import Path
import numpy as np
from .joint_cspace_research import write_json
from .iterative_visibility_cover import to_regions
from .visibility_clique_experiment import covered


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    source=parser.parse_args().source;old=Path('outputs/results/visibility_clique_v1')
    assert json.loads((source/'run_status.json').read_text())['status']=='completed'
    summary=json.loads((source/'summary.json').read_text());previous=json.loads((old/'additional_queries/summary.json').read_text())
    transitions={};current={}
    history=json.loads((source/'history.json').read_text())
    for method in ['uniform','vcc']:
        for key in ['accepted_regions','monitor_count','test_count']:
            values=[r[key] for r in history[method]]
            assert values==sorted(values), (method,key,'Coverage union must only grow')
        assert summary[method]['additional_calls']<=8
        new=summary[method]['queries'][1:];before=previous[method]
        assert len(new)==len(before)==100
        for a,b in zip(new,before):
            np.testing.assert_array_equal(a['start'],b['start']);np.testing.assert_array_equal(a['goal'],b['goal'])
        a=np.array([x['accepted'] for x in before]);b=np.array([x['accepted'] for x in new]);current[method]=b
        transitions[method]=dict(before=int(a.sum()),after=int(b.sum()),gained_queries=(np.flatnonzero(~a&b)+1).tolist(),
            lost_queries=(np.flatnonzero(a&~b)+1).tolist(),both=int((a&b).sum()),neither=int((~a&~b).sum()))
    was_failed=~np.array([x['accepted'] for x in previous['uniform']]) & ~np.array([x['accepted'] for x in previous['vcc']])
    rescued=np.flatnonzero(was_failed & current['uniform'] & current['vcc'])+1
    display=int(rescued[0]) if len(rescued) else 0
    paired=dict(both=int((current['uniform']&current['vcc']).sum()),uniform_only=int((current['uniform']&~current['vcc']).sum()),
        vcc_only=int((~current['uniform']&current['vcc']).sum()),neither=int((~current['uniform']&~current['vcc']).sum()))
    common=np.flatnonzero(current['uniform']&current['vcc'])
    timing=dict(common_successful_queries=(common+1).tolist(),methods={})
    for method in ['uniform','vcc']:
        values=[summary[method]['queries'][int(i)+1]['solve_seconds'] for i in common]
        timing['methods'][method]=dict(total_100_query_solver_seconds=sum(q.get('solve_seconds',0) for q in summary[method]['queries'][1:]),
            matched_mean_solver_seconds=float(np.mean(values)) if values else None,
            matched_median_solver_seconds=float(np.median(values)) if values else None)
    write_json(source/'matched_solver_timing.json',timing)
    write_json(source/'query_transitions.json',dict(methods=transitions,paired=paired,display_query_index=display,
        display_policy='Smallest query index previously failed by both, now accepted by both; else original query 0. Visualization only.'))
    rejections={}
    proposal_audit={}
    for method in ['uniform','vcc']:
        records=json.loads((source/f'{method}_regions.json').read_text())
        new=[r for r in records if r.get('coverage_round',0)>=2]
        rejections[method]=dict(attempts=len(new),accepted=sum(r['accepted'] for r in new),
            rejected=[{k:v for k,v in r.items() if k not in ['A','b','initial_ellipse_A']} for r in new if not r['accepted']])
        proposal_audit[method]=[]
        for folder in sorted(source.glob(f'{method}_round_*')):
            index=int(folder.name.rsplit('_',1)[1])
            previous_regions=to_regions([r for r in records if r.get('coverage_round',1)<index])
            points=np.load(folder/'construction.npy')
            assert not covered(points,previous_regions).any()
            proposals=json.loads((folder/'proposals.json').read_text())
            centers=np.array([p['seed'] for p in proposals])
            proposal_audit[method].append(dict(round=index,construction_samples_outside_previous_union=len(points),
                ellipse_centers_already_covered=int(covered(centers,previous_regions).sum()),proposals=len(proposals)))
    write_json(source/'region_rejections.json',rejections)
    write_json(source/'proposal_audit.json',proposal_audit)
    all_cliques=[];visibility=ellipses=0
    for f in sorted(source.glob('vcc_round_*/cliques.json')):
        d=json.loads(f.read_text());visibility+=d['visibility_seconds'];ellipses+=d['clique_and_ellipse_seconds'];all_cliques.append(dict(round=f.parent.name,**d))
    write_json(source/'cliques.json',dict(visibility_seconds=visibility,clique_and_ellipse_seconds=ellipses,rounds=all_cliques))
    print(json.dumps(dict(transitions=transitions,paired=paired,display_query_index=display)))

if __name__=='__main__':main()
