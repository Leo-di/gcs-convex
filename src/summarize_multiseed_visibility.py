"""Verify pairing and summarize seed-level results without pooling independence."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .multiseed_visibility_cover import SEEDS
from .iterative_visibility_cover import to_regions
from .visibility_clique_experiment import covered
from .joint_cspace_research import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    root=parser.parse_args().source
    assert json.loads((root/'run_status.json').read_text())['status']=='completed'
    protocol=json.loads((root/'protocol.json').read_text());anchors=json.loads((root/'anchors.json').read_text());query_points=np.load(root/'points.npz')['query_points']
    for f,digest in json.loads((root/'source_sha256.json').read_text()).items():
        assert hashlib.sha256((root/'source_snapshot'/f).read_bytes()).hexdigest()==digest
    rows=[];all_candidates={};selected_queries={}
    for seed in SEEDS:
        folder=root/f'seed_{seed}';summary=json.loads((folder/'summary.json').read_text());history=json.loads((folder/'history.json').read_text())
        coefficient=json.loads((folder/'coefficient_audit.json').read_text());diagnosis=json.loads((folder/'failure_diagnosis.json').read_text())
        masks={};row=dict(seed=seed,methods={});proposal_audit={}
        for method in ['uniform','vcc']:
            record=summary[method];queries=record['queries'][1:];assert len(queries)==100
            for i,q in enumerate(queries):
                np.testing.assert_array_equal(q['start'],query_points[2*i]);np.testing.assert_array_equal(q['goal'],query_points[2*i+1])
            masks[method]=np.array([q['accepted'] for q in queries])
            regions_records=json.loads((folder/f'{method}_regions.json').read_text())
            for i in range(2):
                np.testing.assert_array_equal(regions_records[i]['A'],anchors[i]['A']);np.testing.assert_array_equal(regions_records[i]['b'],anchors[i]['b'])
            new=regions_records[2:];assert len(new)<=8
            ids={r['candidate_index'] for r in new};assert len(ids)==len(new)
            if method=='uniform':all_candidates[seed]=ids
            for key in ['accepted_regions','monitor_count','test_count']:
                vals=[h[key] for h in history[method]];assert vals==sorted(vals)
            regions=to_regions(regions_records);edge_count=sum(regions[i].IntersectsWith(regions[j]) for i in range(len(regions)) for j in range(i+1,len(regions)))
            checked=[v for v in coefficient[method].values() if v['checked']];assert all(v['passed'] for v in checked)
            row['methods'][method]=dict(successes=int(masks[method].sum()),queries=100,coverage_fraction=record['heldout_coverage_count']/5000,
                covered_endpoint_pairs=sum(q['endpoints_covered'] for q in queries),
                success_given_endpoints_covered=float(masks[method].sum()/sum(q['endpoints_covered'] for q in queries)) if any(q['endpoints_covered'] for q in queries) else None,
                coverage_count=record['heldout_coverage_count'],accepted_regions=len(regions),new_attempts=len(new),new_accepted=sum(r['accepted'] for r in new),
                construction_wall_seconds=record['construction_wall_seconds'],iris_elapsed_sum_seconds=record['iris_elapsed_sum_seconds'],overlap_edges=int(edge_count),stop_reason=record['stop_reason'],
                total_query_solver_seconds=sum(q.get('solve_seconds',0) for q in queries),failure_counts=diagnosis[method]['counts'],
                coefficient_checks=len(checked),accepted_trajectory_samples=sum(q.get('audit',{}).get('samples',0) for q in record['queries'] if q['accepted']))
            proposal_audit[method]=[]
            for rd in sorted(folder.glob(f'{method}_round_*')):
                index=int(rd.name.rsplit('_',1)[1]);prev=to_regions([r for r in regions_records if r['coverage_round']<index])
                points=np.load(rd/'construction.npy');assert len(points)==192 and not covered(points,prev).any()
                proposals=json.loads((rd/'proposals.json').read_text())
                proposal_audit[method].append(dict(round=index,samples_outside_previous_union=192,centers_already_covered=int(covered(np.array([x['seed'] for x in proposals]),prev).sum())))
        common=np.flatnonzero(masks['uniform']&masks['vcc'])
        row['paired']=dict(both=len(common),uniform_only=int((masks['uniform']&~masks['vcc']).sum()),vcc_only=int((masks['vcc']&~masks['uniform']).sum()),neither=int((~masks['uniform']&~masks['vcc']).sum()))
        row['success_difference_pp']=row['methods']['vcc']['successes']-row['methods']['uniform']['successes']
        for method in ['uniform','vcc']:
            times=[summary[method]['queries'][int(i)+1]['solve_seconds'] for i in common]
            row['methods'][method]['matched_mean_solver_seconds']=float(np.mean(times)) if times else None
            row['methods'][method]['matched_median_solver_seconds']=float(np.median(times)) if times else None
        selected_queries[str(seed)]=int(common[0]+1) if len(common) else 0
        write_json(folder/'proposal_audit.json',proposal_audit)
        cliques=[json.loads(f.read_text()) for f in sorted(folder.glob('vcc_round_*/cliques.json'))]
        write_json(folder/'cliques.json',dict(visibility_seconds=sum(x['visibility_seconds'] for x in cliques),clique_and_ellipse_seconds=sum(x['clique_and_ellipse_seconds'] for x in cliques),rounds=cliques))
        rows.append(row)
    for i,a in enumerate(SEEDS):
        for b in SEEDS[i+1:]:assert not all_candidates[a]&all_candidates[b]
    aggregate={}
    for method in ['uniform','vcc']:
        aggregate[method]={}
        for key in ['successes','coverage_fraction','construction_wall_seconds','matched_mean_solver_seconds','overlap_edges']:
            values=[r['methods'][method][key] for r in rows if r['methods'][method][key] is not None]
            aggregate[method][key]=dict(values=values,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)) if len(values)>1 else None,minimum=min(values),maximum=max(values))
    result=dict(protocol=protocol,rows=rows,seed_aggregates=aggregate,success_differences_pp=[r['success_difference_pp'] for r in rows],
        selected_queries=selected_queries,display_rule='Each seed: first query accepted by both; no filtering of statistics. Default displayed seed fixed to 1701.',
        inference_scope='Three seed-level paired observations conditional on shared anchors/scene/query set. Means and sample standard deviations are descriptive, not proof of superiority.')
    write_json(root/'aggregate.json',result)
    print(json.dumps(dict(successes={m:aggregate[m]['successes'] for m in aggregate},differences=result['success_differences_pp'],selected_queries=selected_queries)))

if __name__=='__main__':main()
