"""Independent coefficient audit and paired coverage accounting."""
import argparse
import json
from pathlib import Path
import tempfile
import shutil
import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from .validate_arm_cspace_artifacts import validate
from .visibility_clique_experiment import covered
from .joint_cspace_research import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    source=parser.parse_args().source
    # v1 execution snapshot used a misleading CPU label for elapsed durations.
    summary=json.loads((source/'summary.json').read_text())
    for value in summary.values():
        if 'new_iris_cpu_seconds' in value:
            value['new_iris_elapsed_sum_seconds']=value.pop('new_iris_cpu_seconds')
    write_json(source/'summary.json',summary)
    masks={};audits={};diagnostics={}
    points=np.load(source/'points.npz')['heldout']
    for method in ['uniform','vcc']:
        records=json.loads((source/f'{method}_regions.json').read_text())
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in records if r['accepted']]
        masks[method]=covered(points,regions)
        adjacency=[[j for j,r in enumerate(regions) if s.IntersectsWith(r)] for s in regions]
        diagnoses=[]
        for query in sorted(source.glob(f'{method}_query_*.json')):
            plan=json.loads(query.read_text())
            starts={i for i,r in enumerate(regions) if r.PointInSet(plan['start'])}
            goals={i for i,r in enumerate(regions) if r.PointInSet(plan['goal'])}
            reachable=set(starts);front=list(starts)
            while front:
                i=front.pop()
                for j in adjacency[i]:
                    if j not in reachable:reachable.add(j);front.append(j)
            category=('accepted' if plan['accepted'] else 'endpoint_not_covered' if not starts or not goals
                      else 'region_graph_disconnected' if not reachable.intersection(goals)
                      else 'trajectory_audit_failed' if plan.get('success') else 'solver_failed_despite_region_connection')
            diagnoses.append(dict(query=plan['query_index'],category=category,start_regions=sorted(starts),goal_regions=sorted(goals)))
        diagnostics[method]=dict(region_adjacency=adjacency,queries=diagnoses,
            counts={k:sum(q['category']==k for q in diagnoses) for k in sorted({q['category'] for q in diagnoses})})
        audits[method]={}
        with tempfile.TemporaryDirectory() as folder:
            tmp=Path(folder)
            shutil.copy2(source/f'{method}_regions.json',tmp/f'{method}_regions.json')
            for query in sorted(source.glob(f'{method}_query_*.json')):
                shutil.copy2(query,tmp/f'{method}_plan.json')
                audits[method][query.stem]=validate(tmp,[method])[method]
    paired=dict(both=int((masks['uniform']&masks['vcc']).sum()),uniform_only=int((masks['uniform']&~masks['vcc']).sum()),
                vcc_only=int((~masks['uniform']&masks['vcc']).sum()),neither=int((~masks['uniform']&~masks['vcc']).sum()),
                scope='Same independent uniform-feasible point sample; single pair of constructed covers')
    write_json(source/'coefficient_audit.json',audits);write_json(source/'paired_coverage.json',paired)
    write_json(source/'failure_diagnosis.json',diagnostics)
    failed=[k for a in audits.values() for k,v in a.items() if v['checked'] and not v['passed']]
    print(json.dumps(dict(paired=paired,coefficient_failures=failed)))
    if failed:raise RuntimeError('Independent coefficient audit failed')

if __name__=='__main__':main()
