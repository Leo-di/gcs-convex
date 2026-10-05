"""Post-pilot expansion: fixed additional 100 pairs, no region changes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from .arm_cspace_research import solve
from .arm_boundary_semantics import ArmBoundaryOracle
from .joint_cspace_research import make_scene,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    source=parser.parse_args().source;out=source/'additional_queries';out.mkdir(exist_ok=False)
    heldout=np.load(source/'points.npz')['heldout'];points=heldout[24:224]
    write_json(out/'protocol.json',dict(query_count=100,heldout_point_indices=[24,224],
        note='Expansion specified after observing pilot 0/12 for both methods. Uses the next 200 previously saved points, no selection by outcomes; all pilot failures remain. Regions and solver unchanged.',
        created_unix=time.time(),region_hashes={m:hashlib.sha256((source/f'{m}_regions.json').read_bytes()).hexdigest() for m in ['uniform','vcc']}))
    shutil.copy2(__file__,out/'execution_source.py')
    original=make_scene(json.loads((source/'config.json').read_text()));oracle=ArmBoundaryOracle(original)
    summaries={}
    for method in ['uniform','vcc']:
        records=json.loads((source/f'{method}_regions.json').read_text())
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in records if r['accepted']]
        summaries[method]=[]
        for i in range(100):
            start,goal=points[2*i:2*i+2]
            try:plan=solve(regions,start,goal,original,oracle,True)
            except RuntimeError as exc:plan=dict(success=False,accepted=False,error=str(exc))
            plan.update(query_index=i,start=start.tolist(),goal=goal.tolist(),endpoints_covered=bool(all(any(r.PointInSet(q) for r in regions) for q in [start,goal])))
            write_json(out/f'{method}_query_{i:03}.json',plan)
            summaries[method].append({k:v for k,v in plan.items() if k not in ['sample_times','trajectory_samples','segments']})
            write_json(out/'summary.json',summaries)
            print(method,i,plan['accepted'],flush=True)
    write_json(out/'run_status.json',dict(status='completed'))

if __name__=='__main__':main()
