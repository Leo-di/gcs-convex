"""Audit a historical valid reference against a new 7D cover, without seeding it."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from .joint_cspace_research import make_scene,write_json
from .arm_cspace_research import audit
from .arm_boundary_semantics import ArmBoundaryOracle


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--reference',type=Path,default=Path('outputs/results/joint_cspace_research_v1'))
    args=parser.parse_args()
    config=json.loads((args.source/'config.json').read_text())
    if config!=json.loads((args.reference/'config.json').read_text()):
        raise ValueError('Reference scene differs from current scene')
    protocol=json.loads((args.source/'protocol.json').read_text())
    reference=json.loads((args.reference/'semantic_plan.json').read_text())
    scene=make_scene(config)
    points=np.tile(scene.arm_home_positions,(len(reference['trajectory_samples']),1))
    points[:,:2]=reference['trajectory_samples']
    np.testing.assert_allclose(points[0],protocol['start'],atol=1e-6)
    np.testing.assert_allclose(points[-1],protocol['goal'],atol=1e-6)
    result=dict(reference_source=str(args.reference/'semantic_plan.json'),
                reference_sha256=hashlib.sha256((args.reference/'semantic_plan.json').read_bytes()).hexdigest(),
                role='Post-hoc coverage diagnostic only; never supplied as seed or solution to 7D planner',
                independent_audit=audit(scene,ArmBoundaryOracle(scene),points),
                reference_sampled_length_rad=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum()),
                reference_objective=reference['objective'],conditions={})
    for label in ['collision_only','semantic']:
        records=json.loads((args.source/f'{label}_regions.json').read_text())
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in records if r['accepted']]
        covered=np.array([any(r.PointInSet(q,1e-7) for r in regions) for q in points])
        plan=json.loads((args.source/f'{label}_plan.json').read_text())
        result['conditions'][label]=dict(accepted_regions=len(regions),covered_samples=int(covered.sum()),
            total_samples=len(points),coverage_fraction=float(covered.mean()),
            seven_joint_objective=plan.get('objective'),seven_joint_sampled_length_rad=plan.get('sampled_path_length_rad'),
            scope='Sampled reference inclusion in union of generated polytopes, not full-space coverage fraction')
    write_json(args.source/'reference_coverage.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
