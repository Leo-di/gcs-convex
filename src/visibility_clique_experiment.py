"""Bounded VCC adapter experiment, Werner et al. ICRA 2024, Algorithm 1.

Uses Drake visibility, MVCE, IRIS-NP2 and GCS. HiGHS solves paper Eq.(1).
One visibility round, four new IRIS attempts, two identical cached anchors.
Not the complete adaptive coverage-termination experiment from the paper.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix
from pydrake.common import Parallelism
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.planning import VisibilityGraph
from .arm_cspace_research import setup, parallel_seed, solve
from .joint_cspace_research import make_scene, write_json
from .arm_boundary_semantics import ArmBoundaryOracle


def maximum_clique(adjacency, seconds=120):
    """Exact maximum clique ILP; timeouts cannot silently become exact results."""
    a=np.asarray(adjacency,dtype=bool)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or not np.array_equal(a,a.T):
        raise ValueError('Expected symmetric square adjacency')
    n=len(a)
    if n==0:
        return np.array([],dtype=int),dict(success=True,size=0)
    i,j=np.where(np.triu(~a,1)); m=len(i)
    matrix=coo_matrix((np.ones(2*m),(np.repeat(np.arange(m),2),np.column_stack([i,j]).ravel())),shape=(m,n)).tocsc()
    began=time.perf_counter()
    result=milp(-np.ones(n),integrality=np.ones(n),bounds=Bounds(0,1),
                constraints=LinearConstraint(matrix,-np.inf,np.ones(m)),
                options=dict(time_limit=seconds,mip_rel_gap=0.))
    info=dict(success=bool(result.success),status=int(result.status),message=result.message,
              seconds=time.perf_counter()-began,mip_gap=float(result.mip_gap) if result.x is not None else None)
    if not result.success:
        return None,info
    selected=np.flatnonzero(result.x>.5)
    if not np.all(a[np.ix_(selected,selected)] | np.eye(len(selected),dtype=bool)):
        raise RuntimeError('MILP result is not a clique')
    info['size']=len(selected)
    return selected,info


def covered(points,regions):
    return np.array([any(r.PointInSet(q) for r in regions) for q in points])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output;out.mkdir(parents=True,exist_ok=False)
    old=Path('outputs/results/arm_cspace_research_v2')
    config=json.loads((old/'config.json').read_text()); protocol=json.loads((old/'protocol.json').read_text())
    anchors=json.loads((old/'semantic_regions.json').read_text())[:2]
    assert [r['candidate_index'] for r in anchors]==[0,1] and all(r['accepted'] for r in anchors)
    regions0=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in anchors]
    protocol.update(experiment='VCC vs uniform seeding; same semantic scene and common cached anchors',
        seed_policy='192 uniform feasible configurations outside two cached anchors; first four for uniform, four clique MVCEs for VCC',
        max_attempts_per_condition=4,parallel_seed_policy='One batch of four new calls per method; two shared anchors reused without regeneration',
        new_attempts_per_method=4,visibility_points=192,minimum_clique_size=8,visibility_rounds=1,
        pool_seed=20260930,holdout_seed=20260931,heldout_feasible_points=2000,heldout_query_pairs=12,
        budget='Equal number of IRIS calls, NOT equal wall time; include visibility/MIP/MVCE overhead',
        shared_anchor_source=str(old),scope='Single coverage realization, fixed base, seven independent joints. No statistical success claim.',
        reference='https://arxiv.org/abs/2310.02875',adapter='Drake VisibilityGraph + MVCE + IrisNp2; HiGHS exact max-clique ILP; one bounded round; same affine 7-to-19 mapping',
        clique_time_limit_s=120,workers=4)
    write_json(out/'protocol.json',protocol);write_json(out/'config.json',config)
    hashes={}
    for f in ['src/visibility_clique_experiment.py','src/arm_cspace_research.py','src/joint_cspace_research.py','src/build_scene.py','src/arm_boundary_semantics.py','src/collision_checker.py','src/iris_region_generator.py']:
        dest=out/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
        hashes[f]=hashlib.sha256(Path(f).read_bytes()).hexdigest()
    write_json(out/'source_sha256.json',hashes)
    planning=make_scene(config,semantic=True); original=make_scene(config);oracle=ArmBoundaryOracle(original)
    checker,parameterize,lo,hi=setup(planning)
    def sample(n,seed,exclude):
        rng=np.random.default_rng(seed);points=[];trials=0
        while len(points)<n and trials<200000:
            q=rng.uniform(lo,hi);trials+=1
            if not any(r.PointInSet(q) for r in exclude) and checker.CheckConfigCollisionFree(parameterize(q)):
                points.append(q)
        if len(points)!=n: raise RuntimeError('Sampling budget exhausted')
        return np.array(points),trials
    began=time.perf_counter();points,trials=sample(192,20260930,regions0)
    # Held-out data is fixed before either method is inflated or evaluated.
    holdout,heldout_trials=sample(2000,20260931,[])
    np.savez_compressed(out/'points.npz',construction=points,heldout=holdout)
    write_json(out/'sampling.json',dict(construction_trials=trials,heldout_trials=heldout_trials,seconds=time.perf_counter()-began))
    proposals={'uniform':[dict(seed=q.tolist(),A=None) for q in points[:4]],'vcc':[]}
    began=time.perf_counter()
    adjacency=VisibilityGraph(checker,np.asfortranarray(np.column_stack([parameterize(q) for q in points])),Parallelism(1)).toarray()
    np.save(out/'visibility.npy',adjacency)
    details=dict(visibility_seconds=time.perf_counter()-began,edges=int(np.triu(adjacency,1).sum()),cliques=[])
    remaining=np.arange(len(points)); began=time.perf_counter()
    for _ in range(4):
        selected,info=maximum_clique(adjacency[np.ix_(remaining,remaining)])
        details['cliques'].append(info)
        if selected is None or len(selected)<8: break
        ids=remaining[selected];info['point_indices']=ids.tolist()
        remaining=np.delete(remaining,selected)
        try:
            ell=Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(points[ids].T,1e-6)
            center=ell.center(); info['recentered']=False
            if not checker.CheckConfigCollisionFree(parameterize(center)):
                center=points[ids[np.argmin(np.linalg.norm(points[ids]-center,axis=1))]];info['recentered']=True
            proposals['vcc'].append(dict(seed=center.tolist(),A=ell.A().tolist()))
        except RuntimeError as exc:
            info['ellipse_error']=str(exc)
        write_json(out/'cliques.json',details)
    details['clique_and_ellipse_seconds']=time.perf_counter()-began
    write_json(out/'cliques.json',details);write_json(out/'proposals.json',proposals)
    if len(proposals['vcc'])!=4:
        write_json(out/'run_status.json',dict(status='preparation_failed',reason='Fewer than four valid full-dimensional clique ellipsoids; no fallback'))
        return
    summary={}
    for method in ['uniform','vcc']:
        began=time.perf_counter();records=[dict(r,shared_cached_anchor=True) for r in anchors]
        with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs=[pool.submit(parallel_seed,config,True,p['seed'],100+i,6,p['A']) for i,p in enumerate(proposals[method])]
            for job in jobs:
                records.append(job.result());write_json(out/f'{method}_regions.json',records)
        inflation_wall=time.perf_counter()-began
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in records if r['accepted']]
        mask=covered(holdout,regions)
        metrics=dict(accepted_regions=len(regions),new_attempts=4,new_accepted=sum(r['accepted'] for r in records[2:]),
            inflation_wall_seconds=inflation_wall,new_iris_elapsed_sum_seconds=sum(r.get('iris_seconds',r.get('elapsed_to_error_seconds',0)) for r in records[2:]),
            heldout_coverage_count=int(mask.sum()),heldout_coverage_denominator=len(mask),heldout_coverage_fraction=float(mask.mean()))
        pairs=[(np.array(protocol['start']),np.array(protocol['goal']))]+[(holdout[2*i],holdout[2*i+1]) for i in range(12)]
        plans=[]
        for qi,(start,goal) in enumerate(pairs):
            print('GCS',method,qi,flush=True)
            try: plan=solve(regions,start,goal,original,oracle,True)
            except RuntimeError as exc: plan=dict(success=False,accepted=False,error=str(exc))
            plan.update(query_index=qi,start=start.tolist(),goal=goal.tolist(),endpoints_covered=bool(covered([start,goal],regions).all()))
            write_json(out/f'{method}_query_{qi:02}.json',plan)
            plans.append({k:v for k,v in plan.items() if k not in ['trajectory_samples','sample_times','segments']})
        metrics.update(queries=plans,query_success_count=sum(p['accepted'] for p in plans),heldout_query_success_count=sum(p['accepted'] for p in plans[1:]),query_count=len(plans))
        summary[method]=metrics;write_json(out/'summary.json',summary)
        print('SUMMARY',method,json.dumps({k:v for k,v in metrics.items() if k!='queries'}),flush=True)
    write_json(out/'run_status.json',dict(status='completed'))

if __name__=='__main__':main()
