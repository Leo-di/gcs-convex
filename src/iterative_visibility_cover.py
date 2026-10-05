"""Continue both v1 covers with uncovered-space resampling (VCC Alg. 1).

Maximum three total rounds, four new calls per round, 80% monitor target.
Held-out evaluation points and query endpoints never guide construction.
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
from pydrake.common import Parallelism
from pydrake.geometry.optimization import HPolyhedron,Hyperellipsoid
from pydrake.planning import VisibilityGraph
from .arm_cspace_research import setup,parallel_seed,solve
from .visibility_clique_experiment import maximum_clique,covered
from .joint_cspace_research import make_scene,write_json
from .arm_boundary_semantics import ArmBoundaryOracle


def feasible_samples(checker,param,lo,hi,count,seed,regions=()):
    rng=np.random.default_rng(seed);points=[];trials=0
    while len(points)<count and trials<200000:
        q=rng.uniform(lo,hi);trials+=1
        if not any(r.PointInSet(q) for r in regions) and checker.CheckConfigCollisionFree(param(q)):
            points.append(q)
    if len(points)<count:raise RuntimeError('Feasible sampling budget exhausted')
    return np.array(points),trials


def to_regions(records):
    return [HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in records if r['accepted']]


def clique_proposals(checker,param,points,folder):
    t=time.perf_counter()
    graph=VisibilityGraph(checker,np.asfortranarray(np.column_stack([param(q) for q in points])),Parallelism(4)).toarray()
    np.save(folder/'visibility.npy',graph)
    details=dict(visibility_seconds=time.perf_counter()-t,edges=int(np.triu(graph,1).sum()),cliques=[])
    remaining=np.arange(len(points));proposals=[];t=time.perf_counter()
    while len(proposals)<4 and len(remaining)>=8:
        chosen,info=maximum_clique(graph[np.ix_(remaining,remaining)])
        details['cliques'].append(info)
        if chosen is None or len(chosen)<8:break
        ids=remaining[chosen];remaining=np.delete(remaining,chosen);info['point_indices']=ids.tolist()
        try:
            ellipse=Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(points[ids].T,1e-6)
            center=ellipse.center();info['recentered']=False
            if not checker.CheckConfigCollisionFree(param(center)):
                center=points[ids[np.argmin(np.linalg.norm(points[ids]-center,axis=1))]];info['recentered']=True
            proposals.append(dict(seed=center.tolist(),A=ellipse.A().tolist()))
        except RuntimeError as exc:info['ellipse_error']=str(exc)
    details['clique_and_ellipse_seconds']=time.perf_counter()-t
    write_json(folder/'cliques.json',details)
    return proposals,details


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    old=Path('outputs/results/visibility_clique_v1')
    config=json.loads((old/'config.json').read_text());protocol=json.loads((old/'protocol.json').read_text())
    protocol.update(experiment='Adaptive uncovered-space coverage, continued from recorded round 1',
        max_total_rounds=3,additional_rounds=2,max_additional_calls_per_method=8,calls_per_round=4,
        stopping_target=.8,stopping_points=500,monitor_seed=90300,test_seed=90301,
        construction_seeds=[92202,92203],test_points=2000,query_count=100,
        pool_seed=None,holdout_seed=90301,heldout_query_pairs=100,max_accepted_regions=14,
        adapter='Iterative VCC flow with official visibility/MVCE/IRIS-NP2 and exact clique ILP; budgeted four inflations per round',
        query_source='v1 heldout[24:224], same 100 pairs as prior additional_queries; not construction inputs',
        stopping_scope='Only independent monitor points choose stopping; heldout test points never choose stopping',
        visibility_parallelism=4,workers=4,region_delta=.01/12,
        budget='Same maximum calls per method; actual calls and elapsed costs recorded; stop at 80% monitor or budget',
        max_attempts_per_condition=8,new_attempts_per_method=8,visibility_rounds=2,
        seed_policy='Each round resamples currently uncovered feasible space; uniform first four seeds or VCC cliques',
        parallel_seed_policy='Four independent IRIS calls per round; update union before resampling',
        scope='One continued cover per method, fixed-base seven-joint semantics; no across-seed statistical claim',
        confidence_note='Per-call IRIS delta unchanged for comparison; no experiment-wide probabilistic safety claim',
        metadata_correction=None)
    write_json(out/'protocol.json',protocol);write_json(out/'config.json',config)
    hashes={}
    for f in ['src/iterative_visibility_cover.py','src/visibility_clique_experiment.py','src/arm_cspace_research.py','src/joint_cspace_research.py','src/build_scene.py','src/collision_checker.py','src/arm_boundary_semantics.py','src/iris_region_generator.py']:
        dest=out/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);hashes[f]=hashlib.sha256(Path(f).read_bytes()).hexdigest()
    write_json(out/'source_sha256.json',hashes)
    scene=make_scene(config,semantic=True);checker,param,lo,hi=setup(scene,parallelism=4)
    monitor,_=feasible_samples(checker,param,lo,hi,500,90300)
    holdout,_=feasible_samples(checker,param,lo,hi,2000,90301)
    queries=np.load(old/'points.npz')['heldout'][24:224]
    np.savez_compressed(out/'points.npz',monitor=monitor,heldout=holdout,query_points=queries)
    original=make_scene(config);oracle=ArmBoundaryOracle(original);summary={};history={}
    for method in ['uniform','vcc']:
        records=json.loads((old/f'{method}_regions.json').read_text());regions=to_regions(records)
        initial_mask=covered(holdout,regions)
        history[method]=[dict(round=1,accepted_regions=len(regions),monitor_count=int(covered(monitor,regions).sum()),test_count=int(initial_mask.sum()),cached=True)]
        began=time.perf_counter();additional_calls=0;stop_reason='round_budget'
        for round_index in [2,3]:
            if covered(monitor,regions).mean()>=.8:
                stop_reason='monitor_target';break
            folder=out/f'{method}_round_{round_index}';folder.mkdir()
            write_json(out/'run_status.json',dict(status='running',method=method,round=round_index))
            t=time.perf_counter();points,trials=feasible_samples(checker,param,lo,hi,192,92200+round_index,regions)
            np.save(folder/'construction.npy',points)
            details=dict(sampling_seconds=time.perf_counter()-t,sampling_trials=trials)
            if method=='vcc':
                proposals,graph_info=clique_proposals(checker,param,points,folder);details.update(graph_info)
            else:proposals=[dict(seed=q.tolist(),A=None) for q in points[:4]]
            write_json(folder/'proposals.json',proposals)
            if not proposals:
                stop_reason='no_valid_clique_proposals';break
            t=time.perf_counter();new=[]
            with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs=[pool.submit(parallel_seed,config,True,p['seed'],100*round_index+i,6,p['A']) for i,p in enumerate(proposals)]
                for job in jobs:
                    record=job.result();record['coverage_round']=round_index;new.append(record);records.append(record)
                    write_json(out/f'{method}_regions.json',records)
            additional_calls+=len(new);regions=to_regions(records)
            details.update(round=round_index,accepted_regions=len(regions),new_attempts=len(new),new_accepted=sum(r['accepted'] for r in new),
                inflation_wall_seconds=time.perf_counter()-t,monitor_count=int(covered(monitor,regions).sum()),test_count=int(covered(holdout,regions).sum()))
            history[method].append(details);write_json(folder/'summary.json',details);write_json(out/'history.json',history)
            print('ROUND',method,round_index,'regions',len(regions),'monitor',details['monitor_count'],'test',details['test_count'],flush=True)
        construction_wall=time.perf_counter()-began
        if covered(monitor,regions).mean()>=.8:stop_reason='monitor_target'
        write_json(out/f'{method}_regions.json',records)
        plans=[]
        pairs=[(np.array(protocol['start']),np.array(protocol['goal']))]+[(queries[2*i],queries[2*i+1]) for i in range(100)]
        for i,(start,goal) in enumerate(pairs):
            try:plan=solve(regions,start,goal,original,oracle,True)
            except RuntimeError as exc:plan=dict(success=False,accepted=False,error=str(exc))
            plan.update(query_index=i,start=start.tolist(),goal=goal.tolist(),endpoints_covered=bool(covered([start,goal],regions).all()))
            write_json(out/f'{method}_query_{i:02}.json',plan)
            plans.append({k:v for k,v in plan.items() if k not in ['trajectory_samples','sample_times','segments']})
            print('QUERY',method,i,plan['accepted'],flush=True)
        summary[method]=dict(accepted_regions=len(regions),additional_calls=additional_calls,stop_reason=stop_reason,
            additional_construction_seconds=construction_wall,heldout_coverage_count=int(covered(holdout,regions).sum()),heldout_coverage_denominator=len(holdout),
            queries=plans,heldout_query_success_count=sum(q['accepted'] for q in plans[1:]),heldout_query_count=100)
        write_json(out/'summary.json',summary);write_json(out/'history.json',history)
        print('COMPLETE',method,summary[method]['heldout_query_success_count'],flush=True)
    write_json(out/'run_status.json',dict(status='completed'))

if __name__=='__main__':main()
