"""Three paired random-cover repeats, common anchors and untouched queries.

Official Drake visibility/MVCE/IRIS-NP2/GCS and exact clique ILP unchanged.
Parallel construction across three seeds; serial query timing afterwards.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import numpy as np
from .iterative_visibility_cover import feasible_samples,to_regions,clique_proposals
from .arm_cspace_research import setup,parallel_seed,solve
from .visibility_clique_experiment import covered
from .joint_cspace_research import make_scene,write_json
from .arm_boundary_semantics import ArmBoundaryOracle

SEEDS=(1701,1702,1703)


def method_order(seed):
    return ('uniform','vcc') if SEEDS.index(seed)%2==0 else ('vcc','uniform')


def construct_seed(root,seed):
    root=Path(root);out=root/f'seed_{seed}';out.mkdir(exist_ok=False)
    config=json.loads((root/'config.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    write_json(out/'config.json',config);write_json(out/'protocol.json',dict(protocol,construction_seed=seed))
    data=np.load(root/'points.npz');monitor=data['monitor'];heldout=data['heldout']
    np.savez_compressed(out/'points.npz',monitor=monitor,heldout=heldout,query_points=data['query_points'])
    anchors=json.loads((root/'anchors.json').read_text())
    scene=make_scene(config,semantic=True);checker,param,lo,hi=setup(scene,parallelism=4)
    history={};summary={}
    for method in method_order(seed):
        records=[dict(r,shared_cached_anchor=True,coverage_round=0) for r in anchors];regions=to_regions(records)
        history[method]=[dict(round=0,accepted_regions=2,monitor_count=int(covered(monitor,regions).sum()),test_count=int(covered(heldout,regions).sum()),cached=True)]
        began=time.perf_counter();calls=0;stop='round_budget'
        for round_id in [1,2]:
            if covered(monitor,regions).mean()>=.8:stop='monitor_target';break
            folder=out/f'{method}_round_{round_id}';folder.mkdir()
            write_json(out/'run_status.json',dict(status='constructing',method=method,round=round_id))
            t=time.perf_counter();points,trials=feasible_samples(checker,param,lo,hi,192,seed*100+round_id,regions)
            np.save(folder/'construction.npy',points)
            details=dict(sampling_seconds=time.perf_counter()-t,sampling_trials=trials)
            if method=='vcc':proposals,info=clique_proposals(checker,param,points,folder);details.update(info)
            else:proposals=[dict(seed=q.tolist(),A=None) for q in points[:4]]
            write_json(folder/'proposals.json',proposals)
            if not proposals:stop='no_valid_clique_proposals';break
            t=time.perf_counter();new=[]
            with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
                jobs=[pool.submit(parallel_seed,config,True,p['seed'],seed*100+round_id*10+i,6,p['A']) for i,p in enumerate(proposals)]
                for job in jobs:
                    r=job.result();r['coverage_round']=round_id;records.append(r);new.append(r)
                    write_json(out/f'{method}_regions.json',records)
            calls+=len(new);regions=to_regions(records)
            details.update(round=round_id,accepted_regions=len(regions),new_attempts=len(new),new_accepted=sum(r['accepted'] for r in new),
                inflation_wall_seconds=time.perf_counter()-t,monitor_count=int(covered(monitor,regions).sum()),test_count=int(covered(heldout,regions).sum()))
            history[method].append(details);write_json(folder/'summary.json',details);write_json(out/'history.json',history)
            print('ROUND',seed,method,round_id,'coverage',details['test_count'],'regions',len(regions),flush=True)
        if covered(monitor,regions).mean()>=.8:stop='monitor_target'
        summary[method]=dict(accepted_regions=len(regions),additional_calls=calls,construction_wall_seconds=time.perf_counter()-began,
            stop_reason=stop,heldout_coverage_count=int(covered(heldout,regions).sum()),heldout_coverage_denominator=len(heldout),
            iris_elapsed_sum_seconds=sum(r.get('iris_seconds',r.get('elapsed_to_error_seconds',0)) for r in records if not r['shared_cached_anchor']) if all('shared_cached_anchor' in r for r in records) else sum(r.get('iris_seconds',r.get('elapsed_to_error_seconds',0)) for r in records[2:]))
        write_json(out/f'{method}_regions.json',records);write_json(out/'construction_summary.json',summary);write_json(out/'history.json',history)
    write_json(out/'run_status.json',dict(status='constructed'))
    return seed


def evaluate_seed(root,seed):
    out=root/f'seed_{seed}';config=json.loads((root/'config.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
    original=make_scene(config);oracle=ArmBoundaryOracle(original);queries=np.load(root/'points.npz')['query_points']
    summary=json.loads((out/'construction_summary.json').read_text())
    pairs=[(np.array(protocol['start']),np.array(protocol['goal']))]+[(queries[2*i],queries[2*i+1]) for i in range(100)]
    for method in method_order(seed):
        regions=to_regions(json.loads((out/f'{method}_regions.json').read_text()));plans=[]
        for i,(start,goal) in enumerate(pairs):
            write_json(out/'run_status.json',dict(status='evaluating',method=method,query=i))
            try:plan=solve(regions,start,goal,original,oracle,True)
            except RuntimeError as exc:plan=dict(success=False,accepted=False,error=str(exc))
            plan.update(query_index=i,start=start.tolist(),goal=goal.tolist(),endpoints_covered=bool(covered([start,goal],regions).all()))
            write_json(out/f'{method}_query_{i:02}.json',plan)
            plans.append({k:v for k,v in plan.items() if k not in ['trajectory_samples','sample_times','segments']})
            print('QUERY',seed,method,i,plan['accepted'],flush=True)
        summary[method].update(queries=plans,heldout_query_success_count=sum(q['accepted'] for q in plans[1:]),heldout_query_count=100)
        write_json(out/'summary.json',summary)
        print('METHOD COMPLETE',seed,method,summary[method]['heldout_query_success_count'],flush=True)
    write_json(out/'run_status.json',dict(status='completed'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    root=parser.parse_args().output;root.mkdir(parents=True,exist_ok=False)
    old=Path('outputs/results/arm_cspace_research_v2')
    config=json.loads((old/'config.json').read_text());previous=json.loads((old/'protocol.json').read_text())
    anchors=json.loads((old/'semantic_regions.json').read_text())[:2]
    assert [r['candidate_index'] for r in anchors]==[0,1] and all(r['accepted'] for r in anchors)
    protocol={k:previous[k] for k in ['dimension','base','start','goal','lower','upper','region_particles','region_epsilon','region_delta','region_iterations','separation_iterations','configuration_space_margin_rad','holdout_per_region','semantic_margin_m','objective','velocity_bound_rad_s','continuity']}
    protocol.update(seeds=list(SEEDS),rounds=2,calls_per_round=4,max_calls_per_method_seed=8,visibility_points=192,
        monitor_seed=520001,heldout_seed=520002,query_seed=520003,monitor_points=500,test_points=5000,query_pairs=100,
        coverage_target=.8,construction_seed_formula='100*seed+round',iris_candidate_index='100*seed+10*round+index; IrisNp2 RNG=5100+candidate_index',
        concurrent_seed_jobs=3,workers_per_seed=4,visibility_threads_per_seed=4,
        query_timing='Serial, only after ALL construction workers have finished; methods counterbalanced by seed',
        construction_timing='Concurrent across seeds; descriptive wall time under this execution protocol, not isolated runtime',
        shared_anchor_source=str(old),method_orders={str(s):method_order(s) for s in SEEDS},
        scope='Three random cover expansions conditional on the same two cached anchors, same scene and same 100 held-out query pairs; not 300 independent construction trials',
        comparison='Fresh common-anchor experiment, not direct replication of prior continuation from method-specific five-region covers',
        algorithms=['Drake VisibilityGraph and MVCE','HiGHS exact maximum clique ILP','Drake IrisNp2','Drake GcsTrajectoryOptimization'],
        references=['https://arxiv.org/abs/2310.02875','https://arxiv.org/abs/2205.04422','https://arxiv.org/abs/2410.12649'],
        audit='1000 separate region audit samples; reject on any observed collision or semantic violation; dense trajectory and independent coefficient checks; no continuous safety certificate',
        confidence='Per-call IRIS settings unchanged; no experiment-wide probability or large-sample superiority claim',
        packages={k:importlib.metadata.version(k) for k in ['drake','numpy','scipy']})
    write_json(root/'protocol.json',protocol);write_json(root/'config.json',config);write_json(root/'anchors.json',anchors)
    hashes={}
    for f in ['src/multiseed_visibility_cover.py','src/iterative_visibility_cover.py','src/visibility_clique_experiment.py','src/arm_cspace_research.py','src/joint_cspace_research.py','src/build_scene.py','src/collision_checker.py','src/arm_boundary_semantics.py','src/iris_region_generator.py']:
        dest=root/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest);hashes[f]=hashlib.sha256(Path(f).read_bytes()).hexdigest()
    write_json(root/'source_sha256.json',hashes)
    scene=make_scene(config,semantic=True);checker,param,lo,hi=setup(scene)
    monitor,_=feasible_samples(checker,param,lo,hi,500,520001)
    heldout,_=feasible_samples(checker,param,lo,hi,5000,520002)
    query_points,_=feasible_samples(checker,param,lo,hi,200,520003)
    np.savez_compressed(root/'points.npz',monitor=monitor,heldout=heldout,query_points=query_points)
    write_json(root/'run_status.json',dict(status='constructing',seeds=list(SEEDS)))
    with ProcessPoolExecutor(max_workers=3,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(construct_seed,str(root),s) for s in SEEDS]
        for job in jobs:print('SEED CONSTRUCTED',job.result(),flush=True)
    for seed in SEEDS:
        write_json(root/'run_status.json',dict(status='evaluating',seed=seed))
        evaluate_seed(root,seed)
    write_json(root/'run_status.json',dict(status='completed'))

if __name__=='__main__':main()
