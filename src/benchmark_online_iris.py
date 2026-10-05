"""Fixed-query experiment: offline GCS versus query-local IRIS + GCS."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import time
import numpy as np
from .online_iris_attachment import attach_endpoints, connection_status
from .iterative_visibility_cover import to_regions
from .arm_cspace_research import solve
from .arm_boundary_semantics import ArmBoundaryOracle
from .joint_cspace_research import make_scene, write_json
from .validate_arm_cspace_artifacts import validate


def build_case(root, query):
    root=Path(root);source=Path(json.loads((root/'protocol.json').read_text())['source'])
    folder=root/f'case_{query:03}';folder.mkdir()
    config=json.loads((root/'config.json').read_text());write_json(folder/'config.json',config)
    points=np.load(root/'query_points.npy');start,goal=points[2*(query-1):2*query]
    for method in ['uniform','vcc']:
        write_json(folder/'status.json',dict(stage='attachment',method=method))
        offline=json.loads((source/f'{method}_regions.json').read_text())
        records,info=attach_endpoints(config,offline,start,goal,700000+10*query)
        write_json(folder/f'{method}_regions.json',records)
        write_json(folder/f'{method}_attachment.json',info)
        print('ATTACHED',query,method,info,flush=True)
    write_json(folder/'status.json',dict(stage='constructed'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--query-count',type=int,default=3)
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args()
    if not 1<=args.query_count<=100 or not 1<=args.workers<=24:
        parser.error('query-count must be 1..100 and workers 1..24')
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    source=Path('outputs/results/visibility_clique_multiseed_v1/seed_1701')
    config=json.loads((source/'config.json').read_text());write_json(root/'config.json',config)
    protocol=dict(source=str(source),queries=list(range(1,args.query_count+1)),workers=args.workers,selection='First query-count original random queries in original order; first three previously used in integration pilot; no result-based filtering',
        algorithm='Official Drake IrisNp2 and unchanged GcsTrajectoryOptimization',
        max_calls_per_query_method=2,attempts_per_uncovered_endpoint=1,endpoint_order=['start','goal'],
        candidate_index='700000+10*query+endpoint_index; same RNG arrangement for both methods',
        seed_radius_rad=.01,iris_configuration_space_margin_rad=.01,iris_particles=2000,iris_epsilon=.0025,iris_delta=.01/12,
        iris_iterations=6,separation_iterations=30,region_audit_samples=1000,
        baseline='Rerun original offline GCS on identical endpoints, with original cover unchanged',
        isolation='Fresh copy of offline cover for each query; no sharing of online regions or replacement of rejected attempts',
        timing=f'{args.workers} query construction jobs concurrent, each with serial endpoint calls; ALL baseline and online GCS runs serial after construction',
        scope=f'{args.query_count} paired queries conditional on one fixed cover seed 1701; same queries for both methods, not independent construction replicates; fixed-base seven joints, full-arm boundary predicate',
        safety='Finite region and trajectory audits; no continuous collision certificate')
    write_json(root/'protocol.json',protocol)
    np.save(root/'query_points.npy',np.load(source/'points.npz')['query_points'])
    files=['src/online_iris_attachment.py','src/benchmark_online_iris.py','src/arm_cspace_research.py','src/iterative_visibility_cover.py','src/joint_cspace_research.py','src/build_scene.py','src/arm_boundary_semantics.py','src/collision_checker.py','src/iris_region_generator.py']
    hashes={}
    for f in files:
        dst=root/'source_snapshot'/f;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dst);hashes[f]=hashlib.sha256(Path(f).read_bytes()).hexdigest()
    write_json(root/'source_sha256.json',hashes)
    write_json(root/'status.json',dict(stage='attachment'))
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for result in pool.map(build_case,[str(root)]*args.query_count,protocol['queries']):pass
    original=make_scene(config);oracle=ArmBoundaryOracle(original);points=np.load(root/'query_points.npy');rows=[]
    for query in protocol['queries']:
        folder=root/f'case_{query:03}';start,goal=points[2*(query-1):2*query];summary={}
        for method in ['uniform','vcc']:
            write_json(root/'status.json',dict(stage='evaluating',query=query,method=method))
            online_records=json.loads((folder/f'{method}_regions.json').read_text())
            offline_records=json.loads((source/f'{method}_regions.json').read_text());info=json.loads((folder/f'{method}_attachment.json').read_text())
            row=dict(query=query,method=method,attachment=info)
            for variant,records in [('offline',offline_records),('online',online_records)]:
                began=time.perf_counter()
                try:plan=solve(to_regions(records),start,goal,original,oracle,True)
                except RuntimeError as exc:plan=dict(success=False,accepted=False,error=str(exc))
                plan.update(query_index=query,start=start.tolist(),goal=goal.tolist(),planning_and_audit_wall_seconds=time.perf_counter()-began)
                plan.update(connection_status(to_regions(records),start,goal))
                plan['failure_category']=('accepted' if plan['accepted'] else 'endpoint_not_covered' if not plan['endpoints_covered'] else 'region_graph_disconnected' if not plan['region_graph_connected'] else 'trajectory_audit_failed' if plan.get('success') else 'gcs_solver_failed')
                label=method if variant=='online' else method+'_offline'
                write_json(folder/f'{label}_plan.json',plan);write_json(folder/f'{label}_regions.json',records)
                checks=validate(folder,[label])[label]
                assert not checks['checked'] or checks['passed'], checks
                write_json(folder/f'{label}_coefficient_audit.json',checks)
                row[variant]={k:v for k,v in plan.items() if k not in ['segments','sample_times','trajectory_samples']}
                if variant=='online':
                    write_json(folder/f'{method}_query_00.json',plan)
                    summary[method]=dict(queries=[row[variant]])
            row['online_total_wall_seconds']=info['attachment_wall_seconds']+row['online']['planning_and_audit_wall_seconds']
            rows.append(row);write_json(root/'summary.json',dict(protocol=protocol,rows=rows))
            print('EVALUATED',query,method,row['offline']['accepted'],row['online']['accepted'],flush=True)
        write_json(folder/'summary.json',summary);write_json(folder/'cliques.json',{})
        write_json(folder/'status.json',dict(stage='completed'))
    write_json(root/'status.json',dict(stage='completed'))

if __name__=='__main__':main()
