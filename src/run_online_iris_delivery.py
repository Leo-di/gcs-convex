"""Controlled endpoint holdout: online 10D IRIS inside a full semantic task."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .online_carry_iris import attach_carry_endpoints, sets
from .semantic_candidate_gcs import read_library, q, PICK_PHASES, PLACE_PHASES, build_and_solve, expand


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--library', type=Path, default=Path('outputs/results/semantic_candidates/library'))
    parser.add_argument('--candidate', default='x110')
    parser.add_argument('--seed-base', type=int, default=81000)
    parser.add_argument('--keep-endpoint-regions', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory to preserve earlier evidence')
    args.output.mkdir(parents=True)
    config, library = read_library(args.library)
    candidate = copy.deepcopy(library[args.candidate])
    original_cover = copy.deepcopy(candidate['cover']['regions'])
    pick = [r for r in candidate['rows'] if r['phase'] in PICK_PHASES]
    place = [r for r in candidate['rows'] if r['phase'] in PLACE_PHASES]
    start, goal = q(pick[-1]), q(place[0])
    disks = json.loads(Path('outputs/results/semantic_joint_gcs/shared_model.json').read_text())['empty_cover']
    digest = hashlib.sha256(json.dumps({'config':config,'candidate':candidate,'disks':disks},sort_keys=True).encode()).hexdigest()
    retained, removed, rejected = [], [], []
    for i, (record, region) in enumerate(zip(original_cover, sets(original_cover))):
        if record['independent_holdout_audit']['collisions']:
            rejected.append(i)
        elif not args.keep_endpoint_regions and (region.PointInSet(start) or region.PointInSet(goal)):
            removed.append(i)
        else:
            retained.append(record)
    protocol = {'experiment':'controlled_endpoint_region_holdout', 'candidate':args.candidate,
        'source_sha256':digest, 'seed_base':args.seed_base, 'start':start.tolist(),'goal':goal.tolist(),
        'withheld_endpoint_region_indices':removed,'rejected_cached_region_indices':rejected,
        'retained_regions':len(retained), 'original_regions':len(original_cover),
        'scope':'Existing action primitives and offline transit cover; deliberately withhold endpoint regions, grow official online IRIS-NP2, rerun semantic GCS. Not a new-scene generalization benchmark.',
        'solver':'Drake GraphOfConvexSets, Clarabel relaxation and restriction, 10 rounded paths',
        'new_region_audit':'1000 carrying checker samples plus seed and 1000 original-scene/grasp/semantic samples'}
    def save(name, data):
        (args.output/name).write_text(json.dumps(data,indent=2))
    save('protocol.json',protocol)
    save('input_candidate.json',candidate)
    save('scene_config_snapshot.json',config)
    candidate['cover']['regions'] = retained
    print('BASELINE: same graph without online endpoint growth',flush=True)
    before, _ = build_and_solve(disks,{args.candidate:candidate},[args.candidate],config)
    save('offline_result.json',before)
    print('ONLINE: growing attached-object 10D regions',flush=True)
    grown, report = attach_carry_endpoints(config,retained,pick[-1],start,goal,seed_base=args.seed_base)
    save('online_attachment.json',report)
    save('online_regions.json',grown)
    if not report['after']['region_graph_connected']:
        save('result.json',{'accepted':False,'failure':'online_region_graph_not_connected'})
        raise RuntimeError('Online cover is not connected; no replacement path will be fabricated')
    candidate['cover']['regions'] = grown
    candidate['cover']['number_of_regions'] = len(grown)
    print('ONLINE: solving full semantic GCS',flush=True)
    began=time.perf_counter()
    result, path = build_and_solve(disks,{args.candidate:candidate},[args.candidate],config)
    save('gcs_result.json',result)
    if path is None:
        save('result.json',{'accepted':False,'failure':result['status']})
        raise RuntimeError('GCS did not return a task path')
    new_indices=set(range(len(retained),len(grown)))
    used=[v['region'] for v in path if v['kind']=='carry' and v['region'] in new_indices]
    print('AUDIT: expand pick/place primitives and check full new trajectory',flush=True)
    audit = expand(path,{args.candidate:candidate},disks,config,args.output/'reference')
    result.update(audit=audit,accepted=audit['accepted'],used_online_regions=used,
                  planning_and_audit_seconds=time.perf_counter()-began,
                  attachment_seconds=report['wall_seconds'],physical_execution_run=False)
    save('result.json',result)
    if not audit['accepted']:
        raise RuntimeError('Full-task trajectory audit rejected the GCS output')
    print(json.dumps({'accepted':True,'used_online_regions':used,'frames':audit['frames']}),flush=True)


if __name__ == '__main__':
    main()
