"""Prospective IID synthetic scenes, paired policies and repeated RNG streams."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from .scene_spec import PROJECT_ROOT,load_scene_config
from .doorway_scope_study import METHODS,method_config
from .doorway_scope_matrix import snapshot,terminate_tree
from .planner_randomness import seeded_config


def sample_scenes(spec):
    d=spec['distribution'];rng=np.random.default_rng(spec['scene_sampler_seed']);original=load_scene_config(PROJECT_ROOT/spec['scene'])
    def draw(key):return float(rng.uniform(*d[key]))
    scenes=[]
    for i in range(spec['independent_scenes']):
        cfg=copy.deepcopy(original);family=str(rng.choice(d['families'],p=d['family_probabilities']));side=-1 if family=='near_lower' else 1
        width,depth=draw('table_width_m'),draw('table_depth_m');left=draw('far_table_left_edge_m' if family=='far_upper' else 'near_table_left_edge_m')
        cy=side*draw('table_abs_y_m');cx=left+width/2
        table=cfg['objects']['table'];table['size'][:2]=[width,depth];table['center'][:2]=[cx,cy]
        cup=cfg['objects']['cup'];cup['center'][:2]=[left+draw('cup_from_left_edge_m'),cy-side*(depth/2-draw('cup_from_front_edge_m'))]
        cfg['task_zones']['pickup']['center'][:2]=[cx,cy]
        cfg['robot']['start']=[draw('start_x_m'),draw('start_y_m'),draw('start_yaw_rad')]
        half=draw('door_half_width_m')
        for wall in cfg['world']['walls']:
            if wall['name'] in ('shared_wall_upper','shared_wall_lower'):
                wall['size'][1]=3-half;wall['center'][1]=(3+half)/2*(1 if wall['name'].endswith('upper') else -1)
            elif wall['name']=='doorway_header':wall['size'][1]=2*half
        pillar=None
        if rng.random()<d['room1_pillar_probability']:
            size=list(d['pillar_size_m']);pillar=dict(name='heldout_pillar',size=size,center=[draw('pillar_x_m'),draw('pillar_y_m'),size[2]/2])
            cfg['world']['walls'].append(pillar)
        cfg['semantic_refinement'].update(copy.deepcopy(spec['sampling']))
        for axis in (0,1):
            assert abs(cup['center'][axis]-table['center'][axis])+cup['radius'] <= table['size'][axis]/2
        assert left>.05 and abs(cy)+depth/2<1.4
        record=dict(id=f'scene_{i:02d}',family=family,table_center=table['center'],table_size=table['size'],
            cup_center=cup['center'],door_half_width_m=half,start_base=cfg['robot']['start'],pillar=pillar)
        scenes.append((record,cfg))
    return scenes


def freeze(root,spec):
    from .semantic_refinement import generate_candidates
    from .joint_grounding_study import validate_bridge_yaws
    scenes=sample_scenes(spec);root.mkdir(parents=True,exist_ok=False)
    manifest=snapshot(root);runs=[];records=[]
    for record,scene in scenes:
        candidates=generate_candidates(scene);validate_bridge_yaws(scene,candidates,'rotation_safe_gcs')
        assert candidates
        records.append(dict(record,candidate_count=len(candidates)))
        for trial,seed in enumerate(spec['planner_seeds']):
            folder=root/record['id']/f'trial_{trial}';(folder/'configs').mkdir(parents=True)
            cfg=seeded_config(scene,seed)
            protocol=dict(candidates=candidates,seed=cfg['planning_seeds']['navigation'],
                planning_seeds=cfg['planning_seeds'],scene_id=record['id'],family=record['family'],trial=trial,
                total_search_seconds=spec['total_search_seconds'],per_candidate_seconds=spec['per_candidate_seconds'],
                bridge_method='rotation_safe_gcs',methods=list(METHODS))
            (folder/'protocol.json').write_text(json.dumps(protocol,indent=2));(folder/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
            for method in METHODS:
                configfile=folder/'configs'/f'{method}.json';configfile.write_text(json.dumps(method_config(cfg,method),indent=2))
                runs.append(dict(scene=record['id'],family=record['family'],trial=trial,method=method,
                    folder=str(folder.relative_to(root)),config_sha256=hashlib.sha256(configfile.read_bytes()).hexdigest()))
    protocol=dict(spec=spec,scenes=records,runs=runs,
        scope='Prospective IID samples from the declared synthetic distribution. Same robot and room topology; no arbitrary-world claim.',
        development_evidence='Reports 84–91; these holdout instances were not solved before freezing.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));return protocol,manifest


def run(output,config,resume=False):
    root=Path(output).resolve()
    if resume:
        protocol=json.loads((root/'protocol.json').read_text());manifest=json.loads((root/'source_manifest.json').read_text())
    else:protocol,manifest=freeze(root,load_scene_config(config))
    spec=protocol['spec'];env=dict(os.environ,LCM_DEFAULT_URL='memq://',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    rows=json.loads((root/'comparison.json').read_text()) if (root/'comparison.json').exists() else []
    key=lambda r:(r['scene'],r['trial'],r['method'])
    finished={key(r) for r in rows};jobs=[j for j in protocol['runs'] if key(j) not in finished]
    def job(item):
        folder=root/item['folder'];method=item['method']
        assert all(hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()==v for p,v in manifest.items()),'Frozen source changed'
        assert hashlib.sha256((folder/'configs'/f'{method}.json').read_bytes()).hexdigest()==item['config_sha256']
        if (folder/method).exists():raise RuntimeError('Incomplete existing attempt needs explicit retained-run handling; do not overwrite')
        began=time.monotonic()
        with (folder/f'{method}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.joint_grounding_study','--output',str(folder),'--worker',method,'operation_first'],
                stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
            try:code=proc.wait(timeout=spec['total_search_seconds']+spec['outer_timeout_slack_seconds'])
            except subprocess.TimeoutExpired:terminate_tree(proc);code='timeout'
        file=folder/method/'operation_first/search_result.json';search=json.loads(file.read_text()) if file.exists() else {}
        return dict(item,process_exit=code,wall_seconds=time.monotonic()-began,search=search,success=bool(code==0 and search.get('success')))
    order={key(j):i for i,j in enumerate(protocol['runs'])}
    with ThreadPoolExecutor(max_workers=spec['parallel_workers']) as pool:
        futures=[pool.submit(job,j) for j in jobs]
        for future in as_completed(futures):
            row=future.result();rows.append(row);rows.sort(key=lambda r:order[key(r)])
            (root/'comparison.json').write_text(json.dumps(rows,indent=2))
            print(len(rows),'/',len(protocol['runs']),row['scene'],row['trial'],row['method'],row['success'],row['search'].get('selected'),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',default=str(PROJECT_ROOT/'configs/doorway_generalization.yaml'));p.add_argument('--resume',action='store_true')
    a=p.parse_args();run(a.output,a.config,a.resume)
