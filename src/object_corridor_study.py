"""Frozen paired physical transport cases with/without carry corridors."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .scene_spec import PROJECT_ROOT, load_scene_config
from .semantic_ablation_matrix import bounded


def run(output, config_path=PROJECT_ROOT/'configs/object_corridor_study.yaml'):
    from .stateful_transport_grounding import make_config
    from .build_scene import build_scene
    from .object_corridor import ObjectCorridorEvaluator
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    study=load_scene_config(config_path)
    original=load_scene_config(PROJECT_ROOT/study['task'])
    manifest={}
    for p in sorted((PROJECT_ROOT/'src').glob('*.py')):
        dest=root/'source_snapshot'/p.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
        manifest[str(p.relative_to(PROJECT_ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    protocol={'study':study,'source_sha256':manifest,
        'comparison':'Identical world, initial configuration, placement, cover and execution within each pair; only corridor enforcement differs.',
        'scope':'One development layout and two predefined placement/table variations; not an independent broad scene distribution.'}
    jobs=[]
    # Freeze every condition and every method before launching the first job.
    for case in study['cases']:
        spec=copy.deepcopy(original)
        delta=np.array(case['destination_delta_xyz'])
        spec['actions'][0]['placements']=[(np.array(original['actions'][0]['placements'][0])+delta).tolist()]
        spec['destination_table']['center']=(np.array(original['destination_table']['center'])+delta).tolist()
        case_dir=root/case['id'];case_dir.mkdir()
        (case_dir/'spec.json').write_text(json.dumps(spec,indent=2))
        for method in ('without_corridor','with_corridor'):
            local=copy.deepcopy(spec);local['object_corridor_display']=copy.deepcopy(spec['object_corridors'])
            if method=='without_corridor':local['object_corridors']=[]
            file=case_dir/f'{method}.json';file.write_text(json.dumps(local,indent=2))
            jobs.append((case['id'],method,file,spec))
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for case,method,file,spec in jobs:
        current={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (PROJECT_ROOT/'src').glob('*.py')}
        if current!=manifest:raise RuntimeError('Source changed during frozen corridor study')
        folder=root/case/method
        code,wall=bounded([sys.executable,'-m','src.stateful_transport_grounding','--config',str(file),
            '--output',str(folder/'run'),'--html'],folder,study['worker_timeout_s'])
        row={'case':case,'method':method,'exit':code,'wall_seconds':wall,'accepted':False}
        result_file=folder/'run/result.json'
        if result_file.exists():
            row['result']=json.loads(result_file.read_text());row['accepted']=code==0 and row['result']['satisfied']
        physics=folder/'run/physics.json'
        if physics.exists():
            trace=json.loads(physics.read_text())['trace']
            evaluator=ObjectCorridorEvaluator(build_scene(config=make_config(spec),visualize=False))
            samples=[dict(time_s=f['time_s'],phase=f['phase'],**evaluator.evaluate(
                {k:v['pose'] for k,v in f['objects'].items()},f['phase'])) for f in trace]
            active=[f for f in samples if f['minimum_residual_m'] is not None]
            row['shadow_corridor']={'minimum_residual_m':min(f['minimum_residual_m'] for f in active),
                'samples':len(active),'violations':sum(not f['satisfied'] for f in active)}
            (folder/'shadow_trace.json').write_text(json.dumps(samples))
            search=json.loads((folder/'run/search.json').read_text())
            bridge=search['selected'][0]['report']['payload_bridge']
            row['carry_base_path_length_m']=bridge['base_path_length_m']
            row['simulated_seconds']=trace[-1]['time_s']
        rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2))
        print(case,method,row['accepted'],row.get('shadow_corridor'),flush=True)
    # Endpoint equality across the pair is measured from the saved references.
    endpoints=[]
    from .semantic_plan import configuration
    for case in study['cases']:
        files=[root/case['id']/method/'run/reference.json' for method in ('without_corridor','with_corridor')]
        row={'case':case['id'],'matched':False}
        if all(p.exists() for p in files):
            paths=[[f for f in json.loads(p.read_text()) if f['mode']=='carry'] for p in files]
            qerror=max(float(np.max(abs(configuration(paths[0][i])-configuration(paths[1][i])))) for i in (0,-1))
            perror=max(float(np.max(abs(np.array(paths[0][i]['world_objects']['cup'])-
                                           np.array(paths[1][i]['world_objects']['cup'])))) for i in (0,-1))
            row.update(matched=qerror<1e-6 and perror<1e-6,configuration_error=qerror,object_pose_error=perror)
        endpoints.append(row)
    (root/'endpoint_matching.json').write_text(json.dumps(endpoints,indent=2))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/object_corridor_study.yaml')
    a=p.parse_args();run(a.output,a.config)
