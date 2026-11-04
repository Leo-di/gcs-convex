"""Matched physical cup transport with/without object-specific exclusion."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .scene_spec import PROJECT_ROOT, load_scene_config
from .semantic_ablation_matrix import bounded


def run(output, spec_path=PROJECT_ROOT/'configs/object_exclusion_transport.yaml'):
    from .stateful_transport_grounding import make_config
    from .build_scene import build_scene
    from .object_exclusion import ObjectExclusionEvaluator
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    spec=load_scene_config(spec_path)
    protocol={'spec':spec,'comparison':'Same initial world, action, placement, cover settings, execution and geometry. Only object exclusion enforcement changes.',
        'scope':'One development scene; fixed-arm/yaw 2D carrying; full 3D object collision-volume exclusion; no generalization claim',
        'source_sha256':{str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/'src').glob('*.py')}}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for p in (PROJECT_ROOT/'src').glob('*.py'):
        dest=root/'source_snapshot'/p.relative_to(PROJECT_ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    for method in ('without_exclusion','with_exclusion'):
        local=copy.deepcopy(spec);local['object_exclusion_display']=copy.deepcopy(spec['object_exclusions'])
        if method=='without_exclusion':local['object_exclusions']=[]
        (root/f'{method}.json').write_text(json.dumps(local,indent=2))
    rows=[]
    for method in ('without_exclusion','with_exclusion'):
        folder=root/method
        code,wall=bounded([sys.executable,'-m','src.stateful_transport_grounding','--config',str(root/f'{method}.json'),
            '--output',str(folder/'run'),'--html'],folder,300.)
        row={'method':method,'exit':code,'wall_seconds':wall,'accepted':False}
        file=folder/'run/result.json'
        if file.exists():
            result=json.loads(file.read_text());row['result']=result;row['accepted']=code==0 and result['satisfied']
        if (folder/'run/physics.json').exists():
            trace=json.loads((folder/'run/physics.json').read_text())['trace']
            # Evaluate both executed trajectories against the same semantic
            # specification, including the baseline where it was not enforced.
            shadow=ObjectExclusionEvaluator(build_scene(config=make_config(spec),visualize=False))
            samples=[dict(time_s=f['time_s'],phase=f['phase'],**shadow.evaluate({k:v['pose'] for k,v in f['objects'].items()})) for f in trace]
            row['shadow_exclusion']={'minimum_residual_m':min(v['minimum_residual_m'] for v in samples),
                'samples':len(samples),'violations':sum(not v['satisfied'] for v in samples)}
            (folder/'shadow_trace.json').write_text(json.dumps(samples))
            search=json.loads((folder/'run/search.json').read_text());chosen=search['selected'][0]['branch']
            bridge=json.loads((folder/f'run/branch_{chosen:03d}/payload_bridge.json').read_text())
            row['carry_base_path_length_m']=bridge['base_path_length_m']
            row['simulated_seconds']=trace[-1]['time_s']
        rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2))
        print(method,row['accepted'],row.get('shadow_exclusion'),flush=True)
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/object_exclusion_transport.yaml')
    a=p.parse_args();run(a.output,a.config)
