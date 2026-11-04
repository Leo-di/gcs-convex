"""Frozen whole-body room entry followed by composed object semantics."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from .scene_spec import PROJECT_ROOT, load_scene_config
from .semantic_ablation_matrix import bounded


def run(output, study_path=PROJECT_ROOT/'configs/room_entry_study.yaml'):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    study=load_scene_config(study_path);task=load_scene_config(PROJECT_ROOT/study['task'])
    manifest={}
    for p in sorted((PROJECT_ROOT/'src').glob('*.py')):
        dest=root/'source_snapshot'/p.relative_to(PROJECT_ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes());manifest[str(p.relative_to(PROJECT_ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
    jobs=[]
    for case in study['cases']:
        spec=copy.deepcopy(task);spec['start_base']=case['start_base']
        file=root/(case['id']+'.json');file.write_text(json.dumps(spec,indent=2));jobs.append((case,file))
    (root/'protocol.json').write_text(json.dumps({'study':study,'source_sha256':manifest,
        'scope':'Two predefined Room 1 starts; identical Room 2 handoff and manipulation task; fixed arm during navigation/carry; no generalization claim'},indent=2))
    rows=[]
    for case,file in jobs:
        current={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/'src').glob('*.py')}
        if current!=manifest:raise RuntimeError('Source changed during frozen entry study')
        folder=root/case['id']
        code,wall=bounded([sys.executable,'-m','src.stateful_transport_grounding','--config',str(file),
            '--output',str(folder/'run'),'--html'],folder,study['worker_timeout_s'])
        row=dict(case=case['id'],exit=code,wall_seconds=wall,accepted=False)
        result=folder/'run/result.json'
        if result.exists():
            row['result']=json.loads(result.read_text());row['accepted']=code==0 and row['result']['satisfied']
        entry=folder/'run/room_entry/acceptance.json'
        if entry.exists():row['entry_acceptance']=json.loads(entry.read_text())
        rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2))
        print(case['id'],row['accepted'],row.get('result',{}).get('room_entry_gate'),flush=True)
    from .semantic_plan import configuration
    paths=[]
    for case,_ in jobs:
        file=root/case['id']/'run/reference.json'
        if file.exists():paths.append([f for f in json.loads(file.read_text()) if f['mode']!='enter_room'])
    match={'available_cases':len(paths),'all_operation_frames_match':False}
    if len(paths)==len(jobs) and len({len(p) for p in paths})==1:
        error=max(float(np.max(abs(configuration(a)-configuration(b)))) for p in paths for a,b in zip(paths[0],p))
        match.update(maximum_configuration_error=error,all_operation_frames_match=error<1e-6)
    (root/'operation_matching.json').write_text(json.dumps(match,indent=2))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',type=Path,default=PROJECT_ROOT/'configs/room_entry_study.yaml')
    a=p.parse_args();run(a.output,a.config)
