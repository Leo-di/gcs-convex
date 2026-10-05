"""Preserve an accepted nominal plan; otherwise try bounded mode-aware repairs."""
import copy
import json
from pathlib import Path
import time
from .semantic_candidate_gcs import read_library,q,PICK_PHASES,PLACE_PHASES,build_and_solve,expand
from .online_carry_iris import attach_carry_endpoints
from .online_carry_bridge import bridge
from .prepare_full_task_development import change_place
from .benchmark_semantic_baselines import atomic_json


def run(bundle,out,index):
    out=Path(out);out.mkdir(parents=True,exist_ok=True);begin=time.perf_counter()
    base_config=bundle['config'];cell=bundle['cell'];_,library=read_library(Path('outputs/results/semantic_candidates/library'))
    record={'cell':cell['id'],'method':'unified','planning_success':False,'end_to_end_success':False,
            'status':'planning','stages':[],'preparation_seconds_shared':bundle['preparation_seconds']}
    def save():
        record['planning_wall_seconds']=time.perf_counter()-begin;atomic_json(out/'record.json',record)
    save()
    for hi,height in enumerate([.22,.14,.30]):
        config=copy.deepcopy(base_config);config['delivery_task']['preplace_height']=height
        selected={};rejections={}
        if hi==0:
            selected={n:copy.deepcopy(bundle['candidates'][n]) for n in cell['allowed'] if n in bundle['candidates']}
        else:
            for name in cell['allowed']:
                candidate=copy.deepcopy(library[name]);candidate['cover']['regions']=[r for r in candidate['cover']['regions'] if not r['independent_holdout_audit']['collisions']]
                try:selected[name]=change_place(config,candidate,placement_base=cell['placement_base'])
                except RuntimeError as e:rejections[name]=str(e)
        stage={'height':height,'valid_candidates':list(selected),'primitive_rejections':rejections,'online':{},'bridges':{},'solves':[]}
        record['stages'].append(stage);save()
        if not selected:continue
        for j,(name,candidate) in enumerate(selected.items()):
            rows=candidate['rows'];pick=[r for r in rows if r['phase'] in PICK_PHASES];place=[r for r in rows if r['phase'] in PLACE_PHASES]
            regions,info=attach_carry_endpoints(config,candidate['cover']['regions'],pick[-1],q(pick[-1]),q(place[0]),seed_base=91000+100*index+2*j+10000*hi)
            candidate['cover']['regions']=regions;stage['online'][name]=info;save()
        for bridging in [False,True]:
            if bridging:
                for j,(name,candidate) in enumerate(selected.items()):
                    rows=candidate['rows'];pick=[r for r in rows if r['phase'] in PICK_PHASES];place=[r for r in rows if r['phase'] in PLACE_PHASES]
                    regions,info=bridge(config,candidate['cover']['regions'],pick[-1],q(pick[-1]),q(place[0]),seed_base=210000+100*index+10*j+10000*hi,max_calls=4)
                    candidate['cover']['regions']=regions;stage['bridges'][name]=info;save()
            directory=out/f'height_{hi}_bridge_{int(bridging)}';directory.mkdir(exist_ok=True)
            # Preserve the original baseline's exact initial-state source even when
            # its first library candidate is forbidden by the task policy.
            graph_candidates={**copy.deepcopy(bundle['candidates']),**selected} if hi==0 else selected
            result,path=build_and_solve(bundle['disks'],graph_candidates,list(selected),config,rounding_seed=0,max_rounded_paths=10)
            atomic_json(directory/'gcs.json',result);entry={'bridging':bridging,'status':result['status']};stage['solves'].append(entry)
            if path is None:save();continue
            audit=expand(path,graph_candidates,bundle['disks'],config,directory,visualize=False);entry['audit']=audit
            if audit['accepted']:
                record.update(planning_success=True,status='planned',reference=str(directory),selected_candidate=result['selected_candidate'],
                              accepted_height=height,accepted_bridge_stage=bridging)
                save();return record  # Do not replace a passing nominal plan with a repair.
            save()
    record.update(status='failed',failure='bounded_repairs_exhausted');save();return record
