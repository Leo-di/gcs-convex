"""Freeze an independent holdout using object-facing upright-cup grasps.

Reuses the unchanged phase/bridge/physical acceptance pipeline. Only the
candidate orientation adapter changes; no successful endpoint bypasses checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.doorway_generalization import freeze,run
from src.scene_spec import PROJECT_ROOT,load_scene_config
from scripts.object_facing_grasps import object_facing_candidates


def freeze_facing(root,spec):
    root=Path(root);protocol,manifest=freeze(root,spec)
    for scene in protocol['scenes']:
        for trial in range(len(spec['planner_seeds'])):
            folder=root/scene['id']/f'trial_{trial}'
            cfg=json.loads((folder/'configs/full_motion.json').read_text())
            candidates=object_facing_candidates(cfg)
            local=json.loads((folder/'protocol.json').read_text())
            local.update(candidates=candidates,candidate_heading_adapter='object_facing_upright_cup',
                heading_scope='Positive local-Y grasp point faces target from docking XY; yaw-only library offsets; same complete acceptance pipeline')
            (folder/'protocol.json').write_text(json.dumps(local,indent=2))
            scene['candidate_count']=len(candidates)
    for script in (Path(__file__),Path(__file__).with_name('object_facing_grasps.py')):
        rel=script.resolve().relative_to(PROJECT_ROOT);data=script.read_bytes()
        manifest[str(rel)]=hashlib.sha256(data).hexdigest()
        dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    for scene in protocol['scenes']:
        for trial in range(len(spec['planner_seeds'])):
            (root/scene['id']/f'trial_{trial}'/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    protocol['candidate_heading_adapter']='object_facing_upright_cup'
    protocol['development_evidence']='Initial 12-scene holdout and fixed-scene heading diagnosis; all instances in this second cohort use a new frozen sampler seed.'
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    return protocol,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/object_facing_holdout.yaml');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not a.resume:freeze_facing(Path(a.output).resolve(),load_scene_config(a.config))
    run(a.output,a.config,resume=True)
