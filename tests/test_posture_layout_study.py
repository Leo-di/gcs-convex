"""Frozen variations must align physical meshes, supports, cups and controls."""
import copy
import numpy as np
from src.scene_spec import load_scene_config
from src.posture_layout_study import RECIPES,make_case,assert_consistency,normalized_task
from src.stateful_transport_grounding import validate_task


def test_all_new_layouts_keep_mesh_support_metadata_and_controls_consistent():
    scene=load_scene_config('configs/scene_replicacad_human_passage.yaml');task=load_scene_config('configs/replicacad_human_passage_transport.yaml')
    before=copy.deepcopy((scene,task))
    for recipe in RECIPES:
        cfg,tasks=make_case(scene,task,recipe,'frozen_scene.json');assert_consistency(cfg,tasks)
        for t in tasks.values():validate_task(t)
        assert normalized_task(tasks['automatic'])==normalized_task(tasks['manual'])
        assert tasks['automatic']['human_passage']==task['human_passage']
    assert (scene,task)==before


def test_variations_are_not_rigid_translations_and_change_cup_relative_reach():
    scene=load_scene_config('configs/scene_replicacad_human_passage.yaml');task=load_scene_config('configs/replicacad_human_passage_transport.yaml')
    base={f['name']:np.array(f['center']) for f in scene['asset_fixtures']}
    signatures=[]
    for recipe in RECIPES:
        cfg,tasks=make_case(scene,task,recipe,'scene.json')
        shifts=np.array([np.array(f['center'])-base[f['name']] for f in cfg['asset_fixtures']])
        assert np.ptp(shifts[:,0])>.1 and np.ptp(shifts[:,1])>.1
        t=tasks['automatic'];standoffs=[t['objects'][a['object_id']][1]-a['pickup_base'][1] for a in t['actions']]
        assert not np.allclose(standoffs,.85)
        signatures.append(tuple(np.round(shifts.ravel(),5)))
    assert len(set(signatures))==3
