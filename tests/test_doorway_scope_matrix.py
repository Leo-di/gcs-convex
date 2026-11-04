import copy
import json
import pytest
from src.scene_spec import load_scene_config
from src.doorway_scope_matrix import case_config, freeze
from src.doorway_scope_study import METHODS


def test_cases_preserve_original_and_supported_cup():
    spec=load_scene_config('configs/doorway_scope_matrix.yaml');original=load_scene_config(spec['scene']);before=copy.deepcopy(original)
    for case in spec['cases']:
        cfg=case_config(original,case,spec['sampling'])
        assert cfg['robot']['start']==case['start_base']
        assert cfg['objects']['table']['center'][0]==pytest.approx(original['objects']['table']['center'][0]+case['table_dx'])
        assert cfg['objects']['cup']['center'][1]==pytest.approx(original['objects']['cup']['center'][1]+case['cup_dy'])
    assert original==before


def test_invalid_scene_offset_rejected():
    spec=load_scene_config('configs/doorway_scope_matrix.yaml');case=copy.deepcopy(spec['cases'][0]);case['cup_dx']=2
    with pytest.raises(ValueError,match='tabletop'):case_config(load_scene_config(spec['scene']),case,spec['sampling'])
    case['cup_dx']=float('nan')
    with pytest.raises(ValueError,match='Finite'):case_config(load_scene_config(spec['scene']),case,spec['sampling'])


def test_freeze_all_methods_before_run(tmp_path):
    spec=load_scene_config('configs/doorway_scope_matrix.yaml');spec['cases']=spec['cases'][:1]
    root=tmp_path/'matrix';manifest=freeze(root,spec)
    assert manifest and (root/'source_manifest.json').exists()
    protocol=json.loads((root/'nominal/protocol.json').read_text())
    assert len(protocol['candidates'])==protocol['candidate_count']>64
    configs=[json.loads((root/'nominal/configs'/f'{m}.json').read_text()) for m in METHODS]
    for key in ('objects','world','robot','semantic_refinement'):
        assert all(cfg[key]==configs[0][key] for cfg in configs)
    assert len({c['base_q'][2] for c in protocol['candidates']})==3
    assert len({tuple(c['grasp_rpy']) for c in protocol['candidates']})==3


def test_duplicate_case_rejected_before_directory_created(tmp_path):
    spec=load_scene_config('configs/doorway_scope_matrix.yaml');spec['cases']*=2
    with pytest.raises(ValueError,match='IDs'):freeze(tmp_path/'matrix',spec)
    assert not (tmp_path/'matrix').exists()
