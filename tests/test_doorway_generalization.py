import copy
import numpy as np
import pytest
from src.scene_spec import load_scene_config
from src.planner_randomness import planning_seed,seeded_config
from src.doorway_generalization import sample_scenes


def test_legacy_seeds_and_distinct_explicit_streams():
    cfg=load_scene_config('configs/scene_doorway_table.yaml');before=copy.deepcopy(cfg)
    assert planning_seed(cfg,'ik',1701)==1701
    a=seeded_config(cfg,10);b=seeded_config(cfg,20)
    assert cfg==before and a['planning_seeds']!=b['planning_seeds']
    assert len(set(a['planning_seeds'].values()))==6
    assert a['green_room']['iris']['seed']==2010
    assert a['green_room']['random_seed']==3010
    for invalid in [-1,True,1.5,2**31]:
        with pytest.raises(ValueError):planning_seed({'planning_seeds':{'ik':invalid}},'ik',1701)


def test_iid_scene_generation_reproducible_and_unmodified():
    spec=load_scene_config('configs/doorway_generalization.yaml');before=copy.deepcopy(spec)
    a=sample_scenes(spec);b=sample_scenes(spec);assert a==b and spec==before
    assert len(a)==12 and len({r['id'] for r,_ in a})==12
    assert len({r['family'] for r,_ in a})==3
    assert any(r['pillar'] for r,_ in a)
    assert len({tuple(r['table_size']) for r,_ in a})==12
    for r,cfg in a:
        assert cfg['robot']['start']==r['start_base']
        assert .48<=r['door_half_width_m']<=.78
        table,cup=cfg['objects']['table'],cfg['objects']['cup']
        assert all(abs(cup['center'][i]-table['center'][i])+cup['radius']<=table['size'][i]/2 for i in (0,1))
    spec['scene_sampler_seed']+=1
    assert sample_scenes(spec)!=a
