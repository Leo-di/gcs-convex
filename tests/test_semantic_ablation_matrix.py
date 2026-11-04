import copy
import pytest
from src.scene_spec import load_scene_config
from src.semantic_ablation_matrix import doorway_config


def test_frozen_door_perturbation_changes_aperture_without_moving_other_walls():
    cfg=load_scene_config('configs/scene_doorway_table.yaml');before=copy.deepcopy(cfg)
    spec=load_scene_config('configs/semantic_ablation_study.yaml')
    altered=doorway_config(cfg,spec['doorway']['layouts'][2])
    assert cfg==before
    walls={w['name']:w for w in altered['world']['walls']}
    upper=walls['shared_wall_upper'];lower=walls['shared_wall_lower']
    assert upper['center'][1]-upper['size'][1]/2==pytest.approx(.55)
    assert lower['center'][1]+lower['size'][1]/2==pytest.approx(-.55)
    assert altered['objects']==cfg['objects']
    assert spec['policies']==['geometry_only','goal_only','full_motion']
