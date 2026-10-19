import copy
import pytest
from src.semantic_policy import accepts
from src.semantic_refinement import screen_candidates, accepted_result
from src.scene_spec import load_scene_config


def test_ablation_changes_only_semantic_gates():
    assert accepts('geometry_only', geometry=True, goals=False, motion=False)
    assert not accepts('goal_only', geometry=True, goals=False, motion=True)
    assert accepts('goal_only', geometry=True, goals=True, motion=False)
    assert not accepts('full_motion', geometry=True, goals=True, motion=False)
    assert accepts('full_motion', geometry=True, goals=True, motion=True)
    for policy in ['geometry_only','goal_only','full_motion']:
        assert not accepts(policy, geometry=False, goals=True, motion=True)


def test_geometry_baseline_can_keep_outside_dock():
    cfg=load_scene_config('configs/scene_green_room_iris.yaml')
    candidate={'id':'outside','base_q':[-1.,0.,0.], 'grasp_rpy':[0,0,0]}
    for policy in ['geometry_only','goal_only','full_motion']:
        cfg['semantic_policy']=policy
        row=screen_candidates(cfg,[candidate])[0]
        assert (row['status']=='eligible') == (policy=='geometry_only')
        assert row['dock_room_residual_m'] < 0


def test_baseline_planner_success_does_not_imply_semantic_success():
    result={'satisfied':True, 'semantic_task_satisfied':False,
            'reference_audit':{'satisfied':False,'policy_satisfied':True}}
    assert accepted_result(result,False)
    assert not result['semantic_task_satisfied']
