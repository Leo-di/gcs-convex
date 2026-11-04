import copy
import hashlib
import json
import numpy as np
import pytest
from scripts.object_facing_grasps import facing_yaw,object_facing_candidates
from src.scene_spec import load_scene_config
from src.semantic_refinement import generate_candidates


def test_positive_y_gripper_axis_faces_target_in_every_quadrant():
    for target in ([1.,2.],[-1.,2.],[-1.,-2.],[1.,-2.]):
        angle=facing_yaw([0,0],target)
        np.testing.assert_allclose([-np.sin(angle),np.cos(angle)],np.array(target)/np.linalg.norm(target),atol=1e-14)
    with pytest.raises(ValueError):facing_yaw([0,0],[0,0])


def test_heading_follows_scene_rotation_and_translation():
    base=np.array([.6,.1]);target=np.array([.2,-.7]);original=facing_yaw(base,target)
    for angle in [-2.4,-.3,1.2]:
        c,s=np.cos(angle),np.sin(angle);R=np.array([[c,-s],[s,c]]);shift=np.array([3.,-2.])
        moved=facing_yaw(R@base+shift,R@target+shift)
        assert abs(np.arctan2(np.sin(moved-original-angle),np.cos(moved-original-angle)))<1e-14


def test_candidate_targets_and_configuration_are_preserved():
    config=load_scene_config('configs/scene_doorway_table.yaml')
    config['semantic_refinement']['grasp_rpy_candidates']=[[0,0,-.15],[0,0,0],[0,0,.15]]
    config['semantic_refinement']['base_yaw_candidates']=[-.3,0,.3]
    before=copy.deepcopy(config)
    a=object_facing_candidates(config)
    assert a==object_facing_candidates(config) and config==before
    assert len({r['id'] for r in a})==len(a)
    assert all(r['grasp_heading_mode']=='object_facing' for r in a)
    assert any(abs(r['grasp_rpy'][2])>.3 for r in a)
    for old,new in zip(generate_candidates(config),a):
        assert old['id']==new['id'] and old['base_q']==new['base_q']
        assert old['grasp_position_offset_m']==new['grasp_position_offset_m']
        assert old['grasp_rpy'][2]==new['grasp_yaw_offset_rad']
    config['gripper']['grasp_point_in_body']=[.1,0,0]
    with pytest.raises(ValueError,match='positive local Y'):object_facing_candidates(config)


def test_freeze_records_adapter_candidates_and_source_before_solving(tmp_path):
    from scripts.run_object_facing_holdout import freeze_facing
    spec=load_scene_config('configs/object_facing_holdout.yaml');spec['independent_scenes']=1;spec['planner_seeds']=[31001]
    root=tmp_path/'cohort';protocol,manifest=freeze_facing(root,spec)
    assert protocol['candidate_heading_adapter']=='object_facing_upright_cup'
    assert len(protocol['runs'])==3 and not (root/'comparison.json').exists()
    folder=root/'scene_00/trial_0'
    local=json.loads((folder/'protocol.json').read_text())
    assert all(c['grasp_heading_mode']=='object_facing' for c in local['candidates'])
    assert json.loads((folder/'source_manifest.json').read_text())==manifest
    for name in ('scripts/object_facing_grasps.py','scripts/run_object_facing_holdout.py'):
        assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==manifest[name]
