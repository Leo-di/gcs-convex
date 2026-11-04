import copy
import numpy as np
import pytest
from src.build_scene import build_scene
from src.scene_spec import load_scene_config
from scripts.certify_room_reference import certify_frames


@pytest.fixture(scope='module')
def scene():return build_scene(config=load_scene_config('configs/scene_semantic_choice_v6.yaml'),visualize=False)


def frame(scene,arm,base=(.45,0,0)):
    return dict(base_q=list(base),arm_q=list(arm),gripper_q=scene.gripper_open_positions.tolist(),phase='approach_cup')


def test_detects_between_legal_endpoints(scene):
    frames=[frame(scene,[-.8,-.53,0,0,0,0,0]),frame(scene,[.8,-.53,0,0,0,0,0])]
    result=certify_frames(scene,frames)
    assert result['status']=='violated' and result['residual_m']<0


def test_bound_certifies_small_interior_motion_and_refuses_unresolved(scene):
    q=scene.config['pick_task']['navigation_arm_q'];a=frame(scene,q,base=(1.0,0,0));b=copy.deepcopy(a);b['arm_q'][0]+=.1
    r=certify_frames(scene,[a,b]);assert r['status']=='certified_model_bound' and r['minimum_interval_lower_bound_m']>0
    a=frame(scene,[-.8,-.53,0,0,0,0,0],base=(1.1,0,0));b=frame(scene,[.8,-.53,0,0,0,0,0],base=(1.1,0,0))
    assert certify_frames(scene,[a,b],max_depth=0)['status']=='unresolved'
    b['base_q'][0]+=.01;b['base_q'][2]+=.01
    assert certify_frames(scene,[a,b])['status']=='certified_model_bound'


def test_budget_and_gripper_motion(scene):
    q=scene.config['pick_task']['navigation_arm_q'];a=frame(scene,q,base=(1.,0,0));b=copy.deepcopy(a);b['gripper_q']=scene.gripper_closed_positions.tolist()
    assert certify_frames(scene,[a,b])['status']=='certified_model_bound'
    with pytest.raises(ValueError):certify_frames(scene,[],max_queries=1)
    with pytest.raises(ValueError):certify_frames(scene,[a,b],guard_m=0)
    b['gripper_q']=[10.,10.]
    with pytest.raises(ValueError,match='Finger positions'):certify_frames(scene,[a,b])
