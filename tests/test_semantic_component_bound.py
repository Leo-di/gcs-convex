import numpy as np
from src.build_scene import build_scene
from src.scene_spec import load_scene_config
from src.green_room_gcs import room_oracle, semantic_box_bound
from src.whole_robot_room import arm_point_reach_bound


def test_fixed_base_margin_is_not_penalized_for_arm_motion():
    cfg=load_scene_config('configs/scene_semantic_choice_v6.yaml')
    scene=build_scene(config=cfg,visualize=False)
    oracle=room_oracle(scene)
    arm=np.asarray(cfg['pick_task']['navigation_arm_q'])
    base=np.array([.45,0,0])
    reach=arm_point_reach_bound(scene,oracle)
    lo,hi=arm-.005,arm+.005
    bound=semantic_box_bound(oracle,base,lo,hi,scene.gripper_open_positions,reach)
    assert bound['center_residual_m']-bound['point_displacement_bound_m'] < 0
    assert abs(bound['whole_box_semantic_lower_bound_m']-.03)<1e-10
    # Independent samples check the bound against exact shape support.
    for q in np.random.default_rng(20261019).uniform(lo,hi,(100,7)):
        actual=oracle.query(np.r_[base,q])
        assert actual['residual_m'] >= bound['whole_box_semantic_lower_bound_m']-1e-10
    # A positive base margin must not hide an arm outside the room.
    found=False
    for q in np.random.default_rng(7).uniform(-1.5,1.5,(100,7)):
        actual=oracle.query(np.r_[base,q])
        if actual['arm_gripper_residual_m']<0:
            b=semantic_box_bound(oracle,base,q-.001,q+.001,scene.gripper_open_positions,reach)
            assert b['whole_box_semantic_lower_bound_m']<0
            found=True
            break
    assert found
