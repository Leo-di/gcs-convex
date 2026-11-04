import copy
import numpy as np
import pytest
from pydrake.geometry import Cylinder, Box
from pydrake.math import RigidTransform, RotationMatrix
from src.object_exclusion import ObjectExclusionEvaluator, shape_box_residual, validate_exclusions
from src.build_scene import build_scene, set_robot_configuration
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config
from src.multiobject_grounding import target_config
from src.collision_checker import evaluate_collision_status


def zone(object_id='cup'):
    return dict(id='test',object_id=object_id,lower_xyz=[1.,.7,.9],upper_xyz=[1.2,1.,1.4],margin_m=.02)


def test_whole_shape_is_checked_even_when_center_is_outside():
    z=zone();shape=Cylinder(.05,.18)
    assert .96 < z['lower_xyz'][0]
    assert shape_box_residual(shape,RigidTransform([.96,.85,1.1]),np.array(z['lower_xyz']),np.array(z['upper_xyz']),.02) < 0
    assert shape_box_residual(shape,RigidTransform([.90,.85,1.1]),np.array(z['lower_xyz']),np.array(z['upper_xyz']),.02) > 0


def test_rotation_height_and_translation_bound():
    shape=Box(.1,.1,.5);lo=np.array([1.,.7,.9]);hi=np.array([1.2,1.,1.4])
    upright=RigidTransform([.8,.85,1.1])
    rotated=RigidTransform(RotationMatrix.MakeYRotation(np.pi/2),[.8,.85,1.1])
    assert shape_box_residual(shape,upright,lo,hi)>0
    assert shape_box_residual(shape,rotated,lo,hi)<0
    rng=np.random.default_rng(25)
    for _ in range(30):
        a=rng.uniform([.5,.3,.7],[1.5,1.4,1.8]);delta=rng.normal(0,.03,3)
        ra=shape_box_residual(shape,RigidTransform(a),lo,hi)
        rb=shape_box_residual(shape,RigidTransform(a+delta),lo,hi)
        assert abs(ra-rb)<=np.linalg.norm(delta)+1e-12


def test_only_named_object_is_restricted_even_after_active_object_switch():
    spec=load_scene_config('configs/stateful_transport_grounding.yaml')
    spec['object_exclusions']=[zone('cup_b')]
    cfg=make_config(spec)
    for config in (cfg,target_config(cfg,'cup_b')):
        scene=build_scene(config=config,visualize=False);evaluator=ObjectExclusionEvaluator(scene)
        assert evaluator.evaluate({'cup_b':RigidTransform([.68,.85,.84]),'cup':RigidTransform([1.1,.85,1.1])})['satisfied']
        assert not evaluator.evaluate({'cup_b':RigidTransform([1.1,.85,1.1])})['satisfied']
        with pytest.raises(ValueError,match='Missing world pose'):
            evaluator.evaluate({'cup':RigidTransform()})


def test_evaluator_owns_geometry_when_built_from_a_temporary_scene():
    import gc
    spec=load_scene_config('configs/stateful_transport_grounding.yaml')
    spec['object_exclusions']=[zone()]
    evaluator=ObjectExclusionEvaluator(build_scene(config=make_config(spec),visualize=False))
    gc.collect()
    result=evaluator.evaluate({'cup':RigidTransform([.96,.85,1.1])})
    assert result['minimum_residual_m']==pytest.approx(-.03)


def test_virtual_region_does_not_become_a_robot_collision_obstacle():
    spec=load_scene_config('configs/stateful_transport_grounding.yaml');cfg=make_config(spec)
    base=[.45,0.,0.];arm=cfg['pick_task']['navigation_arm_q']
    plain=build_scene(config=cfg,visualize=False)
    # This zone contains the base center, but excludes the cup's position.
    cfg['object_exclusions']=[dict(id='base_area',object_id='cup',lower_xyz=[.2,-.3,0.],upper_xyz=[.8,.3,.5],margin_m=0.)]
    semantic=build_scene(config=cfg,visualize=False)
    for scene in (plain,semantic):set_robot_configuration(scene,base,arm,scene.gripper_open_positions)
    a=evaluate_collision_status(plain,base,arm_q=arm)
    b=evaluate_collision_status(semantic,base,arm_q=arm)
    assert a.collision_free==b.collision_free and a.minimum_signed_distance==pytest.approx(b.minimum_signed_distance)
    assert ObjectExclusionEvaluator(semantic).current(semantic)['satisfied']


def test_payload_cover_oracle_accounts_for_virtual_exclusion():
    from src.grasp_planner import plan_pick
    from src.payload_translation_bridge import PayloadTranslationOracle
    spec=load_scene_config('configs/object_exclusion_transport.yaml')
    cfg=make_config(spec)
    scene=build_scene(config=cfg,visualize=False)
    base=np.array(spec['actions'][0]['pickup_base']);pick=plan_pick(scene,base)
    set_robot_configuration(scene,base,pick.grasp_q,scene.gripper_closed_positions)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse() @ scene.cup_initial_pose
    constrained=PayloadTranslationOracle(scene,pick.lift_q,scene.gripper_closed_positions,0.,X)
    assert constrained.query([.95,-.1])[0]<0
    plain_cfg=copy.deepcopy(cfg);plain_cfg['object_exclusions']=[]
    plain=build_scene(config=plain_cfg,visualize=False)
    geometry=PayloadTranslationOracle(plain,pick.lift_q,plain.gripper_closed_positions,0.,X)
    assert geometry.query([.95,-.1])[0]>0


@pytest.mark.parametrize('change',[{'object_id':'unknown'},{'margin_m':-1.},{'upper_xyz':[0.,0.,0.]},{'lower_xyz':[np.nan,0.,0.]}])
def test_invalid_exclusions_fail_closed(change):
    z=zone();z.update(change)
    with pytest.raises(ValueError):validate_exclusions([z],{'cup'})
