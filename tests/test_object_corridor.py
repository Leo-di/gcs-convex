import copy
import gc
import numpy as np
import pytest
from pydrake.geometry import Cylinder, Box
from pydrake.math import RigidTransform, RotationMatrix
from src.object_corridor import ObjectCorridorEvaluator, shape_containment_residual, validate_corridors
from src.build_scene import build_scene, set_robot_configuration
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config, audit_action


def spec():
    return load_scene_config('configs/object_corridor_transport.yaml')


def test_containment_checks_shape_rotation_height_and_margin():
    lo, hi = [0., 0., 0.], [1., 1., 1.]
    assert shape_containment_residual(Cylinder(.05,.2),RigidTransform([.5,.97,.5]),lo,hi,.02) == pytest.approx(-.04)
    box=Box(.1,.1,.5)
    assert shape_containment_residual(box,RigidTransform([.15,.5,.5]),lo,hi)>0
    assert shape_containment_residual(box,RigidTransform(RotationMatrix.MakeYRotation(np.pi/2),[.15,.5,.5]),lo,hi)<0
    assert shape_containment_residual(box,RigidTransform([.5,.5,.9]),lo,hi)<0


def test_phase_scope_full_volume_and_temporary_scene_lifetime():
    evaluator=ObjectCorridorEvaluator(build_scene(config=make_config(spec()),visualize=False))
    gc.collect()
    pose={'cup':RigidTransform([.96,.69,1.08])}
    assert evaluator.evaluate(pose,'move_first_cup.carry')['satisfied']
    # Center is inside the middle cell; full radius plus margin is outside.
    pose={'cup':RigidTransform([.96,.75,1.08])}
    assert not evaluator.evaluate(pose,'move_first_cup.carry')['satisfied']
    assert evaluator.evaluate({},'move_first_cup.lift')['minimum_residual_m'] is None
    assert evaluator.evaluate({},'other_action.carry')['satisfied']
    with pytest.raises(ValueError,match='Missing world pose'):
        evaluator.evaluate({},'move_first_cup.carry')
    with pytest.raises(ValueError,match='Explicit phase'):
        evaluator.evaluate(pose,None)


def test_union_seams_are_conservative_and_translation_lipschitz():
    s=spec();s['object_corridors'][0]['margin_m']=0.
    s['object_corridors'][0]['cells']=[dict(lower_xyz=[0.,0.,0.],upper_xyz=[1.,1.,2.]),
                                     dict(lower_xyz=[1.,0.,0.],upper_xyz=[2.,1.,2.])]
    e=ObjectCorridorEvaluator(build_scene(config=make_config(s),visualize=False))
    def value(p):return e.evaluate({'cup':RigidTransform(p)},'move_first_cup.carry')['minimum_residual_m']
    # A cup centered on a touching seam lies in the union, but neither cell.
    assert value([1.,.5,1.])<0
    rng=np.random.default_rng(25)
    for _ in range(50):
        a=rng.uniform([0.,0.,.5],[2.,1.,1.5]);d=rng.normal(0,.08,3)
        assert abs(value(a)-value(a+d))<=np.linalg.norm(d)+1e-12


def test_payload_oracle_applies_corridor_before_planning():
    from src.grasp_planner import plan_pick
    from src.payload_translation_bridge import PayloadTranslationOracle
    s=spec();cfg=make_config(s);scene=build_scene(config=cfg,visualize=False)
    base=np.array(s['actions'][0]['pickup_base']);pick=plan_pick(scene,base)
    set_robot_configuration(scene,base,pick.grasp_q,scene.gripper_closed_positions)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@scene.cup_initial_pose
    with pytest.raises(ValueError,match='action ID'):
        PayloadTranslationOracle(scene,pick.lift_q,scene.gripper_closed_positions,0.,X)
    constrained=PayloadTranslationOracle(scene,pick.lift_q,scene.gripper_closed_positions,0.,X,action_id=s['actions'][0]['id'])
    assert constrained.query([.95,-.1])[0]<0
    plain_cfg=copy.deepcopy(cfg);plain_cfg['object_corridors']=[]
    plain=build_scene(config=plain_cfg,visualize=False)
    geometry=PayloadTranslationOracle(plain,pick.lift_q,plain.gripper_closed_positions,0.,X)
    assert geometry.query([.95,-.1])[0]>0
    assert constrained.query([.95,-.2])[0]>0


@pytest.mark.parametrize('change',[{'object_id':'unknown'},{'action_id':'unknown'}, {'mode':'lift'},
    {'margin_m':-1.}, {'cells':[]}, {'cells':[dict(lower_xyz=[0,0,0],upper_xyz=[1,1,np.nan])]}])
def test_bad_corridors_fail_closed(change):
    s=spec();s['object_corridors'][0].update(change)
    with pytest.raises(ValueError):make_config(s)


def test_corridor_keeps_named_identity_after_active_object_switch():
    from src.multiobject_grounding import target_config
    s=spec();s['objects']['cup_b']=[.68,.85,.84]
    s['object_corridors'][0]['object_id']='cup_b';s['actions'][0]['object_id']='cup_b'
    cfg=make_config(s)
    for local in (cfg,target_config(cfg,'cup_b')):
        e=ObjectCorridorEvaluator(build_scene(config=local,visualize=False))
        assert e.evaluate({'cup_b':RigidTransform([.96,.69,1.08]),'cup':RigidTransform()},'move_first_cup.carry')['satisfied']


def test_all_collision_shapes_must_fit_the_same_cell():
    s=spec();s['object_corridors'][0]['margin_m']=0.
    s['object_corridors'][0]['cells']=[dict(lower_xyz=[0.,0.,0.],upper_xyz=[.9,1.,2.]),
                                     dict(lower_xyz=[1.1,0.,0.],upper_xyz=[2.,1.,2.])]
    e=ObjectCorridorEvaluator(build_scene(config=make_config(s),visualize=False))
    e.geometry['cup']=[(Cylinder(.05,.1),RigidTransform([-.4,0.,0.])),
                       (Cylinder(.05,.1),RigidTransform([.4,0.,0.]))]
    assert not e.evaluate({'cup':RigidTransform([1.,.5,1.])},'move_first_cup.carry')['satisfied']


def test_phase_audit_rejects_illegal_carry_entry_without_restricting_hold():
    from src.grasp_planner import plan_pick
    s=spec()
    # Both boxes exclude the source cup; this is an invalid carry entry.
    s['object_corridors'][0]['cells']=[dict(lower_xyz=[.8,.5,.9],upper_xyz=[1.6,1.,1.3])]
    scene=build_scene(config=make_config(s),visualize=False)
    base=np.array(s['actions'][0]['pickup_base']);pick=plan_pick(scene,base)
    set_robot_configuration(scene,base,pick.grasp_q,scene.gripper_closed_positions)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@scene.cup_initial_pose
    f=dict(base_q=base.tolist(),arm_q=pick.lift_q.tolist(),gripper_q=scene.gripper_closed_positions.tolist(),
        mode='carry',phase='move_first_cup.carry',object_id='cup',
        world_objects={'cup':scene.cup_initial_pose.GetAsMatrix4().tolist()})
    checked=audit_action(scene,[f],X)
    assert not checked['satisfied'] and checked['minimum_object_corridor_residual_m']<0
    assert not checked['counterexamples'][0]['corridor']['satisfied']
    f.update(mode='hold',phase='move_first_cup.hold')
    assert audit_action(scene,[f],X)['minimum_object_corridor_residual_m'] is None


@pytest.mark.parametrize('endpoint',['start','goal'])
def test_bad_bridge_boundaries_rejected_before_cover(monkeypatch,tmp_path,endpoint):
    import src.payload_translation_bridge as bridge
    class Oracle:
        def __init__(self,*args,**kwargs):pass
        def query(self,xy):
            bad=xy[0]==(0. if endpoint=='start' else 1.)
            return (-.01 if bad else 1.),1.,None
    monkeypatch.setattr(bridge,'PayloadTranslationOracle',Oracle)
    def forbidden(*args,**kwargs):raise AssertionError('Cover must not run with illegal endpoints')
    monkeypatch.setattr(bridge,'make_cover',forbidden)
    with pytest.raises(RuntimeError,match=endpoint+' violates'):
        bridge.plan_payload_translation(None,[0,0,0],[1,0,0],None,None,None,
            dict(lower_xy=[-1,-1],upper_xy=[2,1],margin_m=.005,
                 max_radius_m=.25,spacing_m=.2),tmp_path/'audit.json')
