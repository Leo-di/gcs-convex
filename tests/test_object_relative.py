import copy
import numpy as np
import pytest
from pydrake.math import RigidTransform, RotationMatrix
from src.build_scene import build_scene, set_robot_configuration
from src.object_relative import ObjectRelativeEvaluator
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config, initial_world, successor, world_config, audit_action


def task():
    return load_scene_config('configs/object_relative_transport.yaml')


@pytest.fixture
def evaluator():
    return ObjectRelativeEvaluator(build_scene(config=make_config(task()), visualize=False))


def test_full_volume_and_phase_activation(evaluator):
    ref = RigidTransform([1.42, .85, .84])
    poses = {'cup': ref, 'cup_b': RigidTransform([1.2, .95, 1.12])}
    assert evaluator.evaluate(poses, 'move_second_cup.carry')['satisfied']
    # Cup center is in the middle cell, but its radius and margin are not.
    poses['cup_b'] = RigidTransform([1.2, .88, 1.12])
    assert not evaluator.evaluate(poses, 'move_second_cup.carry')['satisfied']
    assert evaluator.evaluate({}, 'move_second_cup.hold')['minimum_residual_m'] is None
    assert evaluator.evaluate({}, 'move_first_cup.carry')['satisfied']
    with pytest.raises(ValueError, match='Both'):
        evaluator.evaluate({'cup_b': poses['cup_b']}, 'move_second_cup.carry')


def test_common_rigid_transform_invariance_and_reference_motion(evaluator):
    phase = 'move_second_cup.carry'
    poses = {'cup': RigidTransform([1.42, .85, .84]),
             'cup_b': RigidTransform([1.2, .95, 1.12])}
    expected = evaluator.evaluate(poses, phase)['minimum_residual_m']
    rotation = RotationMatrix.MakeYRotation(.4) @ RotationMatrix.MakeZRotation(.7)
    transform = RigidTransform(rotation, [2., -3., .8])
    transformed = {key: transform @ value for key, value in poses.items()}
    assert evaluator.evaluate(transformed, phase)['minimum_residual_m'] == pytest.approx(expected)
    # Identical carried-object world pose becomes invalid as reference moves.
    poses['cup'] = RigidTransform([1.42, 1.15, .84])
    assert not evaluator.evaluate(poses, phase)['satisfied']


def test_translation_bound_for_rotated_reference(evaluator):
    ref = RigidTransform(RotationMatrix.MakeZRotation(.9), [1.42, .85, .84])
    rng = np.random.default_rng(102)
    for _ in range(30):
        a = rng.uniform([-.7,-.1,.2],[.3,.2,.4])
        delta = rng.normal(0,.04,3)
        def residual(p):
            return evaluator.evaluate({'cup': ref, 'cup_b': ref @ RigidTransform(p)},
                                      'move_second_cup.carry')['minimum_residual_m']
        assert abs(residual(a)-residual(a+delta)) <= np.linalg.norm(delta)+1e-12


def test_successor_reference_pose_used_after_active_object_switch():
    cfg = make_config(task()); world = initial_world(cfg)
    moved = RigidTransform(RotationMatrix.MakeZRotation(.05), [1.42,.85,.84])
    world = successor(world, 'cup', moved.GetAsMatrix4(), 'dropoff_table', world['robot'])
    scene = build_scene(config=world_config(cfg,world,'cup_b'),visualize=False)
    e = ObjectRelativeEvaluator(scene)
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body,
                               moved @ RigidTransform([-.22,.1,.28]))
    checked = e.current(scene,'move_second_cup.carry')
    assert checked['satisfied']
    assert np.allclose(checked['constraints'][0]['reference_pose'], moved.GetAsMatrix4())


def test_relative_bound_in_payload_oracle_and_stage_audit():
    from src.grasp_planner import plan_pick
    from src.payload_translation_bridge import PayloadTranslationOracle
    s = task(); cfg = make_config(s); world = initial_world(cfg)
    world = successor(world,'cup',RigidTransform([1.42,.85,.84]).GetAsMatrix4(),
                      'dropoff_table',world['robot'])
    scene=build_scene(config=world_config(cfg,world,'cup_b'),visualize=False)
    base=np.array([.65,0.,0.]);pick=plan_pick(scene,base)
    set_robot_configuration(scene,base,pick.grasp_q,scene.gripper_closed_positions)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@scene.cup_initial_pose
    oracle=PayloadTranslationOracle(scene,pick.lift_q,scene.gripper_closed_positions,0.,X,action_id='move_second_cup')
    assert oracle.query([1.17,0.])[0] < 0
    assert oracle.last_residuals['object_relative'] < 0
    assert oracle.query([1.17,.1])[0] > 0
    frame=dict(base_q=[1.17,0.,0.],arm_q=pick.lift_q.tolist(),
        gripper_q=scene.gripper_closed_positions.tolist(),mode='carry',
        phase='move_second_cup.carry',object_id='cup_b',world_objects=world['poses'])
    result=audit_action(scene,[frame],X)
    assert not result['satisfied'] and result['minimum_object_relative_residual_m']<0
    assert not result['counterexamples'][0]['object_relative']['satisfied']


@pytest.mark.parametrize('change',[{'reference_object_id':'cup_b'},
    {'reference_object_id':'absent'}, {'reference_policy':'moving'},
    {'action_id':'absent'}, {'object_id':'cup'}, {'margin_m':-.1}, {'cells':[]}])
def test_unsupported_specs_rejected_before_planning(change):
    s=task();s['object_relative'][0].update(change)
    with pytest.raises(ValueError):make_config(s)


def test_operation_budget_can_scale_without_changing_semantics_or_audit_rounds():
    s=task();original=make_config(s)
    s['operation_refinement']={'time_budget_s':180.}
    expanded=make_config(s)
    assert expanded['green_room']['operation_refinement']=={
        'max_rounds':2,'max_regions':512,'time_budget_s':180.}
    expanded['green_room']['operation_refinement']['time_budget_s']=60.
    assert expanded==original


@pytest.mark.parametrize('policy',[{'time_budget_s':0},{'time_budget_s':float('inf')},
    {'time_budget_s':True},{'max_regions':1.5},{'max_rounds':-1},{'skip_validation':True}])
def test_bad_operation_budgets_rejected(policy):
    s=task();s['operation_refinement']=policy
    with pytest.raises(ValueError):make_config(s)
