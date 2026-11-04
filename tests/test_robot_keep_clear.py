import copy
import numpy as np
import pytest
from pydrake.math import RigidTransform
from src.scene_spec import load_scene_config, PROJECT_ROOT
from src.stateful_transport_grounding import make_config, semantic_plan, TransportRejected
from src.build_scene import build_scene, set_robot_configuration
from src.robot_keep_clear import RobotKeepClearEvaluator, validate_keep_clear


def task():
    spec = load_scene_config(PROJECT_ROOT/'configs/ordered_region_adaptive.yaml')
    spec['robot_keep_clear'] = [dict(id='aisle', lower_xyz=[.1,-1.,0.],
                                    upper_xyz=[2.7,-.3,1.8], margin_m=.02)]
    return spec


@pytest.fixture
def scene():
    return build_scene(config=make_config(task()), visualize=False)


def test_robot_restriction_is_not_a_physical_obstacle(scene):
    from src.collision_checker import evaluate_collision_status
    arm = scene.config['pick_task']['navigation_arm_q']
    geometries = scene.scene_graph.model_inspector().num_geometries()
    geometric = evaluate_collision_status(scene, [.45,-.08,0], arm_q=arm,
                                          gripper_q=scene.gripper_open_positions)
    result = RobotKeepClearEvaluator(scene).current()
    assert geometric.collision_free and not result['satisfied']
    assert result['worst']['body'] == 'mobile_base'
    scene.config['robot_keep_clear'] = []
    assert RobotKeepClearEvaluator(scene).current()['satisfied']
    assert scene.scene_graph.model_inspector().num_geometries() == geometries


def test_arm_intrusion_rejected_when_entire_base_is_outside(scene):
    set_robot_configuration(scene,[.45,0,0],scene.arm_home_positions,scene.gripper_open_positions)
    body=scene.plant.GetBodyByName('iiwa_link_7',scene.arm_instance)
    center=scene.plant.EvalBodyPoseInWorld(scene.plant_context,body).translation()
    region=dict(id='head_height_space', lower_xyz=(center-.025).tolist(),
                upper_xyz=(center+.025).tolist(),margin_m=.01)
    result=RobotKeepClearEvaluator(scene,[region]).current()
    assert result['base_residual_m']>0
    assert result['arm_gripper_residual_m']<0 and not result['satisfied']
    assert result['worst']['body']!='mobile_base'


def test_navigation_clearance_includes_reserved_space(scene):
    from src.scene_slice_gcs import SliceOracle
    oracle=SliceOracle(scene,scene.config['pick_task']['navigation_arm_q'],scene.gripper_open_positions)
    blocked=oracle.query([.45,-.08])
    assert blocked[0]<0 and blocked[2][1]=='semantic_reserved_space:aisle'
    assert oracle.query([.45,0])[0]>.03


def test_goal_rejected_before_ik_or_iris(scene,monkeypatch,tmp_path):
    from src.multiobject_grounding import plan_action
    import src.grasp_planner
    def forbidden(*args,**kwargs):raise AssertionError('IK must not run for blocked dock')
    monkeypatch.setattr(src.grasp_planner,'plan_pick',forbidden)
    action=task()['actions'][0]
    previous=dict(base_q=[.45,0.,0.],arm_q=scene.config['pick_task']['navigation_arm_q'],
                  gripper_q=scene.gripper_open_positions.tolist())
    with pytest.raises(TransportRejected,match='pickup_blocks_reserved_space'):
        plan_action(scene.config,action,[.45,-.08,0.],previous,tmp_path)
    assert (tmp_path/'keep_clear_rejection.json').exists()


def test_operation_domain_bounds_include_keep_clear(scene):
    from src.green_room_gcs import operation_oracle,semantic_box_bound
    from src.whole_robot_room import arm_point_reach_bound
    oracle=operation_oracle(scene);arm=np.array(scene.config['pick_task']['navigation_arm_q'])
    base=[.45,0,0];lo,hi=arm-.001,arm+.001
    report=semantic_box_bound(oracle,base,lo,hi,scene.gripper_open_positions,arm_point_reach_bound(scene,oracle))
    assert report['whole_box_semantic_lower_bound_m']>0
    rng=np.random.default_rng(101)
    for _ in range(30):
        value=oracle.query(np.r_[base,rng.uniform(lo,hi)],scene.gripper_open_positions)
        assert value['residual_m']+1e-10>=report['whole_box_semantic_lower_bound_m']
        assert value['keep_clear']['satisfied']


def test_supplied_plan_applies_to_entry_and_every_action_phase(scene):
    from src.semantic_plan import SemanticPlanEvaluator
    q=dict(base_q=[.45,-.08,0],arm_q=scene.config['pick_task']['navigation_arm_q'],
           gripper_q=scene.gripper_open_positions.tolist())
    frames=[dict(q,phase='room_entry.navigate',mode='enter_room'),
            dict(q,phase='move_first_cup.carry',mode='carry')]
    plan=semantic_plan(scene.config,frames)
    assert all('customer_space' in p['invariants'] for p in plan['phases'])
    checked=SemanticPlanEvaluator(scene,plan).audit(frames)
    assert not checked['satisfied']
    assert all(p['predicates']['customer_space']['minimum_residual']<0 for p in checked['phases'])


def test_payload_cover_cannot_accept_base_in_reserved_aisle(scene):
    from src.payload_translation_bridge import PayloadTranslationOracle
    arm=scene.config['pick_task']['navigation_arm_q']
    oracle=PayloadTranslationOracle(scene,arm,scene.gripper_closed_positions,0,RigidTransform(),action_id='move_first_cup')
    # The test targets the reserved-space bound, independent of the grasp permission adapter.
    oracle.permissions.specs=[]
    oracle.query([.45,-.08])
    assert oracle.last_residuals['robot_keep_clear']<0


@pytest.mark.parametrize('change',[{'margin_m':-1},{'lower_xyz':[0,0]},
    {'upper_xyz':[float('nan'),0,1]},{'lower_xyz':[5,5,5]},{'id':''}])
def test_invalid_zones_fail_before_planning(change):
    spec=task();spec['robot_keep_clear'][0].update(change)
    with pytest.raises(ValueError):make_config(spec)


def test_duplicate_zone_ids_rejected():
    zones=task()['robot_keep_clear']
    with pytest.raises(ValueError):validate_keep_clear(zones+copy.deepcopy(zones))
