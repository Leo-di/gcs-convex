import numpy as np
import pytest
from pydrake.geometry.optimization import HPolyhedron
from pydrake.math import RigidTransform
from pydrake.common import RandomGenerator
from src.build_scene import build_scene, set_robot_configuration
from src.scene_spec import load_scene_config
from src.grasp_planner import solve_arm_ik, cartesian_arm_path, plan_pick
from src.green_room_gcs import guide_edge_validator
from src.green_room_iris import make_checker, grow_cover
from src.contact_boundary import support_boundary, close_boundary, interior_witness


def supported_case():
    config=load_scene_config('configs/scene_semantic_choice_v6.yaml')
    base=np.array([-.4,.4,0])
    q=np.array([-.04348672158934056,1.0582273642006632,.8107386229278892,
                -.7165937959314767,.8900287134170002,1.276691237421579,1.548741867243699])
    scene=build_scene(config=config,visualize=False)
    set_robot_configuration(scene,base,q,scene.gripper_closed_positions)
    pose=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)
    X=pose.inverse()@scene.cup_initial_pose
    return config,scene,base,q,X,pose


def test_support_roundoff_is_distinct_from_penetration_and_keeps_collision_pair():
    cfg,s,base,q,X,_=supported_case()
    mode,checker,mapping=make_checker(cfg,base,q,s.gripper_closed_positions,X)
    assert support_boundary(mode,mapping(q))
    inspector=mode.scene_graph.model_inspector()
    table=mode.plant.GetCollisionGeometriesForBody(mode.plant.GetBodyByName('table'))
    assert all(not inspector.CollisionFiltered(mode.attached_cup_geometry_id,g) for g in table)
    # Move the payload down only in this negative test, never in production.
    set_robot_configuration(s,base,q,s.gripper_closed_positions)
    pose=s.plant.EvalBodyPoseInWorld(s.plant_context,s.gripper_body)
    lowered=RigidTransform(s.cup_initial_pose.rotation(),s.cup_initial_pose.translation()-[0,0,.001])
    bad,_,mp=make_checker(cfg,base,q,s.gripper_closed_positions,pose.inverse()@lowered)
    assert not support_boundary(bad,mp(q))


def test_closure_refuses_geometric_expansion():
    box=HPolyhedron.MakeBox(np.zeros(7),np.ones(7))
    point=np.ones(7);point[0]+=1e-12
    result,shift=close_boundary(box,[point])
    assert result.PointInSet(point,0)
    assert 0 < shift < 1e-9
    point[0]=1.001
    with pytest.raises(RuntimeError,match='roundoff budget'):
        close_boundary(box,[point])


def test_official_iris_includes_original_supported_endpoint():
    cfg,s,base,q,X,pose=supported_case()
    target=pose.multiply(np.array(cfg['gripper']['grasp_point_in_body']))+[0,0,.01]
    q1=solve_arm_ik(s,base_q=base,target_position=target,initial_arm_q=q,
        position_tolerance=.0002,minimum_clearance=.003,target_orientation=pose.rotation(),
        orientation_tolerance=.005,maximum_joint_step=.12)
    assert np.max(np.abs(q1-q))<.120001
    regions,records,_=grow_cover(s,base,np.array([q,q1]),s.gripper_closed_positions,X_GC=X,semantic=False)
    assert any(r.PointInSet(q,1e-9) for r in regions)
    assert any(r.PointInSet(q1,1e-9) for r in regions)
    assert records[0]['support_boundary_points']
    assert records[0]['numerical_closure_shift_rad'] <= 1e-9
    assert records[0]['contact_domain_cap_radius_rad'] > 0
    region=regions[0]
    sample=region.ChebyshevCenter()
    assert np.min((region.b()-region.A()@sample)/np.linalg.norm(region.A(),axis=1)) > 1e-9
    mode,checker,mapping=make_checker(cfg,base,q,s.gripper_closed_positions,X)
    far_seed=q+(q1-q)*(.2/np.linalg.norm(q1-q))
    witness,boundary=interior_witness(mode,checker,mapping,q,far_seed)
    assert boundary and 0 < np.linalg.norm(witness-q) <= 1e-9
    assert checker.CheckConfigCollisionFree(mapping(witness))
    rng=RandomGenerator(128915)
    for _ in range(500):
        sample=region.UniformSample(rng,sample,mixing_steps=20)
        assert checker.CheckConfigCollisionFree(mapping(sample))


def test_previous_three_radian_jump_has_continuous_collision_checked_guide():
    cfg=load_scene_config('configs/scene_semantic_choice_v6.yaml')
    cfg['pick_task']['gripper_rpy']=[0.,0.,0.]
    s=build_scene(config=cfg,visualize=False);base=np.array([.45,0,0])
    pick=plan_pick(s,base)
    edge=guide_edge_validator(s,base,pick.pregrasp_q,s.gripper_open_positions,semantic=True)
    path=cartesian_arm_path(s,base,pick.pregrasp_q,pick.grasp_q,25,stage='regression',edge_validator=edge)
    assert np.max(np.abs(np.diff(path,axis=0))) <= .120001
    assert all(edge(a,b) for a,b in zip(path,path[1:]))
