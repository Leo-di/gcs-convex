"""Functional access must reject geometric counterexamples on real imported meshes."""
from pathlib import Path
import numpy as np
import pytest
from pydrake.math import RigidTransform
from src.human_passage import widest_path, capsule_box_distances, HumanPassageEvaluator, validate_passage
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config
from src.build_scene import build_scene,set_robot_configuration
from src.scene_slice_gcs import SliceOracle
from src.green_room_gcs import operation_oracle
from src.whole_robot_room import arm_point_reach_bound


def test_four_neighbor_connectivity_rejects_diagonal_gap_and_negative_cells():
    route,slack=widest_path(np.array([1.,-1.,-1.,1.]),(2,2),0,3)
    assert route is None
    values=np.array([1.,.2,.8,.1,1.,1.])
    route,slack=widest_path(values,(2,3),0,5)
    assert route[0]==0 and route[-1]==5 and slack==.2
    coords=np.column_stack(np.divmod(route,3))
    assert np.all(np.abs(np.diff(coords,axis=0)).sum(axis=1)==1)


def test_capsule_vertical_geometry_and_full_cell_guard():
    d=capsule_box_distances(np.array([[0.,0.],[1.,0.]]),np.array([[.4,-.1,0.],[0.,0.,2.]]),np.array([[.5,.1,.8],[.1,.1,3.]]),.28,1.42)
    assert np.allclose(d[0],[.4,.58])
    # A center with only 2 cm spare clearance cannot certify an 8 cm cell.
    assert .32-(.28+.02+.08/np.sqrt(2))<0


@pytest.fixture
def cafe():
    if not Path('data/replicacad_drake/human_passage/service_island/asset.json').exists():pytest.skip('Prepared mesh assets required')
    cfg=make_config(load_scene_config('configs/replicacad_human_passage_transport.yaml'))
    return cfg,build_scene(config=cfg,visualize=False)


def test_actual_mesh_docks_require_different_human_routes(cafe):
    cfg,s=cafe;h=HumanPassageEvaluator(s)
    paths=[]
    for base in [[.45,0.,0.],[3.45,-1.65,0.]]:
        set_robot_configuration(s,base,cfg['pick_task']['navigation_arm_q'],s.gripper_open_positions)
        r=h.current();assert r['satisfied'];assert r['static_witness_slack_m']>=0
        paths.append(np.array(r['path_xy']))
    crossing=[p[np.abs(p[:,0]-1.9)<.05,1].mean() for p in paths]
    assert crossing[0]<-1.3 and crossing[1]>-.3
    # The path valid at the first dock is physically blocked at the second.
    lo,hi=h.bounds(h.dynamic)
    assert (capsule_box_distances(paths[0],lo,hi,h.zlo,h.zhi)-h.required).min()<0


def test_collision_free_exit_blocker_is_semantically_rejected(cafe):
    cfg,s=cafe;disabled=dict(cfg['human_passage'],enforce=False);s.config['human_passage']=disabled
    geometric=SliceOracle(s,cfg['pick_task']['navigation_arm_q'],s.gripper_open_positions)
    d,sd,_=geometric.query([4.6,-.8]);assert min(d,sd)>0
    assert not HumanPassageEvaluator(s,force=True).current()['satisfied']
    s.config['human_passage']=dict(disabled,enforce=True)


def test_payload_override_and_conservative_arm_motion_bound(cafe):
    cfg,s=cafe;set_robot_configuration(s,[.45,0.,0.],cfg['pick_task']['navigation_arm_q'],s.gripper_open_positions)
    X=RigidTransform([0.,0.,.2]);oracle=operation_oracle(s,X)
    assert 'payload' in oracle.clear.passage.categories
    assert oracle.aabb_motion_factor==np.sqrt(3)
    assert arm_point_reach_bound(s,oracle)>arm_point_reach_bound(s,oracle.room)
    actual=s.plant.EvalBodyPoseInWorld(s.plant_context,s.cup_body).GetAsMatrix4().copy()
    oracle.query_current()
    assert np.array_equal(actual,s.plant.EvalBodyPoseInWorld(s.plant_context,s.cup_body).GetAsMatrix4())


def test_explicit_model_validation(cafe):
    cfg,_=cafe;bad=dict(cfg['human_passage'],height_m=.3)
    with pytest.raises(ValueError):validate_passage(bad)


def test_yaw_cover_radius_bounds_conservative_link_aabb_motion(cafe):
    from src.rotation_safe_bridge import rigid_yaw_radius
    cfg,s=cafe;arm=np.array(cfg['pick_task']['navigation_arm_q']);R=rigid_yaw_radius(s,arm)
    h=HumanPassageEvaluator(s)
    set_robot_configuration(s,[0.,0.,0.],arm,s.gripper_open_positions);lo,hi=h.bounds(h.dynamic)
    moving=np.array([category in ('base','arm') for category in h.categories])
    # Test rotated extrema, rather than a test that repeats the radius formula.
    for yaw in [-.4,-.07,.1,.4]:
        set_robot_configuration(s,[0.,0.,yaw],arm,s.gripper_open_positions);a,b=h.bounds(h.dynamic)
        delta=np.maximum(np.abs(a-lo),np.abs(b-hi))[moving]
        assert np.linalg.norm(delta,axis=1).max()<=R*abs(yaw)+1e-12


def test_grid_guide_finds_a_detour_and_rejects_blocked_goal():
    from src.grid_navigation_guide import grid_guide
    # Rectangle blocks the direct route but leaves a route above and below it.
    def distance(q):
        p=np.asarray(q[:2]);lo=np.array([.4,-.2]);hi=np.array([.6,.2])
        outside=np.linalg.norm(np.maximum(np.maximum(lo-p,p-hi),0))
        d=outside if outside else -min(np.r_[p-lo,hi-p])
        return d,1.
    path,report=grid_guide(distance,[0,0,0],[1,0,0],[-.2,-.5,0],[1.2,.5,0],spacing=.08)
    assert abs(path[:,1]).max()>.23 and len(path)>2
    for a,b in zip(path,path[1:]):
        assert min(distance((1-t)*a+t*b)[0] for t in np.linspace(0,1,101))>=.03
    with pytest.raises(RuntimeError,match='endpoint'):
        grid_guide(distance,[0,0,0],[.5,0,0],[-.2,-.5,0],[1.2,.5,0])


def test_payload_query_cap_preserves_every_clearance_that_changes_a_cover(cafe):
    from src.payload_translation_bridge import PayloadTranslationOracle
    cfg,s=cafe;arm=np.array(cfg['pick_task']['navigation_arm_q'])
    set_robot_configuration(s,[.45,0,0],arm,s.gripper_open_positions)
    X=s.plant.EvalBodyPoseInWorld(s.plant_context,s.gripper_body).inverse()@s.cup_initial_pose
    full=PayloadTranslationOracle(s,arm,s.gripper_open_positions,0.,X)
    cap=.355;bounded=PayloadTranslationOracle(s,arm,s.gripper_open_positions,0.,X,max_distance=cap)
    for p in [[.45,0],[-.8,-1.65],[1.8,1.3],[4.6,-.8]]:
        a=full.query(p);b=bounded.query(p)
        assert abs(min(a[0],cap)-b[0])<1e-10
        assert (a[1]>=0)==(b[1]>=0)


def test_valid_but_exhausted_cached_route_is_refreshed_before_it_blocks_guidance(cafe):
    cfg,s=cafe;h=HumanPassageEvaluator(s)
    slacks=[]
    for x in [-.9,-.5,-.32,0.,.5]:
        set_robot_configuration(s,[x,-1.65,0.],cfg['pick_task']['navigation_arm_q'],s.gripper_open_positions)
        r=h.current();assert r['satisfied'];slacks.append(r['motion_slack_m'])
    assert min(slacks)>.12 and h.grid_searches>1
