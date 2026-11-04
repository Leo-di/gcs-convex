"""Real mesh regression: near-contact lift domains and unchanged guards."""
from pathlib import Path
import numpy as np
import pytest
from pydrake.math import RigidTransform
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config, initial_world, world_config
from src.build_scene import build_scene, set_robot_configuration
from src.green_room_iris import make_checker, grow_cover
from src.green_room_gcs import make_operation_cover, region_holdout
from src.contact_boundary import near_support, support_boundary, support_proximity, contact_domain

pytestmark=pytest.mark.skipif(not Path('data/replicacad_drake/cafe/frl_apartment_table_01/asset.json').exists(),
    reason='Prepared pinned ReplicaCAD assets required')
START=np.array([-.49918741327101285,-1.0559537419897032,1.7747104702842489,
    -1.2707310331062,2.384709402727721,-.06644526607451448,1.777774337171071])
END=np.array([-.5118045660066182,-.9994939447730005,1.8123718763456282,
    -1.2462389838748302,2.3625259390478623,-.09258322304704455,1.73579416451196])
BAD=np.array([-.5126341616298902,-1.0568047624158246,1.7891861432525469,
    -1.2508618211000784,2.3758768299999655,-.08732449171770709,1.7621048904810452])


def case(key='cup',raw_world=False):
    cfg=make_config(load_scene_config('configs/replicacad_cafe_transport.yaml'))
    if not raw_world:cfg=world_config(cfg,initial_world(cfg),key)
    base=np.array([.45,3. if key=='cup_c' else 0.,0.])
    scene=build_scene(config=cfg,visualize=False)
    set_robot_configuration(scene,base,START,scene.gripper_closed_positions)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@scene.cup_initial_pose
    mode,checker,mapping=make_checker(cfg,base,START,scene.gripper_closed_positions,X)
    return cfg,scene,base,X,mode,checker,mapping


def test_positive_mesh_gap_activates_domain_taper_without_relaxing_contact():
    cfg,scene,base,X,mode,checker,mapping=case()
    state=support_proximity(mode,mapping(START))
    assert 1e-10 < state['distance_m'] < 1e-6
    assert near_support(mode,mapping(START)) and not support_boundary(mode,mapping(START))
    assert not checker.CheckConfigCollisionFree(mapping(BAD))
    assert not near_support(mode,mapping(BAD))
    inspector=mode.scene_graph.model_inspector()
    for gid in mode.plant.GetCollisionGeometriesForBody(mode.plant.GetBodyByName('table')):
        assert not inspector.CollisionFiltered(mode.attached_cup_geometry_id,gid)


def test_official_mesh_iris_rejects_original_counterexample_and_passes_fresh_audits():
    cfg,scene,base,X,mode,checker,mapping=case()
    regions,records,_=grow_cover(scene,base,np.array([START,END]),scene.gripper_closed_positions,X_GC=X)
    assert records[0]['method']=='Drake IrisNp2'
    assert regions[0].PointInSet(START,1e-9) and regions[0].PointInSet(END,1e-9)
    assert not regions[0].PointInSet(BAD,1e-9)
    assert records[0]['contact_domain_cap_radius_rad']>1e-6
    center=regions[0].ChebyshevCenter()
    radius=np.min((regions[0].b()-regions[0].A()@center)/np.linalg.norm(regions[0].A(),axis=1))
    assert radius>1e-6
    for seed in (2026103101,2026103102):
        audit=region_holdout(scene,base,records,scene.gripper_closed_positions,X_GC=X,seed=seed)
        assert audit['collisions']==audit['raw_grasp_violations']==audit['semantic_violations']==0


def test_named_source_support_and_real_penetration_remain_distinct():
    cfg,scene,base,X,mode,checker,mapping=case('cup_c')
    assert support_proximity(mode,mapping(START))['support']=='customer_c'
    assert near_support(mode,mapping(START))
    lowered=RigidTransform(scene.cup_initial_pose.rotation(),scene.cup_initial_pose.translation()-[0,0,.001])
    grip=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)
    bad,check,mp=make_checker(cfg,base,START,scene.gripper_closed_positions,grip.inverse()@lowered)
    assert not near_support(bad,mp(START)) and not support_boundary(bad,mp(START))
    assert not check.CheckConfigCollisionFree(mp(START))


def test_other_resting_objects_do_not_collapse_the_moving_payload_domain():
    cfg,scene,base,X,mode,checker,mapping=case(raw_world=True)
    _,records,reach=make_operation_cover(scene,base,np.array([START,END]),scene.gripper_closed_positions)
    r=records[0]
    domain,seed,cap=contact_domain(mode,checker,mapping,START,END,r['lower'],r['upper'],reach+np.linalg.norm(X.translation())+.18)
    assert cap>1e-6 and domain.PointInSet(START,1e-9) and domain.PointInSet(END,1e-9)
