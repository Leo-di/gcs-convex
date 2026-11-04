import copy
import json
import numpy as np
import pytest
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config
from src.build_scene import build_scene, set_robot_configuration
from src.combined_semantics_study import method_spec, METHODS


def test_factorial_policies_preserve_world_endpoints_and_visual_regions():
    original = load_scene_config('configs/combined_object_semantics.yaml')
    saved = copy.deepcopy(original)
    for method in METHODS:
        local = method_spec(original, method)
        assert bool(local['object_exclusions']) == (method in ('room_exclusion','all_constraints'))
        assert bool(local['object_corridors']) == (method in ('room_corridor','all_constraints'))
        assert local['object_exclusion_display'] == original['object_exclusions']
        assert local['object_corridor_display'] == original['object_corridors']
        for key in original.keys()-{'object_exclusions','object_corridors'}:
            assert local[key] == original[key]
    assert original == saved
    with pytest.raises(ValueError):method_spec(original,'unknown')


@pytest.fixture
def oracle():
    from src.grasp_planner import plan_pick
    from src.payload_translation_bridge import PayloadTranslationOracle
    s=load_scene_config('configs/combined_object_semantics.yaml')
    scene=build_scene(config=make_config(s),visualize=False)
    base=np.array(s['actions'][0]['pickup_base']);pick=plan_pick(scene,base)
    set_robot_configuration(scene,base,pick.grasp_q,scene.gripper_closed_positions)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@scene.cup_initial_pose
    return PayloadTranslationOracle(scene,pick.lift_q,scene.gripper_closed_positions,0.,X,action_id=s['actions'][0]['id'])


@pytest.mark.parametrize('y,exclusion_ok,corridor_ok',[(-.16,False,True),(-.24,True,False),(-.19,True,True)])
def test_no_constraint_can_overwrite_another(oracle,y,exclusion_ok,corridor_ok):
    clearance,_,_=oracle.query([.95,y]);r=oracle.last_residuals
    assert (r['object_exclusion']>0)==exclusion_ok
    assert (r['object_corridor']>0)==corridor_ok
    assert r['robot_room']>0 and r['payload_room']>0 and r['environment']>0
    assert (clearance>0)==(exclusion_ok and corridor_ok)
    assert clearance == pytest.approx(min(v for k,v in r.items() if k!='internal' and v is not None))


def test_room_constraint_remains_active_with_both_object_constraints(oracle):
    oracle.query([-.1,-.19])
    assert oracle.last_residuals['robot_room']<0


def test_disconnected_cover_retains_diagnostics_and_never_enters_gcs(monkeypatch,tmp_path):
    import src.payload_translation_bridge as bridge
    from pydrake.geometry.optimization import Point
    from types import SimpleNamespace
    class Oracle:
        def __init__(self,*a,**k):
            self.exclusions=SimpleNamespace(specs=[{'id':'blocked_zone'}])
            self.corridors=SimpleNamespace(specs=[{'id':'corridor','action_id':'move','mode':'carry'}])
            self.phase='move.carry'
        def query(self,xy):return 1.,1.,None
    monkeypatch.setattr(bridge,'PayloadTranslationOracle',Oracle)
    records=[dict(center=[0,0],radius=.1),dict(center=[1,0],radius=.1)]
    monkeypatch.setattr(bridge,'make_cover',lambda *a,**k:([Point([0.,0.]),Point([1.,0.])],records))
    monkeypatch.setattr(bridge,'prune_contained_disks',lambda r,s:(r,s))
    def forbidden(*a,**k):raise AssertionError('Disconnected cover must not reach a GCS solve')
    monkeypatch.setattr(bridge,'GcsTrajectoryOptimization',forbidden)
    artifact=tmp_path/'bridge.json'
    with pytest.raises(bridge.PayloadBridgeRejected) as error:
        bridge.plan_payload_translation(None,[0,0,0],[1,0,0],None,None,None,
            dict(lower_xy=[-1,-1],upper_xy=[2,1],margin_m=.005,spacing_m=.01,max_radius_m=.2),artifact)
    assert error.value.stage=='payload_cover_disconnected'
    saved=json.loads(artifact.read_text())
    assert not saved['satisfied'] and saved['source_region_count']==saved['target_region_count']==1
    assert saved['object_exclusion_ids']==['blocked_zone'] and saved['active_object_corridor_ids']==['corridor']


def test_small_safe_regions_preserve_narrow_corridor_connectivity():
    from src.geometry_outbound_gcs import make_cover, select_region_chain
    from src.payload_translation_bridge import overlap_edges
    lo,hi=np.array([0.,-.02]),np.array([.1,.02])
    box=[dict(name='narrow',lower=lo,upper=hi)]
    # Infinite strip with 15 mm center clearance. A 5 mm planning margin
    # leaves safe 10 mm disks, which the old 25 mm cutoff discarded.
    query=lambda q:(.015-abs(q[1]),1.,None)
    args=(query,box,[0,0],[.1,0],lo,hi)
    old,_=make_cover(*args,spacing=.01,margin=.005,max_radius=.1)
    assert not old
    regions,records=make_cover(*args,spacing=.01,margin=.005,max_radius=.1,min_radius=.0025)
    sources=[i for i,r in enumerate(regions) if r.PointInSet([0.,0.])]
    targets=[i for i,r in enumerate(regions) if r.PointInSet([.1,0.])]
    assert select_region_chain(records,overlap_edges(records),sources,targets)
    for r in records:
        # Every retained disk remains inside the strip with the same margin.
        assert abs(r['center'][1])+r['radius']<=.015-.005+1e-12


@pytest.mark.parametrize('change',[{'min_radius_m':0.},{'min_radius_m':1.},
    {'refinement_spacing_m':.03},{'refinement_radius_threshold_m':float('nan')}])
def test_invalid_refinement_parameters_are_rejected(change):
    spec=load_scene_config('configs/combined_object_semantics.yaml');spec['carry'].update(change)
    with pytest.raises(ValueError):make_config(spec)


@pytest.mark.parametrize('config,blocked',[('combined_object_semantics_conflict.yaml',True),
                                         ('combined_object_semantics.yaml',False)])
def test_cross_section_checker_distinguishes_conflict_from_narrow_passage(tmp_path,config,blocked):
    from scripts.verify_combined_semantics import conflict_cross_section
    from pydrake.math import RigidTransform
    spec=load_scene_config('configs/'+config)
    folder=tmp_path/'conflict/room_corridor/run';folder.mkdir(parents=True)
    (tmp_path/'conflict/spec.json').write_text(json.dumps(spec))
    frames=[dict(mode='carry',world_objects={'cup':RigidTransform([x,.85,1.12]).GetAsMatrix4().tolist()}) for x in (.46,1.42)]
    (folder/'reference.json').write_text(json.dumps(frames))
    result=conflict_cross_section(tmp_path)
    assert result['checked'] and result['blocking_slice']==blocked
    assert result['endpoint_centers_separated_by_slice']
