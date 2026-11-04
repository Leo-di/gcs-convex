"""Collision and contact checks for real imported ReplicaCAD cafe assets."""
import json
from pathlib import Path
import numpy as np
import pytest

from src.replicacad import Y_UP_TO_Z_UP
from src.scene_spec import load_scene_config, support_names

ASSETS=Path('data/replicacad_drake/cafe/frl_apartment_table_01/asset.json')


def test_coordinate_conversion_is_proper_and_maps_height_to_z():
    assert np.allclose(Y_UP_TO_Z_UP.T @ Y_UP_TO_Z_UP,np.eye(3))
    assert np.linalg.det(Y_UP_TO_Z_UP)==pytest.approx(1.)
    assert np.allclose(Y_UP_TO_Z_UP @ [0,1,0],[0,0,1])
    assert np.allclose(Y_UP_TO_Z_UP @ [0,0,1],[0,-1,0])


@pytest.mark.skipif(not ASSETS.exists(),reason='Run pinned ReplicaCAD asset preparation first')
def test_original_table_components_remain_separate():
    asset=json.loads(ASSETS.read_text())
    assert len(asset['collisions'])==6
    assert all(p['type']=='convex' for p in asset['collisions'])
    from pydrake.geometry import Convex
    for p in asset['collisions']:
        assert Convex(p['path']).GetConvexHull().num_vertices()>=4


@pytest.mark.skipif(not ASSETS.exists(),reason='ReplicaCAD assets required')
def test_all_three_source_supports_use_actual_mesh_contact_geometry():
    from src.stateful_transport_grounding import make_config,initial_world,world_config
    from src.build_scene import build_scene
    from src.grasp_audit import measure_cup_gripper
    spec=load_scene_config('configs/replicacad_cafe_corridor_transport.yaml')
    cfg=make_config(spec);world=initial_world(cfg)
    assert world['supports']==dict(cup='table',cup_b='customer_b',cup_c='customer_c')
    for key,support in world['supports'].items():
        scene=build_scene(config=world_config(cfg,world,key),visualize=False)
        rows=[r for r in measure_cup_gripper(scene)['pairs'] if r['body']==support]
        assert len(rows)==6 and abs(min(r['distance_m'] for r in rows))<1e-6
        assert len(scene.plant.GetCollisionGeometriesForBody(scene.plant.GetBodyByName(support)))==6


@pytest.mark.skipif(not ASSETS.exists(),reason='ReplicaCAD assets required')
def test_imported_supports_sustain_all_three_free_cups_in_contact_simulation():
    from src.stateful_transport_grounding import make_config
    from src.execute_delivery_dynamics import execute
    cfg=make_config(load_scene_config('configs/replicacad_cafe_corridor_transport.yaml'))
    frame=dict(base_q=cfg['robot']['start'],arm_q=cfg['pick_task']['navigation_arm_q'],
               gripper_q=[-.055,.055],phase='settle.arrive',object_id='cup',mode='arrive')
    report,scene=execute(cfg,[frame,frame],visualize=False,time_scale=1.)
    last=report['trace'][-1]['objects']
    for key,support in cfg['initial_supports'].items():
        assert last[key]['contacts_N'][support]>.05
        assert abs(last[key]['pose'][2][3]-.84)<.01
        assert last[key]['speed_m_s']<.01
    assert report['unexpected_contact_count']==0


@pytest.mark.skipif(not ASSETS.exists(),reason='ReplicaCAD assets required')
def test_full_scene_has_architecture_rigid_furniture_and_joint_snapshots():
    manifest=json.loads(Path('data/replicacad_drake/scene.json').read_text())
    assert len(manifest['entries'])==25
    assert sum(e['name'].startswith('rc_joint') for e in manifest['entries'])==6
    assert len(manifest['entries'][0]['asset']['collisions'])>1
    assert len(support_names({'support_surfaces':{'customer_b':{},'customer_c':{}}}))==4
