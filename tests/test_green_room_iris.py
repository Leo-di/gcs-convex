"""Official IRIS parameterization, payload collision model and admission gates."""
import numpy as np
from pydrake.math import RigidTransform
from src.build_scene import build_scene, set_robot_configuration
from src.green_room_iris import make_checker, grow_cover
from src.scene_spec import load_scene_config
from src.iris_region_generator import _reduced_position_indices


def test_parameterization_fixes_base_and_fingers_and_updates_all_seven_joints():
    cfg = load_scene_config('configs/scene_green_room_iris.yaml')
    base = np.array([1.2, -.1, 0])
    arm = np.asarray(cfg['pick_task']['navigation_arm_q'])
    scene, checker, mapping = make_checker(cfg, base, arm, [-.055, .055])
    before = scene.plant.GetPositions(scene.plant_context).copy()
    changed = arm + np.linspace(.001, .007, 7)
    full = mapping(changed)
    indices = _reduced_position_indices(scene)
    np.testing.assert_array_equal(full[indices[:3]], base)
    np.testing.assert_array_equal(full[indices[3:]], changed)
    fixed = np.setdiff1d(np.arange(len(full)), indices[3:])
    np.testing.assert_array_equal(full[fixed], before[fixed])
    assert checker.CheckConfigCollisionFree(mapping(arm))
    # Reach mode does not suppress fingertip/cup collision pairs.
    inspector = scene.scene_graph.model_inspector()
    cup = scene.plant.GetCollisionGeometriesForBody(scene.cup_body)
    for bi in scene.plant.GetBodyIndices(scene.gripper_instance):
        for gid in scene.plant.GetCollisionGeometriesForBody(scene.plant.get_body(bi)):
            assert all(not inspector.CollisionFiltered(cid, gid) for cid in cup)


def test_attached_payload_keeps_table_collision_enabled():
    cfg = load_scene_config('configs/scene_green_room_iris.yaml')
    base = np.array([1.2, -.1, 0])
    arm = np.asarray(cfg['pick_task']['navigation_arm_q'])
    scene, _, _ = make_checker(cfg, base, arm, [-.0475, .0475], RigidTransform([0, .1, 0]))
    inspector = scene.scene_graph.model_inspector()
    table = scene.plant.GetCollisionGeometriesForBody(scene.plant.GetBodyByName('table'))
    assert all(not inspector.CollisionFiltered(scene.attached_cup_geometry_id, gid) for gid in table)


def test_official_iris_region_is_generated_inside_semantic_domain():
    cfg = load_scene_config('configs/scene_green_room_iris.yaml')
    scene = build_scene(config=cfg, visualize=False)
    arm = np.asarray(cfg['pick_task']['navigation_arm_q'])
    regions, records, _ = grow_cover(scene, np.array([1.2, -.1, 0]),
                                    np.array([arm, arm + .001]), scene.gripper_open_positions)
    assert len(regions) == 1
    assert records[0]['method'] == 'Drake IrisNp2'
    assert regions[0].ambient_dimension() == 7
    assert records[0]['whole_box_semantic_lower_bound_m'] > 0
    assert regions[0].PointInSet(arm)
    assert regions[0].PointInSet(arm + .001)
