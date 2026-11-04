import copy
import numpy as np
import pytest
from pydrake.math import RigidTransform, RotationMatrix

from src.scene_spec import load_scene_config
from src.build_scene import build_scene, set_robot_configuration
from src.stateful_transport_grounding import (
    make_config, initial_world, world_config, successor, digest, destination_occupancy,
    search, TransportRejected, measured_precondition, summarize_action, validate_task,
)
from src.semantic_plan import configuration


@pytest.fixture(scope='module')
def spec():
    return load_scene_config('configs/stateful_transport_grounding.yaml')


@pytest.fixture(scope='module')
def config(spec):
    return make_config(spec)


def test_successor_preserves_identity_orientation_and_branch_isolation(config):
    world = initial_world(config)
    before = digest(world)
    pose = RigidTransform(RotationMatrix.MakeZRotation(.15), [1.42, .85, .84])
    after = successor(world, 'cup', pose.GetAsMatrix4(), 'dropoff_table', world['robot'])
    assert digest(world) == before
    assert after['poses']['cup_b'] == world['poses']['cup_b']
    scene = build_scene(config=world_config(config, after, 'cup_b'), visualize=False)
    actual = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.object_bodies['cup'])
    assert np.allclose(actual.GetAsMatrix4(), pose.GetAsMatrix4())
    assert scene.cup_body == scene.object_bodies['cup_b']
    assert scene.config['planning_object_supports']['cup'] == ['dropoff_table']


def test_real_occupied_destination_uses_updated_object_geometry(config):
    world = initial_world(config)
    target = RigidTransform([1.66, .85, .84])
    world = successor(world, 'cup', target.GetAsMatrix4(), 'dropoff_table', world['robot'])
    scene = build_scene(config=world_config(config, world, 'cup_b'), visualize=False)
    blocked = destination_occupancy(scene, target)
    assert not blocked['satisfied']
    assert blocked['pairs'][0]['other_object'] == 'cup'
    assert blocked['pairs'][0]['distance_m'] < -.09
    free = destination_occupancy(scene, RigidTransform([1.42, .85, .84]))
    assert free['satisfied']


def fake_ground(config, world, action, placement, spec, folder):
    """Unit search adapter; occupancy still uses real Drake object geometry."""
    scene = build_scene(config=world_config(config, world, action['object_id']), visualize=False)
    pose = RigidTransform(placement)
    audit = destination_occupancy(scene, pose)
    if not audit['satisfied']:
        raise TransportRejected('occupied', audit)
    frame = dict(copy.deepcopy(world['robot']), phase=action['id']+'.test')
    return [frame], successor(world, action['object_id'], pose.GetAsMatrix4(),
                             action['destination_support'], frame), {}


def test_downstream_real_occupancy_rejection_rolls_back_previous_world(config, spec, tmp_path):
    result = search(config, spec, tmp_path, ground=fake_ground)
    assert result['satisfied'] and result['expansions'] == 4
    a, b, alternative, last = result['branches']
    assert a['local_satisfied'] and not b['local_satisfied']
    assert b['failure_stage'] == 'occupied' and alternative['local_satisfied'] and last['local_satisfied']
    assert a['input_world_sha256'] == alternative['input_world_sha256']
    assert b['input_world_sha256'] != last['input_world_sha256']
    assert any(e['event'] == 'rollback_after_downstream_failure' for e in result['events'])
    assert np.allclose(np.array(result['final_world']['poses']['cup'])[:3, 3], [1.42, .85, .84])
    assert np.allclose(np.array(result['final_world']['poses']['cup_b'])[:3, 3], [1.66, .85, .84])


def test_greedy_and_finite_budget_cannot_claim_complete_sequence(config, spec, tmp_path):
    greedy = search(config, spec, tmp_path, strategy='greedy', ground=fake_ground)
    assert not greedy['satisfied'] and greedy['expansions'] == 2
    bounded = copy.deepcopy(spec)
    bounded['max_expansions'] = 2
    result = search(config, bounded, tmp_path, ground=fake_ground)
    assert not result['satisfied'] and result['expansions'] == 2
    assert 'infeasibility' in result['scope']


def measured_start(config):
    world = initial_world(config)
    moved = RigidTransform([1.42, .85, .84]).GetAsMatrix4().tolist()
    world = successor(world, 'cup', moved, 'dropoff_table', world['robot'])
    frame = dict(world['robot'], world_objects=world['poses'], world_supports=world['supports'], phase='next.arrive')
    row = {'phase': 'next.arrive', 'time_s': 10., 'objects': {}}
    for key, X in world['poses'].items():
        row['objects'][key] = {'pose': copy.deepcopy(X), 'speed_m_s': 0., 'angular_speed_rad_s': 0.,
            'contacts_N': {'table': float(key == 'cup_b'), 'dropoff_table': float(key == 'cup')}}
    return frame, row


def test_measured_boundary_checks_previous_placement_orientation_support_and_tracking(config):
    frame, row = measured_start(config)
    q = configuration(frame)
    assert measured_precondition(frame, row, q, 'world')['satisfied']
    wrong = copy.deepcopy(row)
    wrong['objects']['cup']['pose'][0][3] -= .05
    assert not measured_precondition(frame, wrong, q, 'world')['satisfied']
    wrong = copy.deepcopy(row)
    wrong['objects']['cup']['pose'] = RigidTransform(RotationMatrix.MakeZRotation(.2), [1.42, .85, .84]).GetAsMatrix4().tolist()
    assert not measured_precondition(frame, wrong, q, 'world')['satisfied']
    wrong = copy.deepcopy(row)
    wrong['objects']['cup']['contacts_N']['dropoff_table'] = 0.
    assert not measured_precondition(frame, wrong, q, 'world')['satisfied']
    q[0] += .02
    assert not measured_precondition(frame, row, q, 'world')['satisfied']


def test_payload_oracle_moves_cup_and_counts_payload_obstacles(config, spec):
    from src.grasp_planner import plan_pick
    from src.payload_translation_bridge import PayloadTranslationOracle
    cfg = copy.deepcopy(config)
    scene = build_scene(config=cfg, visualize=False)
    pick = plan_pick(scene, np.array(spec['actions'][0]['pickup_base']))
    set_robot_configuration(scene, pick.base_q, pick.grasp_q, scene.gripper_closed_positions)
    X_GC = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body).inverse() @ scene.cup_initial_pose
    oracle = PayloadTranslationOracle(scene, pick.lift_q, scene.gripper_closed_positions, 0., X_GC)
    oracle.query(pick.base_q[:2])
    initial = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.cup_body).translation().copy()
    oracle.query(pick.base_q[:2]+[.15, 0.])
    moved = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.cup_body).translation().copy()
    assert np.allclose(moved-initial, [.15, 0, 0])
    cfg['objects']['distractors'].append({'name': 'payload_only_blocker', 'shape': 'box',
        'size': [.06, .06, .10], 'center': moved.tolist(), 'color': [1., 0., 0., 1.]})
    blocked = build_scene(config=cfg, visualize=False)
    oracle = PayloadTranslationOracle(blocked, pick.lift_q, blocked.gripper_closed_positions, 0., X_GC)
    assert oracle.query(pick.base_q[:2]+[.15, 0.])[0] < 0
    query = blocked.scene_graph.get_query_output_port().Eval(blocked.scene_graph.GetMyContextFromRoot(blocked.root_context))
    cup = blocked.plant.GetCollisionGeometriesForBody(blocked.cup_body)[0]
    obstacle = blocked.plant.GetCollisionGeometriesForBody(blocked.plant.GetBodyByName('payload_only_blocker'))[0]
    assert query.ComputeSignedDistancePairClosestPoints(cup, obstacle).distance < -.05


def test_transport_acceptance_needs_actual_carry_contact_and_final_placement(spec):
    action = spec['actions'][0]
    target, source = np.array(action['placements'][1]), RigidTransform([.46, .85, .84]).GetAsMatrix4()
    rows = []
    for mode, n in [('hold', 5), ('carry', 5), ('release', 50)]:
        for _ in range(n):
            pose = RigidTransform(target+[0, 0, .22] if mode != 'release' else target).GetAsMatrix4().tolist()
            rows.append({'phase': action['id']+'.'+mode, 'mode': mode, 'objects': {'cup': {
                'pose': pose, 'relative_translation_m': [0., 0., 0.], 'speed_m_s': 0., 'angular_speed_rad_s': 0.,
                'contacts_N': {'left_finger': float(mode != 'release'), 'right_finger': float(mode != 'release'),
                    'table': 0., 'dropoff_table': float(mode == 'release'), 'dropoff_bin': 0.}}}})
    assert summarize_action(rows, action, source, target)['satisfied']
    wrong = copy.deepcopy(rows)
    wrong[5]['objects']['cup']['contacts_N']['left_finger'] = 0.
    assert not summarize_action(wrong, action, source, target)['satisfied']
    wrong = copy.deepcopy(rows)
    wrong[5]['objects']['cup']['contacts_N']['table'] = 1.
    assert not summarize_action(wrong, action, source, target)['satisfied']
    wrong = copy.deepcopy(rows)
    wrong[-1]['objects']['cup']['pose'][0][3] += .03
    assert not summarize_action(wrong, action, source, target)['satisfied']


def test_unsupported_yaw_transition_and_task_schema_fail_closed(spec):
    from src.payload_translation_bridge import plan_payload_translation
    with pytest.raises(ValueError, match='equal yaw'):
        plan_payload_translation(None, [0, 0, 0], [1, 0, .1], None, None, None, {}, 'unused')
    bad = copy.deepcopy(spec)
    bad['actions'][1]['object_id'] = 'cup'
    with pytest.raises(ValueError, match='one transport per object'):
        validate_task(bad)
    bad = copy.deepcopy(spec)
    bad['carry']['margin_m'] = float('nan')
    with pytest.raises(ValueError, match='carrying envelope'):
        validate_task(bad)


def test_overlap_pruning_preserves_exact_graph():
    from src.payload_translation_bridge import overlap_edges
    from src.geometry_outbound_gcs import clipped_disks_intersect
    records = [{'center': [x, y], 'radius': r, 'lower': [-1., -1.], 'upper': [1., 1.]}
        for x, y, r in [(0., 0., .3), (.4, 0., .3), (.9, .9, .05), (0., .4, .3),
                        (.6+3e-16, 0., .3), (.6+1e-10, 0., .3)]]
    exact = {(i, j) for i, a in enumerate(records) for j, b in enumerate(records)
             if i != j and clipped_disks_intersect(a, b)}
    assert set(overlap_edges(records)) == exact


def test_entry_bridge_failure_retries_docking_without_changing_world(tmp_path):
    s=load_scene_config('configs/room_entry_transport.yaml')
    s['actions'][0]['pickup_candidates']=[dict(base_q=[.45,0.,0.],grasp_rpy=s['grasp_rpy']),
                                        dict(base_q=[.41,.04,0.],grasp_rpy=s['grasp_rpy'])]
    s['max_expansions']=2
    cfg=make_config(s);before=digest(cfg);seen=[]
    def entry(frames, folder):
        seen.append(copy.deepcopy(frames))
        if frames[0]['base_q'][0]>.43:
            raise TransportRejected('test_disconnected_entry', {'bounded_cover':True})
        nav=dict(frames[0],base_q=cfg['robot']['start'],phase='room_entry.navigate')
        return [nav]+frames, {'satisfied':True}
    result=search(cfg,s,tmp_path,ground=fake_ground,entry_planner=entry)
    assert result['satisfied'] and result['expansions']==2 and len(seen)==2
    first,second=result['branches']
    assert first['manipulation_satisfied'] and not first['local_satisfied'] and not first['entry_satisfied']
    assert first['failure_stage']=='entry_bridge_rejected'
    assert first['details']['cause_stage']=='test_disconnected_entry'
    assert second['entry_satisfied'] and result['selected'][0]['pickup_candidate_index']==1
    assert result['frames'][0]['phase']=='room_entry.navigate'
    assert result['initial_world']['robot']['base_q']==[.41,.04,0.]
    assert digest(cfg)==before
    assert result['initial_world']['supports']['cup']=='table'
    assert result['final_world']['supports']['cup']=='dropoff_table'


def test_all_entry_bridges_rejected_cannot_return_complete_reference(tmp_path):
    s=load_scene_config('configs/room_entry_transport.yaml');cfg=make_config(s)
    def entry(frames, folder): raise RuntimeError('No entry in the bounded cover')
    r=search(cfg,s,tmp_path,ground=fake_ground,entry_planner=entry)
    assert not r['satisfied'] and 'frames' not in r
    assert r['branches'][0]['manipulation_satisfied']
    assert not r['branches'][0]['local_satisfied']
