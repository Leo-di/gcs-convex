import copy
import hashlib
import json
import numpy as np
import pytest

from src.build_scene import build_scene
from src.scene_spec import load_scene_config
from src.semantic_plan import SemanticPlanEvaluator, configuration
from src.dynamic_semantics import schedule_with_waits
from src.learned_semantics import train
from src.multiobject_grounding import target_config


@pytest.fixture(scope='module')
def scene():
    return build_scene(config=load_scene_config('configs/scene_semantic_choice_v6.yaml'), visualize=False)


def frame(scene, x, t=0.):
    return {'base_q': [x, 0., 0.], 'arm_q': scene.config['pick_task']['navigation_arm_q'],
        'gripper_q': scene.gripper_open_positions.tolist(), 'phase': 'move', 'time_s': t}


def moving_plan(knots):
    return {'version': 1, 'predicates': {'person': {'kind': 'moving_keep_clear',
        'size_xy_m': [.36, .36], 'margin_m': .05, 'keyframes': knots}},
        'phases': [{'id': 'move', 'invariants': ['person']}]}


def test_stationary_robot_still_checks_between_timestamp_endpoints(scene):
    plan = moving_plan([[0., -1.5, 1.5], [1., -1.5, 0.], [2., -1.5, 1.5]])
    evaluator = SemanticPlanEvaluator(scene, plan)
    frames = [frame(scene, -1.5, 0.), frame(scene, -1.5, 2.)]
    assert all(evaluator.evaluate('person', configuration(f), time_s=f['time_s'])['satisfied'] for f in frames)
    audit = evaluator.audit(frames)
    assert not audit['satisfied'] and audit['samples'] >= 40


def test_wait_search_checks_wait_and_actual_movement(scene):
    plan = moving_plan([[0., -1.5, 0.], [2., -1.5, 0.], [3., -1.5, 2.], [20., -1.5, 2.]])
    frames = [frame(scene, x) for x in [-3., -2.5, -2., -1.5, -1., -.5]]
    frozen = copy.deepcopy(frames)
    result = schedule_with_waits(frames, plan, scene, candidate_delays_s=[0., 4.],
        frame_period_s=.2, initial_hold_s=0., tail_hold_s=0.)
    assert result['satisfied'] and result['delay_s'] == 4.
    assert not result['attempts'][0]['satisfied']
    assert frames == frozen
    assert not schedule_with_waits([frame(scene, -1.5)]*2, plan, scene,
        candidate_delays_s=[0., 4.], frame_period_s=.2, initial_hold_s=0., tail_hold_s=0.)['satisfied']


def test_missing_reversed_and_outside_prediction_times_rejected(scene):
    evaluator = SemanticPlanEvaluator(scene, moving_plan([[0., -1.5, 0.], [5., -1.5, 2.]]))
    with pytest.raises(ValueError, match='prediction horizon'):
        evaluator.audit([frame(scene, -3., 6.)])
    with pytest.raises(ValueError, match='nondecreasing'):
        evaluator.audit([frame(scene, -3., 2.), frame(scene, -3., 1.)])
    f = frame(scene, -3.)
    f.pop('time_s')
    with pytest.raises(ValueError, match='timestamps'):
        evaluator.audit([f])


@pytest.fixture(scope='module')
def model(scene, tmp_path_factory):
    path = tmp_path_factory.mktemp('model')/'room.json'
    result = train(scene, path, sizes=(120, 40, 40), seed=71)
    return path, result


def learned_plan(path):
    return {'version': 1, 'predicates': {'room': {'kind': 'learned_room', 'region': 'room2', 'margin_m': .02,
        'model_path': str(path), 'model_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}},
        'phases': [{'id': 'move', 'invariants': ['room']}]}


def test_real_model_has_disjoint_split_provenance_and_serialized_inference(scene, model):
    path, result = model
    assert len({r['dataset_sha256'] for r in result['metrics'].values()}) == 3
    assert result['split_seeds'] == [71, 72, 73]
    evaluator = SemanticPlanEvaluator(scene, learned_plan(path))
    value = evaluator.evaluate('room', configuration(frame(scene, .65)))
    assert value['method'] == 'learned_surrogate_with_analytical_guard'
    assert value['satisfied']
    outside = evaluator.evaluate('room', configuration(frame(scene, -3.)))
    assert not outside['satisfied'] and outside['out_of_domain'] and outside['abstained']


def test_confident_wrong_network_cannot_accept_illegal_configuration(scene, model, tmp_path):
    _, saved = model
    bad = copy.deepcopy(saved)
    bad['output_weights'] = [1.] + [0.]*202
    path = tmp_path/'bad.json'
    path.write_text(json.dumps(bad))
    evaluator = SemanticPlanEvaluator(scene, learned_plan(path))
    value = evaluator.evaluate('room', configuration(frame(scene, .25)))
    assert value['prediction_m'] == 1. and not value['satisfied'] and value['surrogate_disagreed']


def test_model_tampering_wrong_scene_and_nonfinite_weights_rejected(scene, model, tmp_path):
    path, saved = model
    supplied = learned_plan(path)
    supplied['predicates']['room']['model_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='hash mismatch'):
        SemanticPlanEvaluator(scene, supplied)
    supplied = learned_plan(path)
    supplied['predicates']['room']['margin_m'] = .03
    with pytest.raises(ValueError, match='scene/domain mismatch'):
        SemanticPlanEvaluator(scene, supplied)
    invalid = copy.deepcopy(saved)
    invalid['output_weights'][0] = float('nan')
    path = tmp_path/'nan.json'
    path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match='weights/domain'):
        SemanticPlanEvaluator(scene, learned_plan(path))


def test_all_objects_exist_and_target_switch_preserves_world_and_identities(scene):
    cfg = copy.deepcopy(scene.config)
    cfg['objects']['additional_cups'] = {'cup_b': dict(copy.deepcopy(cfg['objects']['cup']), center=[.68, .85, .84])}
    before = copy.deepcopy(cfg)
    selected = target_config(cfg, 'cup_b')
    active = build_scene(config=selected, visualize=False)
    assert set(active.object_bodies) == {'cup', 'cup_b'}
    assert active.object_bodies['cup_b'] == active.cup_body
    assert np.allclose(active.object_initial_poses['cup'].translation(), cfg['objects']['cup']['center'])
    assert cfg == before
    assert target_config(selected, 'cup')['objects'] == cfg['objects']
    dynamic = build_scene(config=cfg, visualize=False, dynamics_time_step=.001)
    inspector = dynamic.scene_graph.model_inspector()
    table = dynamic.plant.GetBodyByName('table')
    for body in dynamic.object_bodies.values():
        assert body.index() in dynamic.plant.GetFloatingBaseBodies()
        for cid in dynamic.plant.GetCollisionGeometriesForBody(body):
            for tid in dynamic.plant.GetCollisionGeometriesForBody(table):
                assert not inspector.CollisionFiltered(cid, tid)


def test_inactive_object_is_collision_active_with_every_gripper_shape(scene):
    cfg = copy.deepcopy(scene.config)
    cfg['objects']['additional_cups'] = {'cup_b': dict(copy.deepcopy(cfg['objects']['cup']), center=[.68, .85, .84])}
    current = build_scene(config=cfg, visualize=False)
    inspector = current.scene_graph.model_inspector()
    cup = current.plant.GetCollisionGeometriesForBody(current.object_bodies['cup_b'])[0]
    for index in current.plant.GetBodyIndices(current.gripper_instance):
        for gid in current.plant.GetCollisionGeometriesForBody(current.plant.get_body(index)):
            assert not inspector.CollisionFiltered(cup, gid)


def test_task_adapter_rejects_unknown_primitives_and_object_id_aliasing():
    from src.multiobject_grounding import validate_task
    spec = load_scene_config('configs/multiobject_grounding.yaml')
    validate_task(spec)
    bad = copy.deepcopy(spec)
    bad['actions'][0]['primitive'] = 'deliver_anywhere'
    with pytest.raises(ValueError, match='primitive'):
        validate_task(bad)
    bad = copy.deepcopy(spec)
    bad['actions'][1]['id'] = bad['actions'][0]['id']
    with pytest.raises(ValueError, match='Unique action'):
        validate_task(bad)


def test_large_graph_chain_uses_only_real_overlap_edges():
    from src.geometry_outbound_gcs import select_region_chain
    records = [{'center': [i, 0.]} for i in range(5)]
    edges = [(0, 1), (1, 2), (2, 4), (0, 3), (3, 4)]
    chain = select_region_chain(records, edges, [0], [4])
    assert chain[0] == 0 and chain[-1] == 4
    assert all(pair in edges for pair in zip(chain, chain[1:]))
    with pytest.raises(RuntimeError, match='overlap chain'):
        select_region_chain(records, [(0, 1)], [0], [4])
