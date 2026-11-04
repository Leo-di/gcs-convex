"""Real collision geometry and bounded-search equivalence for lookahead."""
import copy
import pytest
from pydrake.math import RigidTransform

from src.placement_lookahead import rank_placements
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import (
    make_config, initial_world, world_config, successor, destination_occupancy,
    search, TransportRejected, digest, validate_task,
)
from src.build_scene import build_scene


def setup_task():
    spec = load_scene_config('configs/stateful_transport_grounding.yaml')
    cfg = make_config(spec)
    return spec, cfg, initial_world(cfg)


def test_ranks_by_complete_object_geometry_and_keeps_every_option():
    spec, cfg, world = setup_task()
    original = digest(world)
    # Distinct centers still overlap; a center-equality shortcut is insufficient.
    spec['actions'][0]['placements'][0][0] -= .03
    order, rows = rank_placements(cfg, world, spec, 0)
    assert order == [1, 0]
    assert rows[0]['exhausted_actions'] == 1
    assert min(p['distance_m'] for p in rows[0]['future'][0]['options'][0]['pairs']) < 0
    assert digest(world) == original
    assert sorted(order) == [0, 1]


def test_ties_and_all_estimated_blocked_keep_original_order():
    spec, cfg, world = setup_task()
    spec['actions'][0]['placements'] = [[1.66, .85, .84], [1.65, .85, .84]]
    order, rows = rank_placements(cfg, world, spec, 0)
    assert order == [0, 1] and all(r['exhausted_actions'] == 1 for r in rows)
    assert rank_placements(cfg, world, spec, 1) == ([0], [])


def test_search_reduces_ground_calls_with_same_final_world(tmp_path):
    spec, cfg, world = setup_task()
    calls = []
    def ground(config, current, action, placement, task, folder):
        calls.append(action['id'])
        scene = build_scene(config=world_config(config, current, action['object_id']), visualize=False)
        pose = RigidTransform(placement)
        if not destination_occupancy(scene, pose)['satisfied']:
            raise TransportRejected('occupied', {})
        frame = dict(copy.deepcopy(current['robot']), phase=action['id']+'.test')
        return [frame], successor(current, action['object_id'], pose.GetAsMatrix4(),
                                  action['destination_support'], frame), {}
    baseline = search(cfg, spec, tmp_path, ground=ground)
    baseline_calls = len(calls)
    spec['placement_lookahead'] = True
    calls.clear()
    optimized = search(cfg, spec, tmp_path, ground=ground)
    assert baseline['satisfied'] and optimized['satisfied']
    assert digest(baseline['final_world']) == digest(optimized['final_world'])
    assert baseline_calls == 4 and len(calls) == 2
    assert optimized['events'][0]['order'] == [1, 0]


def test_estimate_never_eliminates_a_candidate(tmp_path):
    spec, cfg, world = setup_task()
    spec['placement_lookahead'] = True
    spec['actions'][0]['placements'] = [[1.66, .85, .84], [1.65, .85, .84]]
    seen = []
    def reject(config, current, action, placement, task, folder):
        seen.append(placement)
        raise TransportRejected('motion_rejected', {})
    result = search(cfg, spec, tmp_path, ground=reject)
    assert not result['satisfied']
    assert seen == spec['actions'][0]['placements']


def test_flag_rejects_non_boolean():
    spec, _, _ = setup_task()
    spec['placement_lookahead'] = 'false'
    with pytest.raises(ValueError, match='Boolean'):
        validate_task(spec)
