import copy
import numpy as np
import pytest

from src.build_scene import build_scene
from src.operation_refinement_study import prepare
from src.operation_semantics_study import audit_path
from src.green_room_gcs import make_operation_cover, region_holdout
from src.scene_spec import PROJECT_ROOT, load_scene_config


def test_real_collision_free_connection_has_legal_endpoints_and_illegal_interior():
    probe = load_scene_config('configs/operation_semantics_probe.yaml')
    scene = build_scene(config=load_scene_config(PROJECT_ROOT/probe['scene']), visualize=False)
    audit, _ = audit_path(scene, probe['base_q'], [probe['start_arm_q'], probe['goal_arm_q']])
    assert audit['geometry_satisfied']
    assert audit['minimum_collision_distance_m'] > .01
    assert audit['goal_semantics_satisfied']
    assert audit['minimum_room_residual_m'] < -.03
    assert audit['semantic_violations'] > 0
    # Retracting before sweeping is a geometric witness, not a planner result.
    a, b = np.asarray(probe['start_arm_q']), np.asarray(probe['goal_arm_q'])
    inner_a, inner_b = a.copy(), b.copy()
    inner_a[1] = inner_b[1] = -.35
    valid, _ = audit_path(scene, probe['base_q'], [a, inner_a, inner_b, b])
    assert valid['geometry_satisfied'] and valid['motion_semantics_satisfied']


def test_matched_repair_protocol_needs_no_prior_outputs_and_preserves_scene_input():
    original = load_scene_config('configs/scene_semantic_choice_v6.yaml')
    before = copy.deepcopy(original)
    cfg, protocol = prepare(original, 'narrow')
    assert original == before
    assert cfg['objects'] == original['objects']
    assert cfg['world']['walls'][:-1] == original['world']['walls']
    assert len(protocol['candidates']) == 36
    assert protocol['candidates'][0]['base_q'] == [.45, 0., 0.]
    assert protocol['candidates'][0]['grasp_rpy'] == [0., 0., 0.]
    with pytest.raises(ValueError):
        prepare(original, 'narrow', budget_seconds=0)


def test_near_boundary_guide_gets_a_positive_semantic_domain_without_fixed_padding_floor():
    cfg = load_scene_config('configs/scene_semantic_choice_v6.yaml')
    scene = build_scene(config=cfg, visualize=False)
    # About 3 mm room clearance, too little for the original 1 mrad padding
    # in all seven joints. Subdivision alone never reduced that padding.
    q = np.array([-.4, -.5, 0., 0., 0., 0., 0.])
    regions, records, _ = make_operation_cover(scene, [.45, 0., 0.],
        np.array([q, q+1e-5]), scene.gripper_open_positions)
    assert records and all(r['whole_box_semantic_lower_bound_m'] >= 1e-6 for r in records)
    assert any(r['corridor_padding_rad'] < cfg['green_room']['corridor_padding_rad'] for r in records)
    assert all(np.all(np.array(r['upper']) > r['lower']) for r in records)
    assert any(r.PointInSet(q) for r in regions)
    assert any(r.PointInSet(q+1e-5) for r in regions)


def test_audit_keeps_failure_attribution_beyond_the_global_example_display_limit():
    cfg = load_scene_config('configs/scene_semantic_choice_v6.yaml')
    cfg['green_room']['random_region_samples'] = 1
    scene = build_scene(config=cfg, visualize=False)
    q = np.array([0., -.53, 0., 0., 0., 0., 0.])
    records = [dict(lower=(q-1e-6).tolist(), upper=(q+1e-6).tolist()) for _ in range(25)]
    audit = region_holdout(scene, [.45, 0., 0.], records, scene.gripper_open_positions)
    assert len(audit['counterexamples']) == 20
    assert [r['region_index'] for r in audit['region_failures']] == list(range(25))
    assert all(r['counterexamples'][0]['region_index'] == r['region_index'] for r in audit['region_failures'])


@pytest.mark.parametrize('samples', [0, -1])
def test_empty_region_audit_cannot_pass_vacuously(samples):
    cfg = load_scene_config('configs/scene_semantic_choice_v6.yaml')
    cfg['green_room']['random_region_samples'] = samples
    scene = build_scene(config=cfg, visualize=False)
    q = np.asarray(cfg['pick_task']['navigation_arm_q'])
    records = [dict(lower=(q-1e-6).tolist(), upper=(q+1e-6).tolist())]
    with pytest.raises(ValueError, match='positive sample counts'):
        region_holdout(scene, [.45, 0., 0.], records, scene.gripper_open_positions)
