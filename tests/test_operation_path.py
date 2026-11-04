from types import SimpleNamespace

import numpy as np
import pytest
from pydrake.geometry.optimization import HPolyhedron

from src.operation_path import cover_edge_validator, shortcut_path


def box(lo, hi):
    return HPolyhedron.MakeBox(np.asarray(lo, float), np.asarray(hi, float))


def test_covered_endpoints_cannot_shortcut_a_nonconvex_room_or_obstacle():
    # A U-shaped cover: the direct line crosses the forbidden centre.
    regions = [box([0, 0], [1, 3]), box([0, 2], [3, 3]), box([2, 0], [3, 3])]
    valid = cover_edge_validator(regions, tolerance=0.)
    path = np.array([[.5, .5], [.5, 2.5], [2.5, 2.5], [2.5, .5]])
    assert not valid(path[0], path[-1])
    result, report = shortcut_path(path, valid)
    assert np.array_equal(result[[0, -1]], path[[0, -1]])
    assert all(valid(a, b) for a, b in zip(result, result[1:]))
    assert report['rejected_shortcuts'] > 0
    assert report['output_length_rad'] <= report['input_length_rad']


def test_cover_accepts_a_chain_but_rejects_an_interior_gap():
    a, b = np.array([0., 0.]), np.array([3., 0.])
    chain = [box([i, -1], [i+1, 1]) for i in range(3)]
    assert cover_edge_validator(chain, tolerance=0.)(a, b)
    assert not cover_edge_validator([chain[0], chain[2]], tolerance=0.)(a, b)
    # Stationary points outside the cover must also be rejected.
    assert not cover_edge_validator(chain)(np.array([5., 0.]), np.array([5., 0.]))


def test_duplicate_and_dense_linear_sampling_have_same_execution_geometry():
    from src.green_room_gcs import append_motion
    valid = cover_edge_validator([box([-1]*7, [2]*7)])
    def frames(points):
        compact, _ = shortcut_path(points, valid)
        result = []
        append_motion(result, np.zeros((len(compact), 3)), compact,
                      np.tile([-.055, .055], (len(compact), 1)), 'reach')
        return result
    sparse = np.array([[0.]*7, [1.]*7])
    dense = np.repeat(np.linspace(sparse[0], sparse[-1], 701), 2, axis=0)
    assert frames(sparse) == frames(dense)


def test_shortcut_budget_falls_back_without_unchecked_edges():
    path = np.array([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    def reject(a, b):
        return False
    result, report = shortcut_path(path, reject, max_checks=1)
    assert np.array_equal(result, path)
    assert report['edge_checks'] == 1
    with pytest.raises(ValueError):
        shortcut_path([[np.nan]], reject)


def test_fine_cover_does_not_consume_the_budget_at_its_first_vertex():
    # Only short spans can be joined, as with tightly refined IRIS domains.
    points = np.arange(1001, dtype=float)[:, None]
    result, report = shortcut_path(points, lambda a, b: b[0]-a[0] <= 20)
    assert len(result) < 100
    assert report['edge_checks'] < 200
    assert np.max(np.diff(result[:, 0])) <= 20


@pytest.mark.parametrize('obstacle_kind', ['raw_finger', 'semantic'])
def test_pregrasp_shortcuts_keep_raw_collision_and_semantic_checks(monkeypatch, obstacle_kind):
    import src.safe_pregrasp as module
    scene = SimpleNamespace(
        config={'whole_body_planning': {'joint_lower_limits': [-2.]*7,
                                       'joint_upper_limits': [2.]*7}},
        plant=SimpleNamespace(SetFreeBodyPose=lambda *args: None),
        plant_context=None, cup_body=None, cup_initial_pose=None,
        gripper_open_positions=[-.055, .055])
    current, visited = {}, []
    monkeypatch.setattr(module, 'set_robot_configuration', lambda scene, base, q, grip: current.update(q=q))
    monkeypatch.setattr(module, 'evaluate_collision_status',
                        lambda *args, **kwargs: SimpleNamespace(collision_free=True))
    def raw(scene):
        q = current['q']
        # Physical obstacle blocks the direct line, semantic obstacle is separate.
        bad = obstacle_kind == 'raw_finger' and .4 < q[0] < .6 and abs(q[1]) < .12
        return {'pairs': [{'body': 'left_finger', 'distance_m': 0. if bad else .01}]}
    monkeypatch.setattr(module, 'measure_cup_gripper', raw)
    monkeypatch.setattr(module, 'summarize_grasp_audit', lambda rows: {'satisfied': True})
    def semantic(q):
        visited.append(q.copy())
        forbidden = obstacle_kind == 'semantic' and .4 < q[0] < .6 and abs(q[1]) < .12
        return abs(q[2]) < .8 and not forbidden
    start, goal = np.zeros(7), np.array([1., 0., 0., 0., 0., 0., 0.])
    report = {}
    result = module.plan_pregrasp(scene, [0, 0, 0], start, goal,
                                  state_constraint=semantic, diagnostics=report)
    assert np.array_equal(result[[0, -1]], [start, goal])
    assert np.all(result >= -2) and np.all(result <= 2)
    assert np.max(np.abs(np.diff(result, axis=0))) <= .025+1e-12
    assert not any(.4 < q[0] < .6 and abs(q[1]) < .12 for q in result)
    assert all(abs(q[2]) < .8 for q in result)
    assert visited and report['final_edge_audit']
    assert report['method'] == 'rrt_with_validated_shortcuts'
    invalid_goal = goal.copy()
    invalid_goal[0] = 3.
    with pytest.raises(RuntimeError, match='endpoint rejected'):
        module.plan_pregrasp(scene, [0, 0, 0], start, invalid_goal)
