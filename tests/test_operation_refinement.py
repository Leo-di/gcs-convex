import numpy as np
import pytest

from src.operation_refinement import OperationRefinementError, refine_cover, split_domain


def record(a=(0., 0.), b=(1., 1.)):
    a, b = np.asarray(a), np.asarray(b)
    return dict(lower=(np.minimum(a, b)-.2).tolist(), upper=(np.maximum(a, b)+.2).tolist(),
                guide_start=a.tolist(), guide_end=b.tolist())


def audit(failed=(), *, semantic_only=False):
    return {'region_failures': [dict(region_index=i, collisions=0 if semantic_only else 1,
                                    raw_grasp_violations=0, semantic_violations=int(semantic_only),
                                    counterexamples=[{'arm_q': [.1, .9]}]) for i in failed]}


def test_refinement_retains_shared_endpoints_and_excludes_off_guide_witness():
    old = record()
    children = split_domain(old, [.1, .9])
    np.testing.assert_array_equal(children[0]['guide_start'], old['guide_start'])
    np.testing.assert_array_equal(children[0]['guide_end'], children[1]['guide_start'])
    np.testing.assert_array_equal(children[1]['guide_end'], old['guide_end'])
    for child in children:
        assert np.all(np.array(child['lower']) >= old['lower'])
        assert np.all(np.array(child['upper']) <= old['upper'])
        assert not (np.all(np.array([.1, .9]) >= child['lower']) and
                    np.all(np.array([.1, .9]) <= child['upper']))


def test_only_failed_region_is_regenerated_and_final_audit_has_fresh_seed(tmp_path):
    clean, bad = object(), object()
    r0, r1 = record(), record()
    seeds, calls = [], []
    def check(records, seed):
        seeds.append(seed)
        return audit([1]) if len(seeds) == 1 else audit()
    def grow(children):
        calls.append(children)
        return [object() for _ in children], children, 5.
    regions, records, _, report = refine_cover(([clean, bad], [r0, r1], 5.), check, grow,
                                               seed=100, artifact=tmp_path/'repair.json')
    assert regions[0] is clean and records[0] is r0
    assert bad not in regions
    assert len(calls) == 1 and len(calls[0]) == 2
    assert seeds == [100, 102, 103]
    assert report['status'] == 'accepted' and report['repaired_regions'] == 1


def test_new_validation_failure_becomes_evidence_and_requires_another_audit():
    seeds = []
    def check(records, seed):
        seeds.append(seed)
        return audit([0]) if seed == 11 else audit()
    def grow(children):
        return [object() for _ in children], children, 1.
    _, _, _, report = refine_cover(([object()], [record()], 1.), check, grow, seed=10)
    assert seeds == [10, 11, 12, 13]
    assert report['rounds'][0]['validation']['region_failures']
    assert report['status'] == 'accepted'


def test_exhausted_repair_never_accepts_even_when_endpoints_are_preserved(tmp_path):
    def check(records, seed):
        return audit([0])
    def grow(children):
        return [object() for _ in children], children, 1.
    with pytest.raises(OperationRefinementError, match='no infeasibility claim') as exc:
        refine_cover(([object()], [record()], 1.), check, grow, seed=2,
                     max_rounds=1, artifact=tmp_path/'rejected.json')
    assert exc.value.report['status'] == 'rejected'
    assert exc.value.report['final_audit'] is None


def test_zero_round_baseline_does_not_regrow_and_still_requires_clean_audit():
    with pytest.raises(OperationRefinementError):
        refine_cover(([object()], [record()], 1.), lambda r, s: audit([0]),
                     lambda r: pytest.fail('Baseline must not repair'), seed=4, max_rounds=0)


def test_geometry_policy_does_not_train_on_semantic_violations():
    _, _, _, report = refine_cover(([object()], [record()], 1.),
        lambda r, s: audit([0], semantic_only=True),
        lambda r: pytest.fail('Geometry baseline must not enforce motion semantics'),
        seed=4, semantic=False)
    assert report['status'] == 'accepted'


def test_region_limit_prevents_growth_before_iris_is_called():
    with pytest.raises(OperationRefinementError, match='region budget'):
        refine_cover(([object()], [record()], 1.), lambda r, s: audit([0]),
                     lambda r: pytest.fail('Region budget must be checked first'),
                     seed=4, max_regions=1)


def test_colliding_required_guide_is_not_silently_removed():
    def grow(children):
        raise RuntimeError('IRIS containment endpoint collides')
    with pytest.raises(OperationRefinementError, match='containment endpoint collides'):
        refine_cover(([object()], [record()], 1.), lambda r, s: audit([0]), grow, seed=4)


def test_an_over_budget_clean_audit_is_not_accepted(monkeypatch):
    clock = [0.]
    monkeypatch.setattr('src.operation_refinement.time.monotonic', lambda: clock[0])
    def check(records, seed):
        clock[0] += 2.
        return audit()
    with pytest.raises(OperationRefinementError, match='time budget'):
        refine_cover(([object()], [record()], 1.), check,
                     lambda r: pytest.fail('Time budget must stop planning'), seed=3, time_budget_s=1.)


def test_audit_error_is_preserved_and_never_becomes_an_empty_pass(tmp_path):
    def check(records, seed):
        raise RuntimeError('distance query unavailable')
    with pytest.raises(OperationRefinementError, match='distance query unavailable') as exc:
        refine_cover(([object()], [record()], 1.), check,
                     lambda r: pytest.fail('No repair on unknown audit errors'), seed=3,
                     artifact=tmp_path/'failed.json')
    assert exc.value.report['status'] == 'rejected'
