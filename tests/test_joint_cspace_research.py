"""Check the research adapter, not a duplicate implementation of IRIS/GCS."""
import json
from pathlib import Path
import numpy as np
from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient, ExtractValue

from src.arm_boundary_semantics import ArmBoundaryOracle
from src.joint_cspace_research import make_scene, setup_checker, state, SEMANTIC_MARGIN


def scenes():
    config = json.loads(Path('tests/fixtures/arm_boundary.json').read_text())['config']
    return make_scene(config), make_scene(config, semantic=True)


def test_parameterization_preserves_all_fixed_coordinates_and_autodiff():
    original, _ = scenes()
    _, parameterize, _, _ = setup_checker(original)
    a = parameterize(np.array([-.9, .2]))
    b = parameterize(np.array([.7, -.4]))
    changed = np.flatnonzero(np.abs(a-b) > 1e-10)
    assert len(changed) == 2
    value = parameterize(InitializeAutoDiff(np.array([.7, -.4])).ravel())
    np.testing.assert_allclose(ExtractValue(value).ravel(), b)
    gradient = ExtractGradient(value)
    np.testing.assert_allclose(gradient[changed], np.eye(2))
    np.testing.assert_allclose(np.delete(gradient, changed, axis=0), 0.)


def test_planning_semantic_geometry_matches_full_arm_oracle():
    original, semantic = scenes()
    physical, to_full, lower, upper = setup_checker(original)
    constrained, to_semantic, _, _ = setup_checker(semantic)
    oracle = ArmBoundaryOracle(original)
    samples = np.vstack([[[-1.6,.45],[1.6,.45],[0.,.45],[0.,-1.4]],
                         np.random.default_rng(809).uniform(lower, upper, (80,2))])
    saw_physical_only = saw_valid = False
    for q in samples:
        free = physical.CheckConfigCollisionFree(to_full(q))
        r = oracle.query(state(original,q))['residual_m']
        valid = constrained.CheckConfigCollisionFree(to_semantic(q))
        assert valid == (free and r >= SEMANTIC_MARGIN)
        saw_physical_only |= free and not valid
        saw_valid |= valid
    assert saw_physical_only and saw_valid
