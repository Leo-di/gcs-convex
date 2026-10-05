import json
from pathlib import Path

import numpy as np
from pydrake.autodiffutils import InitializeAutoDiff, ExtractGradient, ExtractValue
from pydrake.geometry.optimization import HPolyhedron

from src.arm_cspace_research import setup, full_dimensional_radius, BASE, SEMANTIC_MARGIN
from src.joint_cspace_research import make_scene
from src.arm_boundary_semantics import ArmBoundaryOracle


def config():
    return json.loads(Path('tests/fixtures/arm_boundary.json').read_text())['config']


def test_all_seven_joint_coordinates_are_independent():
    scene=make_scene(config())
    _,parameterize,lower,upper=setup(scene)
    q=(lower+upper)/2
    value=parameterize(InitializeAutoDiff(q).ravel())
    gradient=ExtractGradient(value)
    assert np.linalg.matrix_rank(gradient)==7
    rows=np.flatnonzero(np.linalg.norm(gradient,axis=1))
    assert len(rows)==7
    np.testing.assert_allclose(gradient[rows],np.eye(7))
    original=ExtractValue(value).ravel()
    changed=parameterize(q+np.arange(1,8)*.05)
    fixed=np.setdiff1d(np.arange(len(original)),rows)
    np.testing.assert_allclose(original[fixed],changed[fixed])


def test_semantic_adapter_in_seven_dimensions():
    original=make_scene(config());semantic=make_scene(config(),semantic=True)
    checker,param,lower,upper=setup(original)
    constrained,sparam,_,_=setup(semantic)
    oracle=ArmBoundaryOracle(original)
    home=original.arm_home_positions.copy()
    endpoint=home.copy();endpoint[0]=1.6
    samples=np.vstack([home,endpoint,np.random.default_rng(832).uniform(lower,upper,(150,7))])
    valid_count=invalid_semantic_count=0
    for q in samples:
        physical=checker.CheckConfigCollisionFree(param(q))
        residual=oracle.query(np.r_[BASE,q])['residual_m']
        observed=constrained.CheckConfigCollisionFree(sparam(q))
        assert observed==(physical and residual>=SEMANTIC_MARGIN)
        valid_count+=int(observed)
        invalid_semantic_count+=int(physical and not observed)
    assert valid_count and invalid_semantic_count


def test_positive_inscribed_radius_distinguishes_a_slice():
    full=HPolyhedron.MakeBox(-np.ones(7),np.ones(7))
    assert full_dimensional_radius(full)>.99
    upper=np.ones(7);lower=-np.ones(7);upper[-1]=lower[-1]=0
    thin=HPolyhedron.MakeBox(lower,upper)
    assert abs(full_dimensional_radius(thin))<1e-8
