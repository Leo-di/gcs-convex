"""Verify exact graph optimization and the affine visibility adapter."""
import itertools
import json
from pathlib import Path
import numpy as np
from pydrake.common import Parallelism
from pydrake.planning import VisibilityGraph
from src.visibility_clique_experiment import maximum_clique
from src.arm_cspace_research import setup
from src.joint_cspace_research import make_scene


def test_milp_matches_exhaustive_small_graphs():
    rng=np.random.default_rng(901)
    for n in (1,5,9):
        for density in (0.,.4,1.):
            a=np.triu(rng.random((n,n))<density,1);a=a|a.T
            best=max(len(s) for k in range(n+1) for s in itertools.combinations(range(n),k)
                     if all(a[i,j] for i,j in itertools.combinations(s,2)))
            ids,info=maximum_clique(a)
            assert info['success'] and len(ids)==best


def test_visibility_mapping_preserves_straight_segments():
    config=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())['config']
    scene=make_scene(config,semantic=True);checker,param,lo,hi=setup(scene)
    rng=np.random.default_rng(87);points=[]
    while len(points)<5:
        q=rng.uniform(lo,hi)
        if checker.CheckConfigCollisionFree(param(q)):points.append(q)
    points=np.array(points)
    a=VisibilityGraph(checker,np.asfortranarray(np.column_stack([param(q) for q in points])),Parallelism(1)).toarray()
    for i,j in itertools.combinations(range(5),2):
        np.testing.assert_allclose(param(.3*points[i]+.7*points[j]),.3*param(points[i])+.7*param(points[j]),atol=1e-14)
        assert a[i,j]==checker.CheckEdgeCollisionFree(param(points[i]),param(points[j]))


def test_timeout_incumbent_is_not_reported_as_maximum(monkeypatch):
    from types import SimpleNamespace
    import src.visibility_clique_experiment as experiment
    monkeypatch.setattr(experiment,'milp',lambda *a,**kw:SimpleNamespace(
        success=False,status=1,message='time limit',mip_gap=.5,x=np.array([1.,0.])))
    ids,info=experiment.maximum_clique(np.eye(2,dtype=bool))
    assert ids is None and not info['success'] and info['mip_gap']==.5


def test_parallel_visibility_matches_serial_and_sampling_excludes_covered():
    from src.iterative_visibility_cover import feasible_samples
    from pydrake.geometry.optimization import HPolyhedron
    config=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())['config']
    scene=make_scene(config,semantic=True);checker,param,lo,hi=setup(scene,parallelism=4)
    midpoint=(lo+hi)/2
    blocked=HPolyhedron.MakeBox(lo,midpoint)
    points,trials=feasible_samples(checker,param,lo,hi,8,42,[blocked])
    assert trials>=8
    assert all(not blocked.PointInSet(q) and checker.CheckConfigCollisionFree(param(q)) for q in points)
    full=np.asfortranarray(np.column_stack([param(q) for q in points]))
    serial=VisibilityGraph(checker,full,Parallelism(1)).toarray()
    parallel=VisibilityGraph(checker,full,Parallelism(4)).toarray()
    np.testing.assert_array_equal(serial,parallel)
