import numpy as np
import pytest
from pydrake.geometry.optimization import HPolyhedron,Hyperellipsoid,VPolytope
from pydrake.planning import IrisNp2Options,IrisParameterizationFunction
from src.iris_conditioning import solve_capped_iris


def setup():
    center=np.array([.3,-1.1,1.6,-1.2,2.2,-.1,1.7]);start=center+np.array([.03,-.02,.01,.01,-.02,.01,.03]);r=6e-5
    import itertools
    v=np.column_stack([start,*[center+r*np.array(s) for s in itertools.product((-1,1),repeat=7)]])
    domain=HPolyhedron(VPolytope(v));ellipse=Hyperellipsoid.MakeHypersphere(r*.1,center)
    options=IrisNp2Options();options.parameterization=IrisParameterizationFunction(lambda q:q,7)
    options.sampled_iris_options.containment_points=np.column_stack([center,start]);options.sampled_iris_options.configuration_space_margin=1e-6
    return center,start,r,domain,ellipse,options


def test_original_success_has_no_retry_or_coordinate_change():
    c,s,r,d,e,o=setup();calls=[]
    region,meta=solve_capped_iris(None,e,d,o,lambda q:q,r,solve=lambda *args:calls.append(args) or d)
    assert region is d and meta is None and len(calls)==1


def test_similarity_preserves_physical_domain_endpoints_and_margin():
    c,s,r,d,e,o=setup();calls=[];old_points=o.sampled_iris_options.containment_points.copy()
    def solve(checker,ellipse,domain,options):
        calls.append(dict(domain=domain,margin=options.sampled_iris_options.configuration_space_margin,points=options.sampled_iris_options.containment_points.copy()))
        if len(calls)==1:raise RuntimeError('maximum inscribed ellipse problem')
        domain.MaximumVolumeInscribedEllipsoid() # Real solver, normalized full-dimensional pyramid.
        return domain
    region,meta=solve_capped_iris(None,e,d,o,lambda q:q,r,solve=solve)
    assert len(calls)==2 and np.isclose(calls[1]['margin']*meta['scale'],1e-6)
    Q=np.array(meta['rotation']);assert np.allclose(Q.T@Q,np.eye(7),atol=1e-13)
    assert region.PointInSet(c,1e-10) and region.PointInSet(s,1e-10)
    rng=np.random.default_rng(8131)
    for _ in range(200):
        alpha=rng.uniform(0,1);point=(1-alpha)*s+alpha*(c+r*rng.uniform(-1,1,7))
        z=Q.T@(point-c)/meta['scale']
        assert d.PointInSet(point,1e-10)==calls[1]['domain'].PointInSet(z,1e-10)==region.PointInSet(point,1e-10)
    assert np.array_equal(o.sampled_iris_options.containment_points,old_points)
    assert o.sampled_iris_options.configuration_space_margin==1e-6


def test_other_errors_and_uncapped_domains_do_not_retry():
    c,s,r,d,e,o=setup()
    for cap,message in [(None,'maximum inscribed ellipse problem'),(r,'collision seed invalid')]:
        calls=[]
        def fail(*args):calls.append(1);raise RuntimeError(message)
        with pytest.raises(RuntimeError,match=message):solve_capped_iris(None,e,d,o,lambda q:q,cap,solve=fail)
        assert len(calls)==1


def test_failed_retry_restores_original_options():
    c,s,r,d,e,o=setup();points=o.sampled_iris_options.containment_points.copy();calls=[]
    def fail(*args):
        calls.append(1)
        raise RuntimeError('maximum inscribed ellipse problem' if len(calls)==1 else 'retry failed')
    with pytest.raises(RuntimeError,match='retry failed'):solve_capped_iris(None,e,d,o,lambda q:q,r,solve=fail)
    assert np.array_equal(o.sampled_iris_options.containment_points,points)
    assert o.sampled_iris_options.configuration_space_margin==1e-6
