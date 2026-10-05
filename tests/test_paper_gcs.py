from dataclasses import replace
import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.paper_gcs import Config,solve,fixed_route,objective,audit,derivative_matrix,smoothing_matrix,evaluate


def test_all_cost_terms_same_graph_and_restriction():
    r=[HPolyhedron.MakeBox(np.array([-1.]),np.array([2.]))];q0=np.array([0.]);q1=np.array([1.]);v=np.array([.5]);cfg=Config(length_weight=.2,energy_weight=.3)
    info,s=solve(r,q0,q1,v,cfg);fixed=fixed_route(r,[0],q0,q1,v,cfg)
    assert info['solved'] and fixed is not None
    assert abs(info['objective']-objective(fixed,1,cfg))<1e-4
    assert abs(info['objective']-info['solver_objective'])<1e-4
    assert audit(s,r,q0,q1,v,cfg)['passed']
    low,_=solve(r,q0,q1,v,cfg,relaxation_only=True)
    assert abs(low['solver_objective']-info['objective'])<1e-4
    fail,_=solve(r,q0,q1,v,replace(cfg,horizon=1.9))
    assert not fail['solved']


def test_c2_continuity_including_clock():
    r=[HPolyhedron.MakeBox(np.array([0.]),np.array([.7])),HPolyhedron.MakeBox(np.array([.3]),np.array([1.]))]
    cfg=Config();q0=np.array([0.]);q1=np.array([1.]);v=np.array([.5]);s=fixed_route(r,[0,1],q0,q1,v,cfg)
    assert s and audit(s,r,q0,q1,v,cfg)['passed']
    # Perturb second clock control point: C0 remains but C2 seam breaks.
    s[1]['x'][9]+=.01
    assert not audit(s,r,q0,q1,v,cfg)['passed']


def test_bernstein_derivatives_and_smoothing_upper_bound():
    rng=np.random.default_rng(5);p=rng.normal(size=(7,2));h=np.arange(7)/6;cfg=Config();x=np.r_[p.ravel(),h]
    u=.37;eps=1e-5
    assert np.max(abs((evaluate(p,u+eps)-evaluate(p,u-eps))/(2*eps)-evaluate(p,u,1)))<1e-6
    t,w=np.polynomial.legendre.leggauss(10)
    exact=sum(wi*(np.sum(evaluate(p,(ui+1)/2,2)**2)+evaluate(h,(ui+1)/2,2)**2)/2 for ui,wi in zip(t,w))
    assert np.sum((smoothing_matrix(2,cfg)@x)**2)>=exact-1e-8
