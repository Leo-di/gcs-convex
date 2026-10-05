import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.timed_carry_bezier import curvature_energy,graph_solve,fixed_route_solve,trajectory_objective,region_yen_solve,audit


def test_exact_energy_against_quadrature():
    rng=np.random.default_rng(2);p=rng.normal(size=(4,3));h=rng.normal(size=4)
    x=np.r_[p.ravel(),h];t,w=np.polynomial.legendre.leggauss(4);u=(t+1)/2
    c=np.column_stack([p,h]);a=6*np.diff(c,n=2,axis=0)
    values=(1-u[:,None])*a[0]+u[:,None]*a[1]
    expected=np.sum(w*np.sum(values**2,axis=1))/2
    assert np.isclose(curvature_energy(x,3),expected)
    assert curvature_energy(x,3)>=0


def test_matched_regularized_models():
    regions=[HPolyhedron.MakeBox(np.array([-1.]),np.array([2.]))]
    start=np.array([0.]);goal=np.array([1.]);limits=np.array([.5])
    info,gcs=graph_solve(regions,start,goal,limits,smoothness=.1)
    fixed=fixed_route_solve(regions,[0],start,goal,limits,smoothness=.1)
    baseline,b=region_yen_solve(regions,start,goal,limits,smoothness=.1)
    assert info['solved'] and baseline['solved']
    assert abs(info['objective']-trajectory_objective(fixed,1,.1))<1e-5
    assert abs(info['objective']-baseline['objective'])<1e-5
    assert audit(gcs,regions,start,goal,limits)['kinematics_passed']
    raw=fixed_route_solve(regions,[0],start,goal,limits)
    assert curvature_energy(fixed[0]['x'],1)<curvature_energy(raw[0]['x'],1)


def test_region_search_preserves_incumbent():
    regions=[HPolyhedron.MakeBox(np.array([-1.]),np.array([2.]))]
    checkpoints=[]
    info,solution=region_yen_solve(regions,np.array([0.]),np.array([1.]),
        np.array([.5]),smoothness=.01,
        on_incumbent=lambda segments,cost,count:checkpoints.append((segments,cost,count)))
    assert checkpoints and checkpoints[-1][0]==solution
    assert checkpoints[-1][1]==info['objective']
    assert checkpoints[-1][2]==1
