import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.full_task_portal_socp import make_roadmap,route

def test_disconnected_regions_remain_unreachable():
    regions=[HPolyhedron.MakeBox(np.array([0.,0.]),np.array([1.,1.])),HPolyhedron.MakeBox(np.array([2.,0.]),np.array([3.,1.]))]
    road=make_roadmap(regions,[r.ChebyshevCenter() for r in regions],np.ones(2))
    assert route(regions,road,np.array([.5,.5]),np.array([2.5,.5]),np.ones(2)) is None

def test_continuous_refinement_preserves_region_membership():
    regions=[HPolyhedron.MakeBox(np.array([0.,0.]),np.array([2.,2.])),HPolyhedron.MakeBox(np.array([1.,0.]),np.array([3.,2.]))]
    road=make_roadmap(regions,[r.ChebyshevCenter() for r in regions],np.ones(2))
    r=route(regions,road,np.array([0.,.1]),np.array([3.,1.9]),np.ones(2))
    assert abs(r['cost']-np.hypot(3,1.8))<1e-5
    assert r['cost']<=r['discrete_cost']+1e-6
    for i,k in enumerate(r['regions']):
        assert all(regions[k].PointInSet(np.array(x),1e-5) for x in r['points'][i:i+2])

def test_same_graph_edge_exclusions_are_respected():
    regions=[HPolyhedron.MakeBox(np.array([0.,0.]),np.array([2.,2.])),HPolyhedron.MakeBox(np.array([1.,0.]),np.array([3.,2.]))]
    road=make_roadmap(regions,[r.ChebyshevCenter() for r in regions],np.ones(2),pair_allowed=lambda i,j:False)
    assert route(regions,road,np.array([0.,.1]),np.array([3.,1.9]),np.ones(2)) is None
