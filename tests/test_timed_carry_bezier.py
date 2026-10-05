import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.timed_carry_bezier import graph_solve,fixed_route_solve,audit,yen_paths

def test_gcs_and_restriction_have_same_time_and_physical_c1_checks():
    regions=[HPolyhedron.MakeBox(np.array([0.]),np.array([2.]))];start=np.array([0.]);goal=np.array([1.]);limits=np.array([.5])
    info,path=graph_solve(regions,start,goal,limits,10.,10);fixed=fixed_route_solve(regions,[0],start,goal,limits,10.)
    assert info['solved'] and fixed is not None
    assert abs(info['duration']-fixed[-1]['x'][-1])<1e-4
    assert info['duration']>=2.-1e-6
    assert audit(path,regions,start,goal,limits)['kinematics_passed']
    assert audit(fixed,regions,start,goal,limits)['kinematics_passed']
    assert fixed_route_solve(regions,[0],start,goal,limits,1.9) is None

def test_yen_is_loopless_and_cost_ordered_with_alternatives():
    adj=[{1:1.,2:2.,3:8.},{2:.5,3:3.},{1:.5,3:1.},{}]
    paths=list(yen_paths(adj,0,3));assert [p for _,p in paths]==[(0,1,2,3),(0,2,3),(0,1,3),(0,2,1,3),(0,3)]
    assert all(len(set(p))==len(p) for _,p in paths)

def test_multiple_regions_check_time_continuity_and_detect_tampering():
    import copy
    regions=[HPolyhedron.MakeBox(np.array([0.]),np.array([1.2])),HPolyhedron.MakeBox(np.array([.8]),np.array([2.]))];start=np.array([0.]);goal=np.array([2.]);limits=np.array([1.])
    path=fixed_route_solve(regions,[0,1],start,goal,limits,10.)
    assert path is not None and audit(path,regions,start,goal,limits)['kinematics_passed']
    broken=copy.deepcopy(path);broken[1]['x'][4]+=.1
    assert not audit(broken,regions,start,goal,limits)['kinematics_passed']
