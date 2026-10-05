import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.screen_carry_regions import probe_region,SPEC


def box():
    r=HPolyhedron.MakeBox(np.array([-1.,-1.]),np.array([1.,1.]));return {'A':r.A(),'b':r.b(),'seed':[0,0]}


def test_boundary_collision_missed_by_center_is_rejected():
    spec={**SPEC,'uniform_points':0,'random_ray_directions':0}
    r=probe_region(box(),lambda q:.98-q[0],seed=10,spec=spec)
    assert not r['passed'] and r['witness']['kind']=='boundary_ray'
    assert r['witness']['q'][0]>.99


def test_old_witness_is_rechecked_in_current_geometry():
    spec={**SPEC,'uniform_points':0,'random_ray_directions':0}
    r=probe_region(box(),lambda q:1.,known=[[.5,0],[2.,0]],seed=10,spec=spec)
    assert r['passed'] and r['counts']['previous_counterexample']==1


def test_all_sampled_points_stay_inside_region():
    def distance(q):
        assert np.max(abs(q))<=1+1e-8
        return .2
    r=probe_region(box(),distance,seed=123)
    assert r['passed'] and r['counts']['uniform']==64 and r['counts']['boundary_ray']==136
