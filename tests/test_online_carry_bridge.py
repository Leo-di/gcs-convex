import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.online_carry_bridge import closest_points,bridge


def record(low,high):
    r=HPolyhedron.MakeBox(np.full(10,low),np.full(10,high))
    return {'A':r.A().tolist(),'b':r.b().tolist(),'accepted':True}


CONFIG={'gcs_base_planning':{'regions':[{'name':'room','lower':[-4,-4],'upper':[4,4],'yaw_bounds':[-4,4]}]}}


def test_closest_set_qp_geometry():
    a=HPolyhedron.MakeBox(np.array([0.,0]),np.array([1.,1]));b=HPolyhedron.MakeBox(np.array([3.,0]),np.array([4.,1]))
    x,y,d=closest_points(a,b,np.ones(2))
    assert abs(d-2)<1e-5 and a.PointInSet(x,1e-6) and b.PointInSet(y,1e-6)


def test_bridge_requires_accepted_region_and_respects_call_budget():
    old=[record(-2,-1),record(1,2)]
    def reject(*args):return dict(record(-3,3),accepted=False)
    regions,r=bridge(CONFIG,old,{},np.full(10,-1.5),np.full(10,1.5),max_calls=2,grow_fn=reject)
    assert len(regions)==2 and len(r['attempts'])==2 and not r['after']['region_graph_connected']
    def accept(*args):return record(-1.5,1.5)
    regions,r=bridge(CONFIG,old,{},np.full(10,-1.5),np.full(10,1.5),grow_fn=accept)
    assert len(old)==2 and len(regions)==3 and len(r['attempts'])==1 and r['after']['region_graph_connected']


def test_missing_endpoint_does_not_get_replaced_by_a_bridge_chord():
    def unexpected(*args):raise AssertionError('Missing endpoint must be handled separately')
    regions,r=bridge(CONFIG,[record(-2,-1)],{},np.full(10,-1.5),np.full(10,1.5),grow_fn=unexpected)
    assert not r['after']['endpoints_covered'] and r['attempts']==[]
