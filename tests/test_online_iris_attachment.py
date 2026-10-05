import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.online_iris_attachment import attach_endpoints, connection_status


def box(lower, upper):
    r=HPolyhedron.MakeBox(np.full(7,lower),np.full(7,upper))
    return dict(A=r.A().tolist(),b=r.b().tolist(),accepted=True)


def test_start_region_can_cover_goal_without_second_call_and_no_cache_leak():
    offline=[box(-2,-1)];calls=[]
    def grow(config,semantic,point,index,max_regions):
        calls.append((semantic,point,index));return dict(box(-1.5,2),candidate_index=index)
    records,info=attach_endpoints({},offline,np.zeros(7),np.ones(7),20,grow)
    assert len(calls)==1 and calls[0][0] and len(records)==2 and len(offline)==1
    assert info['attempts']==1 and info['region_graph_connected']
    assert info['decisions'][1]['action']=='already_covered'
    attach_endpoints({},offline,np.zeros(7),np.ones(7),20,grow)
    assert len(calls)==2  # Query-local regions are not retained by the caller's cover.


def test_rejection_is_retained_but_cannot_supply_coverage_or_retry():
    calls=[]
    def grow(config,semantic,point,index,max_regions):
        calls.append(index);return dict(box(-2,2),accepted=False,candidate_index=index,error='audit failed')
    records,info=attach_endpoints({},[],np.zeros(7),np.ones(7),30,grow)
    assert calls==[30,31] and len(records)==2 and not any(r['accepted'] for r in records)
    assert not info['endpoints_covered'] and not info['region_graph_connected']


def test_covered_endpoints_require_actual_region_connection():
    regions=[HPolyhedron.MakeBox(np.full(7,-2),np.full(7,-1)),HPolyhedron.MakeBox(np.ones(7),np.full(7,2))]
    state=connection_status(regions,np.full(7,-1.5),np.full(7,1.5))
    assert state['endpoints_covered'] and not state['region_graph_connected']
    def unexpected(*args):raise AssertionError('Covered endpoint triggered IRIS')
    records,info=attach_endpoints({},[box(-2,-1),box(1,2)],np.full(7,-1.5),np.full(7,1.5),40,unexpected)
    assert info['attempts']==0 and not info['region_graph_connected']
