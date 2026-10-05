import numpy as np
import pytest
from pydrake.geometry.optimization import HPolyhedron
from src.online_carry_iris import attach_carry_endpoints


def box(low, high, accepted=True):
    r=HPolyhedron.MakeBox(np.full(10,low),np.full(10,high))
    return dict(A=r.A().tolist(),b=r.b().tolist(),accepted=accepted,
                attached_object_included=True,independent_holdout_audit={'collisions':0})


CONFIG={'gcs_base_planning':{'regions':[{'name':'room','lower':[-3,-3],
                                      'upper':[3,3],'yaw_bounds':[-3,3]}]}}


def test_rejected_growth_cannot_enable_carry_or_mutate_cache():
    original=[box(-2,-1)]
    calls=[]
    def reject(config,anchor,seed,label,index):
        calls.append(index)
        return box(-3,3,False)
    records,report=attach_carry_endpoints(CONFIG,original,{'attached':True},
        np.zeros(10),np.ones(10),grow_fn=reject)
    assert len(calls)==2 and len(records)==len(original)==1
    assert not report['after']['endpoints_covered']
    assert len(report['attempts'])==2


def test_new_start_region_may_cover_goal_and_is_query_local():
    original=[box(-2,-1)]
    def grow(*args):return box(-1.5,2)
    records,report=attach_carry_endpoints(CONFIG,original,{'attached':True},
        np.zeros(10),np.ones(10),grow_fn=grow)
    assert len(records)==2 and len(original)==1
    assert len(report['attempts'])==1 and report['after']['region_graph_connected']


def test_wrong_mode_and_known_colliding_cache_are_rejected():
    for region in [dict(box(-2,2),attached_object_included=False),
                   dict(box(-2,2),independent_holdout_audit={'collisions':1})]:
        with pytest.raises(ValueError):
            attach_carry_endpoints(CONFIG,[region],{'attached':True},np.zeros(10),np.ones(10))
    with pytest.raises(ValueError):
        attach_carry_endpoints(CONFIG,[],{'attached':False},np.zeros(10),np.ones(10))
