import numpy as np
import pytest
from src.multibranch_payload import pack_regions,component_indices,certify_region,solve_graph
from src.payload_translation_bridge import overlap_edges

def record(center,radius):return dict(center=center,radius=radius,lower=[-1,-4],upper=[5,4])

def test_semantic_certificate_uses_worst_route_slack_and_never_enlarges():
    class Human:
        spec={'id':'a'}
        def current(self):return dict(satisfied=True,motion_slack_m=.31,connections={'a':dict(satisfied=True,motion_slack_m=.8,path_xy=[[0,0]]),'b':dict(satisfied=True,motion_slack_m=.31,path_xy=[[1,1]])})
    bounded,proof=certify_region(record([0,0],.6),Human(),.02,.025)
    assert bounded['radius']==pytest.approx(.29) and set(proof['connections'])=={'a','b'}
    small,_=certify_region(record([0,0],.05),Human(),.02,.025)
    assert small['radius']==.05

def test_blocked_or_insufficient_witness_is_not_admitted():
    class Human:
        spec={'id':'a'}
        def current(self):return dict(satisfied=False,motion_slack_m=-1e-6,path_xy=[])
    bounded,proof=certify_region(record([0,0],1),Human(),.02,.025)
    assert bounded is None and not proof['satisfied']

def test_spatial_packing_preserves_both_endpoint_seeds():
    records=[record([0,0],.1),record([4,0],.1),record([2,1],.8),record([2,1.01],.1),record([2,-1],.8)]
    ids=pack_regions(records,1.0)
    assert 0 in ids and 1 in ids and 2 in ids and 4 in ids and 3 not in ids

def test_gcs_selects_other_branch_when_semantics_remove_shorter_branch():
    records=[record([0,0],.3),record([2,0],.3),record([1,.35],1.05),record([1,-1.5],1.51)]
    edges=overlap_edges(records);assert len(component_indices(4,edges,[0],[1]))==4
    trajectory,graph=solve_graph(records,edges,[0,0,0],[2,0,0])
    assert 2 in graph['selected_region_path']
    safe=[records[i] for i in [0,1,3]];other,choice=solve_graph(safe,overlap_edges(safe),[0,0,0],[2,0,0])
    assert 2 in choice['selected_region_path']
    assert other.value(other.end_time()/2)[1,0]<trajectory.value(trajectory.end_time()/2)[1,0]-.1
    assert graph['selected_after_solve'] and choice['selected_after_solve']
