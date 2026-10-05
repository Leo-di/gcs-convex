from src.inspection_polyline_audit import inspect_polyline

def row(x,y,attached=False):return {'base_q':[x,y,0.],'attached':attached}
SITE=[{'name':'test','xy':[0,0]}]

def test_between_samples_counts_as_a_visit():
    r=inspect_polyline([row(-1,0),row(1,0),row(2,0,True)],SITE)
    assert r['satisfied'] and r['minimum_distance_m']==0 and abs(r['first_witness']['segment_fraction']-.49)<1e-10

def test_after_grasp_crossing_does_not_count():
    assert not inspect_polyline([row(-1,0),row(-1,0,True),row(1,0,True)],SITE)['satisfied']

def test_stationary_and_outside_segments():
    assert inspect_polyline([row(0,0),row(0,0),row(1,0,True)],SITE)['satisfied']
    assert not inspect_polyline([row(-1,.021),row(1,.021),row(2,0,True)],SITE)['satisfied']
    assert not inspect_polyline([row(-1,0),row(1,0),row(2,0,True)],[])['satisfied']


def test_do_not_invent_visit_across_grasp_mode_transition():
    assert not inspect_polyline([row(-1,0),row(1,0,True)],SITE)['satisfied']
    assert not inspect_polyline([row(-1,0),row(-.02,0,True)],SITE)['satisfied']
