import numpy as np
from src.automatic_motion_bridge import classify_bridge_failure,uncovered_samples


def test_only_representation_failures_are_classified_as_repairable():
    assert classify_bridge_failure('endpoint_clearance: not enough margin')=='endpoint_clearance'
    assert classify_bridge_failure('Current-geometry fixed-posture lane cover is disconnected')=='coverage_disconnected'
    assert classify_bridge_failure('Current-geometry outbound cover does not cover start or goal')=='endpoint_not_covered'
    for reason in ['independent numerical audit failed','SNOPT infeasible','solver failed']:
        assert classify_bridge_failure(reason)=='other_solver_or_audit_failure'


def test_gap_detection_respects_disk_and_clipping_box():
    records=[{'center':[0.,0.],'radius':1.,'lower':[-1.,-1.],'upper':[0.,1.]}]
    points=[np.array(p) for p in [[-.2,0],[.2,0],[0,1.1]]]
    result=uncovered_samples(points,records)
    assert len(result)==2
    assert np.allclose(result[0],[.2,0])
    assert np.allclose(result[1],[0,1.1])


def test_se2_bridge_preserves_rotation_and_does_not_fallback(monkeypatch,tmp_path):
    import json
    from types import SimpleNamespace
    from src.automatic_motion_bridge import plan_bridge
    from src import geometry_outbound_gcs
    start=np.array([-1.,0.,0.]);goal=np.array([.6,0.,.2]);arm=np.zeros(7)
    scene=SimpleNamespace(config={'green_room':{'navigation_regions':['room'],'se2_navigation':{'yaw_step':.2}}})
    def solve(s,a,b,**kwargs):
        assert s is scene and np.array_equal(a,start) and np.array_equal(b,goal)
        assert np.array_equal(kwargs['arm_q'],arm)
        return SimpleNamespace(configurations=np.array([a,b])), {'solver':'official API substitute for unit test'}
    monkeypatch.setattr(geometry_outbound_gcs,'plan_geometry_outbound_se2',solve)
    path=tmp_path/'bridge.json';plan_bridge(scene,start,goal,arm,'se2_gcs',path)
    data=json.loads(path.read_text());assert data['accepted']
    assert data['path'][-1][2]==.2
    def fail(*args,**kwargs):raise RuntimeError('independent numerical audit rejected')
    monkeypatch.setattr(geometry_outbound_gcs,'plan_geometry_outbound_se2',fail)
    import pytest
    with pytest.raises(RuntimeError,match='audit rejected'):plan_bridge(scene,start,goal,arm,'se2_gcs',path)
    assert not json.loads(path.read_text())['accepted']
