from types import SimpleNamespace
import numpy as np
import pytest
from src.joint_grounding_checks import check_pick_endpoints


def test_all_three_full_configuration_endpoints_are_checked(monkeypatch):
    pick=SimpleNamespace(pregrasp_q=np.ones(7),grasp_q=np.ones(7)*2,lift_q=np.ones(7)*3)
    monkeypatch.setattr('src.grasp_planner.plan_pick',lambda scene,base:pick)
    seen=[]
    def query(q):seen.append(q.copy());return {'satisfied':True}
    result,goals=check_pick_endpoints(None,np.array([.5,0,0]),SimpleNamespace(query=query))
    assert result is pick and set(goals)=={'pregrasp','grasp','lift'}
    assert len(seen)==3 and all(q.shape==(10,) for q in seen)
    assert all(np.allclose(q[:3],[.5,0,0]) for q in seen)


def test_lift_semantic_failure_cannot_pass_on_grasp_alone(monkeypatch):
    pick=SimpleNamespace(pregrasp_q=np.ones(7),grasp_q=np.ones(7),lift_q=np.ones(7)*2)
    monkeypatch.setattr('src.grasp_planner.plan_pick',lambda scene,base:pick)
    oracle=SimpleNamespace(query=lambda q:{'satisfied':q[-1]<2})
    with pytest.raises(RuntimeError,match='endpoint violates'):
        check_pick_endpoints(None,np.zeros(3),oracle)


def test_geometry_ablation_records_room_failure_without_rejecting_ik(monkeypatch):
    pick=SimpleNamespace(pregrasp_q=np.ones(7),grasp_q=np.ones(7),lift_q=np.ones(7))
    monkeypatch.setattr('src.grasp_planner.plan_pick',lambda scene,base:pick)
    oracle=SimpleNamespace(query=lambda q:{'satisfied':False,'residual_m':-.2})
    result,goals=check_pick_endpoints(None,np.zeros(3),oracle,enforce_semantic=False)
    assert result is pick
    assert len(goals)==3 and all(not g['satisfied'] for g in goals.values())


def test_geometry_ablation_does_not_swallow_ik_failure(monkeypatch):
    def fail(scene,base):
        raise RuntimeError('collision-aware IK failed')
    monkeypatch.setattr('src.grasp_planner.plan_pick',fail)
    with pytest.raises(RuntimeError,match='collision-aware IK failed'):
        check_pick_endpoints(None,np.zeros(3),None,enforce_semantic=False)
