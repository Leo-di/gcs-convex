"""Planner policy tests: accepted nominal plans must not be replaced by repairs."""
import copy
import pytest
import src.unified_delivery as planner


def setup(monkeypatch, nominal=True):
    candidate={'rows':[{'phase':'lift_cup'}, {'phase':'lower_into_bin'}],
               'cover':{'regions':[{'independent_holdout_audit':{'collisions':0}}]}}
    bundle={'config':{'delivery_task':{}},'cell':{'id':'test','allowed':['a'],'placement_base':[0,0,0]},
            'candidates':{'a':candidate} if nominal else {},'disks':[], 'preparation_seconds':0}
    monkeypatch.setattr(planner,'read_library',lambda _: ({},{'a':copy.deepcopy(candidate)}))
    monkeypatch.setattr(planner,'q',lambda _: [0]*10)
    monkeypatch.setattr(planner,'attach_carry_endpoints',lambda config,regions,*a,**k:(regions,{}))
    monkeypatch.setattr(planner,'build_and_solve',lambda *a,**k:({'status':'success','selected_candidate':'a'},['path']))
    return bundle,candidate


def test_accepted_nominal_never_calls_repairs(monkeypatch,tmp_path):
    bundle,_=setup(monkeypatch)
    def forbidden(*a,**k):raise AssertionError('Do not replace an accepted nominal plan')
    monkeypatch.setattr(planner,'change_place',forbidden)
    monkeypatch.setattr(planner,'bridge',forbidden)
    monkeypatch.setattr(planner,'expand',lambda *a,**k:{'accepted':True})
    result=planner.run(bundle,tmp_path,0)
    assert result['planning_success'] and result['accepted_height']==.22
    assert not result['accepted_bridge_stage'] and len(result['stages'])==1


def test_geometric_audit_failure_is_not_accepted(monkeypatch,tmp_path):
    bundle,_=setup(monkeypatch)
    monkeypatch.setattr(planner,'expand',lambda *a,**k:{'accepted':False})
    monkeypatch.setattr(planner,'bridge',lambda config,regions,*a,**k:(regions,{}))
    def unavailable(*a,**k):raise RuntimeError('No legal primitive')
    monkeypatch.setattr(planner,'change_place',unavailable)
    result=planner.run(bundle,tmp_path,0)
    assert not result['planning_success'] and result['failure']=='bounded_repairs_exhausted'
    assert len(result['stages'])==3


def test_alternative_height_retains_fixed_placement_base(monkeypatch,tmp_path):
    bundle,candidate=setup(monkeypatch,nominal=False)
    original=copy.deepcopy(bundle)
    calls=[]
    def alternative(config,candidate,*,placement_base):
        calls.append((config['delivery_task']['preplace_height'],placement_base))
        return candidate
    monkeypatch.setattr(planner,'change_place',alternative)
    monkeypatch.setattr(planner,'expand',lambda *a,**k:{'accepted':True})
    result=planner.run(bundle,tmp_path,0)
    assert result['accepted_height']==.14 and calls==[(.14,[0,0,0])]
    assert bundle==original


def test_nominal_initial_state_source_matches_baseline(monkeypatch,tmp_path):
    bundle,candidate=setup(monkeypatch)
    bundle['candidates']={'forbidden':copy.deepcopy(candidate),**bundle['candidates']}
    def solve(disks,candidates,allowed,config,**kw):
        assert next(iter(candidates))=='forbidden'
        assert allowed==['a']  # Extra entry supplies only the exact initial state.
        return {'status':'success','selected_candidate':'a'},['path']
    monkeypatch.setattr(planner,'build_and_solve',solve)
    monkeypatch.setattr(planner,'expand',lambda *a,**k:{'accepted':True})
    assert planner.run(bundle,tmp_path,0)['planning_success']
