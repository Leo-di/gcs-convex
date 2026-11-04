import copy
import pytest
from src.task_order_search import available_actions,validate_order
from src import stateful_transport_grounding as core


def task():
    return dict(actions=[dict(id=k,object_id=k,pickup_base=[x,0,0],placements=[[0,0,0]])
                         for k,x in [('a',2),('b',1),('c',3)]],
                task_order=dict(mode='search',ranking='nearest_pickup'),
                max_expansions=20,time_budget_s=10,grasp_rpy=[0,0,0])


def test_precedence_and_stable_ranking_leave_every_eligible_action():
    s=task();s['task_order']['precedence']=[['a','b']]
    world=dict(robot=dict(base_q=[0,0,0]))
    assert [a['id'] for a in available_actions(s,[],world)]==['a','c']
    assert [a['id'] for a in available_actions(s,[dict(action='a')],world)]==['b','c']
    s['task_order']['precedence']+=[['b','a']]
    with pytest.raises(ValueError,match='Cyclic'):validate_order(s)


def test_order_search_changes_order_after_downstream_failure_without_world_leak(monkeypatch,tmp_path):
    s=task();s['actions']=s['actions'][:2];s['task_order']['ranking']='input'
    robot=dict(base_q=[0,0,0],arm_q=[0]*7,gripper_q=[-.05,.05])
    initial=dict(robot=robot,done=[])
    monkeypatch.setattr(core,'initial_world',lambda cfg:copy.deepcopy(initial))
    attempts=[]
    def ground(cfg,world,action,placement,spec,folder):
        attempts.append((action['id'],world['done'][:]))
        # a-first is locally valid, but makes b impossible; b-first works.
        if action['id']=='b' and world['done']==['a']:
            raise core.TransportRejected('blocked_by_previous_action',{})
        next_world=copy.deepcopy(world);next_world['done'].append(action['id'])
        return [copy.deepcopy(robot)],next_world,{}
    result=core.search({},s,tmp_path,ground=ground)
    assert result['satisfied'] and result['selected_task_order']==['b','a']
    assert attempts==[('a',[]),('b',['a']),('b',[]),('a',['b'])]
    assert initial['done']==[] and result['expansions']==4
    assert any(e['event']=='rollback_after_downstream_failure' for e in result['events'])
    s['max_expansions']=2
    bounded=core.search({},s,tmp_path,ground=ground)
    assert not bounded['satisfied'] and bounded['expansions']==2


def test_fixed_order_cannot_ignore_declared_precedence():
    s=task();s['task_order']=dict(mode='fixed',precedence=[['b','a']])
    with pytest.raises(ValueError,match='contradicts'):validate_order(s)


def test_whole_task_failure_returns_to_order_search_even_if_every_local_motion_passed(monkeypatch,tmp_path):
    s=task();s['actions']=s['actions'][:2];s['task_order']['ranking']='input'
    robot=dict(base_q=[0,0,0],arm_q=[0]*7,gripper_q=[-.05,.05])
    monkeypatch.setattr(core,'initial_world',lambda cfg:dict(robot=copy.deepcopy(robot)))
    def ground(cfg,world,action,placement,spec,folder):
        return [copy.deepcopy(robot)],copy.deepcopy(world),{}
    examined=[]
    def accept(frames,world,selected):
        order=[r['action'] for r in selected];examined.append(order)
        return dict(satisfied=order==['b','a'],reason='global joint constraints')
    result=core.search({},s,tmp_path,ground=ground,complete_accept=accept)
    assert result['satisfied'] and result['selected_task_order']==['b','a']
    assert examined==[['a','b'],['b','a']]
    assert all(b['local_satisfied'] for b in result['branches'])
    assert any(e['event']=='complete_task_candidate_rejected' for e in result['events'])
