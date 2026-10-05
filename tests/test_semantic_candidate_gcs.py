import copy

import numpy as np
from pydrake.geometry.optimization import HPolyhedron

from src.semantic_candidate_gcs import build_and_solve


def example():
    def row(x,arm,phase):
        return {'base_q':[x,0,0],'arm_q':[arm]+[0.]*6,'phase':phase}
    region=HPolyhedron.MakeBox(np.array([-1.]*10),np.array([4.]*10))
    candidates={}
    for name,x,arm in [('first_expensive',1.,1.),('second_cheaper',1.2,.1)]:
        candidates[name]={'rows':[row(0,0,'navigate_to_pickup'),row(x,0,'arm_to_pregrasp'),
                                  row(x,arm,'lift_cup'),row(3,0,'lower_into_bin'),row(3,0,'retreat_arm')],
                          'guards':{'pick':{'alignment':.1},'place':{'support':.1}},
                          'cover':{'regions':[{'A':region.A().tolist(),'b':region.b().tolist()}]}}
    return [{'center':[0,0],'radius':5}],candidates,{'delivery_task':{'transfer_path':[[3,0,0]]}}


def test_joint_cost_selects_nonfirst_candidate_and_policy_changes_selection():
    disks,candidates,config=example()
    result,path=build_and_solve(disks,candidates,list(candidates),config)
    assert result['selected_candidate']=='second_cheaper'
    actions=[v for v in path if v['kind'] in ('pick','place')]
    assert len(actions)==2 and all(v['candidate']=='second_cheaper' for v in actions)
    alternative,_=build_and_solve(disks,candidates,['first_expensive'],config)
    assert alternative['selected_candidate']=='first_expensive'
    assert result['objective']<alternative['objective']


def test_invalid_candidate_guard_and_empty_action_library_are_rejected():
    disks,candidates,config=example();candidates=copy.deepcopy(candidates)
    candidates['second_cheaper']['guards']['pick']['alignment']=-.1
    result,path=build_and_solve(disks,candidates,['second_cheaper'],config)
    assert result['status']=='unreachable_in_compiled_graph' and path is None


def test_explicit_start_and_rounding_seed_are_used_without_mutating_library():
    disks,candidates,config=example();before=copy.deepcopy(candidates)
    result,path=build_and_solve(disks,candidates,list(candidates),config,
                               start_base_q=[-.2,.1,0.],rounding_seed=7)
    assert result['status']=='solved'
    assert np.allclose(path[0]['x'][:3],[-.2,.1,0.])
    assert candidates==before
    result,path=build_and_solve(disks,candidates,[],config)
    assert result['status']=='unreachable_in_compiled_graph' and path is None
