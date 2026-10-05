import copy
import numpy as np
from test_multi_goal_gcs import fixture
from src.full_task_portal_socp import prepare
from src.global_portal_socp import solve,PortalGraph


def test_global_goal_choice_refines_and_keeps_grasp_identity():
    c,near,disks,config=fixture();cs={'grasp':c};maps=prepare(cs,disks)
    r,path=solve(disks,cs,['grasp'],config,maps,[0,0,0],{'grasp':{'far':c,'near':near}})
    assert r['selected_goal']=='near' and r['available_pairs']==2
    assert r['global_searches']==1 and r['socp_calls']==2
    assert r['objective']<=r['discrete_cost']+1e-6
    assert all(v['candidate']=='grasp' for v in path if 'candidate' in v)
    for a,b in zip(path,path[1:]):
        if b['kind']!='accept':assert np.allclose(a['x'][10:],b['x'][:10],atol=1e-6)


def test_inspection_is_a_required_directed_transition():
    c,near,disks,config=fixture();cs={'grasp':c};config['inspection_sites']=[{'name':'check','xy':[-.5,0]}]
    r,path=solve(disks,cs,['grasp'],config,prepare(cs,disks),[0,0,0],{'grasp':{'near':near}})
    kinds=[v['kind'] for v in path]
    assert kinds.index('inspect')<kinds.index('pick')<kinds.index('place')
    assert r['inspection_site']=='check' and r['socp_calls']==3
    assert r['objective']>1.99


def test_invalid_goal_and_forbidden_grasp_cannot_win():
    c,near,disks,config=fixture();near['guards']['valid']['margin']=-1
    cs={'grasp':c,'forbidden':copy.deepcopy(c)}
    r,path=solve(disks,cs,['grasp'],config,prepare(cs,disks),[0,0,0],{'grasp':{'far':c,'near':near},'forbidden':{'near':c}})
    assert r['selected_candidate']=='grasp' and r['selected_goal']=='far' and r['available_pairs']==1


def test_unreachable_checkpoint_is_not_silently_omitted():
    c,near,disks,config=fixture();config['inspection_sites']=[{'name':'outside','xy':[10,10]}];cs={'grasp':c}
    r,path=solve(disks,cs,['grasp'],config,prepare(cs,disks),[0,0,0],{'grasp':{'near':near}})
    assert path is None and r['status']=='unreachable_in_portal_graph'
