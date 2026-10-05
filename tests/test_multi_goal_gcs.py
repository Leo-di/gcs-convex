import copy
import numpy as np
from pydrake.geometry.optimization import HPolyhedron
from src.semantic_candidate_gcs import build_and_solve

def row(phase,x):return {'phase':phase,'base_q':[x,0,0],'arm_q':[0.]*7}

def fixture():
    box=HPolyhedron.MakeBox(np.full(10,-4.),np.full(10,4.))
    c={'rows':[row('navigate_to_pickup',0),row('arm_to_pregrasp',.5),row('lift_cup',.5),row('lower_into_bin',2),row('retreat_arm',2)],
       'guards':{'valid':{'margin':1.}},'cover':{'regions':[{'A':box.A().tolist(),'b':box.b().tolist()}]}}
    near=copy.deepcopy(c);near['rows'][-2:]=[row('lower_into_bin',1),row('retreat_arm',1)]
    return c,near,[{'center':[0.,0.],'radius':1.5}],{'delivery_task':{'transfer_path':[[2.,0.,0.]]}}

def test_graph_jointly_selects_goal_and_preserves_grasp_identity():
    c,near,disks,config=fixture()
    r,path=build_and_solve(disks,{'grasp':c},['grasp'],config,goal_options={'grasp':{'far':c,'near':near}})
    assert r['status']=='solved' and r['selected_goal']=='near'
    assert all(v['candidate']=='grasp' for v in path if 'candidate' in v)
    assert r['vertices']==7  # Carry cover appears once, not once per goal.

def test_no_allowed_goal_cannot_reach_accept():
    c,near,disks,config=fixture()
    r,path=build_and_solve(disks,{'grasp':c},['grasp'],config,goal_options={'grasp':{}})
    assert path is None and r['status']=='unreachable_in_compiled_graph'

def test_invalid_goal_guard_cannot_be_bought_with_lower_cost():
    c,near,disks,config=fixture();near['guards']['valid']['margin']=-1.
    r,path=build_and_solve(disks,{'grasp':c},['grasp'],config,goal_options={'grasp':{'far':c,'near':near}})
    assert r['status']=='solved' and r['selected_goal']=='far'
