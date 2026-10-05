import copy

from src.semantic_gcs_experiment import checkpoint_monitor, contact_checkpoint_monitor, solve_graph


def model():
    disk={'center':[0.,0.], 'radius':10.}
    return {'empty_cover':[disk],
            'carry':{'covers':[[disk]],'postures':[[0.]*8], 'transitions':[],
                     'start':[1.2,-.1],'goal':[-2.35,1.3]},
            'guards':{'pick':{'alignment':.001},'place':{'support':.001}}}


def test_same_geometry_changes_acceptance_and_required_action_order():
    shared=model();unchanged=copy.deepcopy(shared)
    geometric,_=solve_graph(shared,[-3.,0.],'geometry_only')
    delivery,_=solve_graph(shared,[-3.,0.],'delivery')
    inspection,_=solve_graph(shared,[-3.,0.],'inspect_then_delivery')
    assert geometric['actions']==[]
    assert delivery['actions']==['pick','place']
    assert inspection['actions']==['inspect','pick','place']
    assert shared==unchanged


def test_invalid_geometric_pick_guard_removes_accepting_path():
    shared=model();shared['guards']['pick']['alignment']=-.01
    result,path=solve_graph(shared,[-3.,0.],'delivery')
    assert not result['pick_edge_present']
    assert result['status']=='unreachable_in_compiled_graph'
    assert path is None


def test_checkpoint_labels_and_late_visit_cannot_spoof_monitor():
    rows=[{'base_q':[0,0,0],'attached':False,'phase':'inspect'},
          {'base_q':[1,0,0],'attached':True},
          {'base_q':[-2.5,-1,0],'attached':True}]
    assert not checkpoint_monitor(rows)['satisfied']
    rows.insert(1,{'base_q':[-2.5,-1,0],'attached':False})
    assert checkpoint_monitor(rows)['satisfied']


def test_physics_monitor_requires_real_bilateral_contact_after_visit():
    def row(t,xy,left,right):
        return {'time_s':t,'base_q':[*xy,0],
                'finger_contact_force_N':{'left_finger':left,'right_finger':right}}
    result={'acceptance':{'minimum_finger_or_bin_contact_force_N':.05},
            'trace':[row(1,[-2.5,-1],.1,0),row(2,[1,0],.1,.1)]}
    assert contact_checkpoint_monitor(result)['satisfied']
    result['trace'].reverse()
    assert not contact_checkpoint_monitor(result)['satisfied']
    result['trace']=[row(1,[-2.5,-1],0,0)]
    assert not contact_checkpoint_monitor(result)['satisfied']
