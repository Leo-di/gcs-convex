from src.coupled_goal_experiment import dock_ok,options,materialize

def test_docking_filter_uses_geometry_not_candidate_id():
    shared={'goals':{'a':{'one':{'base':[-2.5,1.2,0]},'two':{'base':[-2.2,1.2,0]}},'b':{'one':{'base':[-2.5,1.2,0]}}}}
    cell={'allowed':['a'],'docking':{'lower':[-2.6,1.],'upper':[-2.35,1.5]}}
    assert list(options(shared,cell)['a'])==['one'] and 'b' not in options(shared,cell)
    assert not dock_ok([-2.2,1.2,0],cell['docking'])

def test_selected_primitive_retains_its_grasp_cover_and_does_not_move_target():
    goal={'rows':[{'phase':'retreat_arm'}],'guards':{},'base':[-2.2,1.2,1.57],'height':.14}
    shared={'goals':{'a':{'one':goal}},'candidates':{'a':{'cover':{'regions':['original']}}}}
    config={'delivery_task':{'transfer_path':[[0,0,0]],'preplace_height':.22,'place_cup_center':[1,2,3]}}
    cs,c=materialize(shared,'a','one',config)
    assert cs['a']['cover'] is shared['candidates']['a']['cover']
    assert c['delivery_task']['place_cup_center']==[1,2,3]
    assert config['delivery_task']['transfer_path']==[[0,0,0]]
    assert c['delivery_task']['transfer_path'][-1]==goal['base']
