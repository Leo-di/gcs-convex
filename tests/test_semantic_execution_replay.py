from src.audit_semantic_execution import phase_summary


def test_navigation_outside_is_not_an_operation_violation():
    rows=[{'time_s':0.,'phase':'navigate_to_pickup','room_residual_m':-3.}, {'time_s':1.,'phase':'navigate_to_pickup','room_residual_m':.02}, {'time_s':2.,'phase':'approach_cup','room_residual_m':.02}, {'time_s':3.,'phase':'approach_cup','room_residual_m':-.001}]
    nav,op=phase_summary(rows)
    assert nav['violations']==1 and not nav['whole_robot_required']
    assert op['violations']==1 and op['whole_robot_required']
    assert op['minimum_room_residual_m']==-.001
    assert op['start_s']==2. and op['end_s']==3.
