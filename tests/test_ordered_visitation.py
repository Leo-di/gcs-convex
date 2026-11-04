import copy
import numpy as np
import pytest
from pydrake.math import RigidTransform
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config
from src.build_scene import build_scene
from src.ordered_visitation import OrderedVisitMonitor


def task():
    s = load_scene_config('configs/room_entry_transport.yaml')
    s['ordered_visits'] = [dict(id='serve', object_id='cup', action_id='move_first_cup', mode='carry', margin_m=.01,
        regions=[dict(id='A', lower_xyz=[.3,.4,1.], upper_xyz=[.7,.8,1.3]),
                 dict(id='B', lower_xyz=[1.,.4,1.], upper_xyz=[1.4,.8,1.3])])]
    return s


@pytest.fixture
def monitor():
    return OrderedVisitMonitor(build_scene(config=make_config(task()), visualize=False))


def observe(m, x, phase='move_first_cup.carry', y=.6):
    return m.observe({'cup': RigidTransform([x,y,1.15])}, phase, float(m.samples))


def test_valid_order_persists_through_leaving_a_and_finishing_carry(monitor):
    assert observe(monitor,.5)['satisfied']
    assert observe(monitor,.8)['satisfied']
    assert not monitor.report()['complete']
    assert observe(monitor,1.2)['satisfied']
    assert observe(monitor,1.5,'move_first_cup.lower')['satisfied']
    assert monitor.report()['satisfied']
    assert [r['region'] for r in monitor.events] == ['A','B']


def test_future_overlap_fails_before_center_enters_and_failure_latches(monitor):
    assert observe(monitor,.2)['satisfied']
    assert not observe(monitor,.99)['satisfied']
    assert monitor.failures[0]['reason'] == 'future_region_entered_before_predecessor'
    assert not observe(monitor,.5)['satisfied']
    assert monitor.progress['serve'] == 0


def test_center_inside_does_not_count_full_object_visit(monitor):
    assert observe(monitor,.31)['satisfied']
    assert monitor.progress['serve'] == 0
    assert not observe(monitor,.8,'move_first_cup.lower')['satisfied']


def test_other_phase_and_action_do_not_grant_credit(monitor):
    observe(monitor,.5,'move_first_cup.lift')
    observe(monitor,.5,'other.carry')
    assert not monitor.report()['satisfied']
    assert monitor.progress['serve'] == 0
    assert not observe(monitor,1.2)['satisfied']


def test_simultaneous_overlap_is_not_order(monitor):
    monitor.specs = copy.deepcopy(monitor.specs)
    monitor.specs[0]['regions'][1].update(lower_xyz=[.4,.4,1.],upper_xyz=[.8,.8,1.3])
    assert not observe(monitor,.5)['satisfied']
    assert not monitor.events


def test_cannot_reset_progress_by_leaving_and_reentering_phase(monitor):
    observe(monitor,.5)
    assert not observe(monitor,.8,'move_first_cup.hold')['satisfied']
    assert not observe(monitor,1.2)['satisfied']


@pytest.mark.parametrize('change', [{'mode':'lift'},{'object_id':'unknown'}, {'action_id':'unknown'},
    {'margin_m':float('nan')}, {'regions':[]}, {'regions':[{'id':'A','lower_xyz':[0,0,0],'upper_xyz':[1,1,1]}]}])
def test_invalid_specs_rejected(change):
    s=task();s['ordered_visits'][0].update(change)
    with pytest.raises(ValueError): make_config(s)


def test_planner_rejects_checkpoint_at_unreachable_height_before_gcs(tmp_path):
    from src.ordered_visitation import plan_ordered_translation
    from src.payload_translation_bridge import PayloadBridgeRejected
    s=task();cfg=make_config(s);scene=build_scene(config=cfg,visualize=False)
    for region in cfg['ordered_visits'][0]['regions']:
        region.update(lower_xyz=[.3,.4,3.],upper_xyz=[.7,.8,3.5])
    with pytest.raises(PayloadBridgeRejected,match='ordered_checkpoint_not_contained'):
        plan_ordered_translation(scene,[.45,0,0],[1.4,0,0],scene.plant.GetPositions(scene.plant_context, scene.arm_instance),scene.gripper_closed_positions,
            RigidTransform(),s['carry'],tmp_path/'bridge.json',action_id='move_first_cup')


def test_failed_leg_restores_original_spatial_constraints(monkeypatch, tmp_path):
    from src.ordered_visitation import plan_ordered_translation
    from src.build_scene import set_robot_configuration
    import src.payload_translation_bridge as bridge
    s=task();cfg=make_config(s);scene=build_scene(config=cfg,visualize=False)
    base=[.45,0.,0.];arm=scene.arm_home_positions;grip=scene.gripper_closed_positions
    set_robot_configuration(scene,base,arm,grip)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@RigidTransform([.2,.6,1.15])
    original=copy.deepcopy(scene.config['object_exclusions'])
    def failing_solver(*args, **kwargs):
        active=scene.config['object_exclusions']
        assert active[:-1] == original
        assert active[-1]['lower_xyz'] == s['ordered_visits'][0]['regions'][1]['lower_xyz']
        raise RuntimeError('Injected GCS failure')
    monkeypatch.setattr(bridge,'plan_payload_translation',failing_solver)
    with pytest.raises(RuntimeError,match='ordered_candidate_search_exhausted'):
        plan_ordered_translation(scene,base,[1.5,0.,0.],arm,grip,X,s['carry'],tmp_path/'bridge.json',action_id='move_first_cup')
    assert scene.config['object_exclusions'] == original


def test_downstream_failure_backtracks_region_point_and_history(monkeypatch,tmp_path):
    import src.payload_translation_bridge as bridge
    from src.ordered_visitation import plan_ordered_translation
    from src.build_scene import set_robot_configuration
    s=task();s['carry']['ordered_checkpoint_fractions']=[.5,.2]
    scene=build_scene(config=make_config(s),visualize=False)
    base=np.array([.45,0.,0.]);arm=scene.arm_home_positions;grip=scene.gripper_closed_positions
    set_robot_configuration(scene,base,arm,grip)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@RigidTransform([.2,.6,1.15])
    # Translation maps object x=.5 to base x=.75. All exits from this
    # first A candidate fail; a different valid A point can reach B.
    def solver(scene,start,goal,*args,**kwargs):
        if abs(start[0]-.75)<1e-7 and goal[0]>1.1:
            raise RuntimeError('Test disconnected downstream graph')
        return np.linspace(start,goal,40),dict(satisfied=True,minimum_clearance_m=.01)
    monkeypatch.setattr(bridge,'plan_payload_translation',solver)
    path,report=plan_ordered_translation(scene,base,[1.8,0.,0.],arm,grip,X,s['carry'],tmp_path/'bridge.json',action_id='move_first_cup')
    assert report['satisfied'] and report['legs'][0]['candidate_index']>0
    assert report['checkpoint_search']['attempts'][0]['downstream_rejected']
    assert [e['region'] for e in report['ordered_visits']['events']]==['A','B']
    assert np.allclose(path[0],base) and np.allclose(path[-1],[1.8,0,0])


@pytest.mark.parametrize('key,value',[('ordered_checkpoint_fractions',[]),('ordered_checkpoint_fractions',[0,.5]),
    ('ordered_checkpoint_fractions',[.5,.5]),('ordered_checkpoint_fractions',[float('nan')]),
    ('ordered_max_leg_attempts',True),('ordered_max_leg_attempts',0)])
def test_bad_checkpoint_policy_rejected_before_planning(key,value):
    s=task();s['carry'][key]=value
    with pytest.raises(ValueError): make_config(s)


def test_nested_refinement_includes_boundary_and_interior_gaps():
    from src.ordered_visitation import checkpoint_refinement_levels
    levels=checkpoint_refinement_levels(dict(ordered_checkpoint_refinement_rounds=2))
    assert [len(x) for x in levels]==[3,7,15]
    assert levels[1][:3]==levels[0]
    assert np.allclose(levels[1][3:],[.1,.9,.35,.65])
    assert set(levels[1])<=set(levels[2])
    assert all(0<x<1 for x in levels[2])


@pytest.mark.parametrize('budget,success',[(36,True),(9,False)])
def test_refinement_recovers_missed_strip_with_global_budget(monkeypatch,tmp_path,budget,success):
    from src.ordered_visitation import plan_ordered_translation,checkpoint_candidates,OrderedVisitMonitor
    from src.build_scene import set_robot_configuration
    import src.payload_translation_bridge as bridge
    s=task();s['carry'].update(ordered_checkpoint_refinement_rounds=1,ordered_max_leg_attempts=budget)
    scene=build_scene(config=make_config(s),visualize=False)
    base=np.array([.45,0.,0.]);arm=scene.arm_home_positions;grip=scene.gripper_closed_positions
    set_robot_configuration(scene,base,arm,grip)
    initial=RigidTransform([.2,.6,1.15]);X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@initial
    monitor=OrderedVisitMonitor(scene);seq=monitor.specs[0]
    desired=checkpoint_candidates(monitor,seq,seq['regions'][0],initial,base,s['carry'],fractions=[.1])[0]
    calls=[]
    def solver(scene,start,goal,*args,**kwargs):
        calls.append(goal.copy())
        # A has a feasible narrow slice missed by the original sample rows.
        if goal[0]<1.1 and abs(goal[1]-desired[1])>1e-8:
            raise RuntimeError('Test narrow valid band missed')
        return np.linspace(start,goal,40),dict(satisfied=True,minimum_clearance_m=.01)
    monkeypatch.setattr(bridge,'plan_payload_translation',solver)
    if success:
        _,r=plan_ordered_translation(scene,base,[1.8,0,0],arm,grip,X,s['carry'],tmp_path/'bridge.json',action_id='move_first_cup')
        assert [x['satisfied'] for x in r['checkpoint_search']['refinement_levels']]==[False,True]
        assert r['legs'][0]['refinement_level']==1
        assert [e['region'] for e in r['ordered_visits']['events']]==['A','B']
    else:
        with pytest.raises(bridge.PayloadBridgeRejected) as exc:
            plan_ordered_translation(scene,base,[1.8,0,0],arm,grip,X,s['carry'],tmp_path/'bridge.json',action_id='move_first_cup')
        assert exc.value.details['refinement_levels'][-1]['termination']=='attempt_budget_exhausted'
    assert len(calls)<=budget


@pytest.mark.parametrize('value',[-1,3,.5,True])
def test_bad_refinement_rounds_rejected(value):
    s=task();s['carry']['ordered_checkpoint_refinement_rounds']=value
    with pytest.raises(ValueError,match='refinement rounds'):make_config(s)
