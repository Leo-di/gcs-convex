import copy
import numpy as np
import pytest
from pydrake.math import RigidTransform
from src.scene_spec import load_scene_config
from src.stateful_transport_grounding import make_config, semantic_plan
from src.build_scene import build_scene
from src.room_entry import ENTRY_PHASE, ENTRY_MODE, EntryExecutionGate, compress_collinear, join_entry, plan_entry
from src.semantic_plan import SemanticPlanEvaluator, configuration


def frames():
    cfg=make_config(load_scene_config('configs/room_entry_transport.yaml'))
    scene=build_scene(config=cfg,visualize=False)
    f={'base_q':[.45,0.,0.],'arm_q':cfg['pick_task']['navigation_arm_q'],
       'gripper_q':scene.gripper_open_positions.tolist(),'phase':ENTRY_PHASE,'mode':ENTRY_MODE,
       'object_id':'cup','world_objects':{'cup':scene.cup_initial_pose.GetAsMatrix4().tolist()},
       'world_supports':{'cup':'table'}}
    first=copy.deepcopy(f);first['base_q']=cfg['robot']['start']
    last=copy.deepcopy(f);last.update(phase='move_first_cup.arrive',mode='arrive')
    return cfg,[first,f,last]


def test_semantics_switch_only_at_the_entry_exit():
    cfg,path=frames();scene=build_scene(config=cfg,visualize=False)
    plan=semantic_plan(cfg,path);evaluator=SemanticPlanEvaluator(scene,plan)
    assert 'inside' not in plan['phases'][0]['invariants']
    assert plan['phases'][0]['exit']==['inside']
    assert not evaluator.evaluate('inside',configuration(path[0]))['satisfied']
    assert evaluator.audit(path)['satisfied']
    bad=copy.deepcopy(path);bad[1]['base_q'][0]=.2;bad[2]['base_q'][0]=.2
    assert not SemanticPlanEvaluator(scene,semantic_plan(cfg,bad)).audit(bad)['satisfied']


@pytest.mark.parametrize('field',['base_q','arm_q','gripper_q','world_objects','world_supports'])
def test_handoff_rejects_every_configuration_or_world_discontinuity(field):
    _,path=frames();operation=copy.deepcopy(path[-1])
    if field=='world_objects':operation[field]['cup'][0][3]+=.01
    elif field=='world_supports':operation[field]['cup']='dropoff_table'
    else:operation[field][0]+=.01
    with pytest.raises(ValueError):join_entry(path[:2],[operation])


def test_measured_gate_requires_entry_whole_body_and_matching_configuration():
    cfg,path=frames();gate=EntryExecutionGate(cfg,path)
    with pytest.raises(RuntimeError,match='handoff rejected'):
        gate.observe(cfg,dict(path[-1],time_s=1.),configuration(path[-1]),.03)
    assert gate.observe(cfg,dict(path[0],time_s=.01),configuration(path[0]),-1.6)
    with pytest.raises(RuntimeError,match='handoff rejected'):
        gate.observe(cfg,dict(path[-1],time_s=2.),configuration(path[-1]),-.001)
    q=configuration(path[-1]);q[-1]+=.005
    with pytest.raises(RuntimeError,match='handoff rejected'):
        gate.observe(cfg,dict(path[-1],time_s=2.),q,.03)
    assert not gate.observe(cfg,dict(path[-1],time_s=2.),configuration(path[-1]),.03)
    assert gate.report()['satisfied'] and gate.report()['handoff']['time_s']==2.
    with pytest.raises(RuntimeError,match='cannot recur'):
        gate.observe(cfg,dict(path[0],time_s=3.),configuration(path[0]),-1.)


def test_navigation_monitor_rejects_arm_or_gripper_motion():
    cfg,path=frames()
    for index,delta in [(3,.13),(10,.005)]:
        gate=EntryExecutionGate(cfg,path);q=configuration(path[0]);q[index]+=delta
        with pytest.raises(RuntimeError,match='tucked'):
            gate.observe(cfg,dict(path[0],time_s=.01),q,-1.)


def test_gate_cannot_be_bypassed_by_missing_or_repeated_prefix():
    cfg,path=frames()
    for invalid in (path[2:],path[:2],path+path[:1]):
        with pytest.raises(ValueError):EntryExecutionGate(cfg,invalid)
    bad=copy.deepcopy(path);bad[0]['mode']='arrive'
    with pytest.raises(ValueError):EntryExecutionGate(cfg,bad)


def test_collinear_compression_preserves_corners_and_backtracking():
    line=np.c_[np.linspace(0,1,101),np.zeros((101,2))]
    assert len(compress_collinear(line))==2
    path=np.array([[0,0,0],[1,0,0],[1,1,0],[1,0,0],[2,0,0]])
    assert np.array_equal(compress_collinear(path),path)


def test_illegal_entry_exit_rejected_before_navigation_planner(monkeypatch,tmp_path):
    import src.automatic_motion_bridge as bridge
    cfg,path=frames();cfg['entry_navigation']['handoff_base']=[.2,0,0];path[-1]['base_q']=[.2,0,0]
    def forbidden(*a,**k):raise AssertionError('Navigation must not run for illegal entry endpoint')
    monkeypatch.setattr(bridge,'plan_bridge',forbidden)
    with pytest.raises(RuntimeError,match='not inside Room 2'):
        plan_entry(cfg,[path[-1]],tmp_path/'entry')


@pytest.mark.parametrize('entry',[{'handoff_base':[.45,0,0],'method':'unknown'},
    {'handoff_base':[.2,0,0],'method':'rotation_safe_gcs'},
    {'handoff_base':[float('nan'),0,0],'method':'rotation_safe_gcs'},{}])
def test_entry_schema_fails_closed(entry):
    s=load_scene_config('configs/room_entry_transport.yaml');s['entry_navigation']=entry
    with pytest.raises(ValueError):make_config(s)
