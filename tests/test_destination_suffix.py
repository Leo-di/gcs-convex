import copy
from pathlib import Path
import numpy as np
import pytest
from src import stateful_transport_grounding as core
from src import destination_suffix as suffix

def test_translated_suffix_preserves_full_configuration_and_source():
    frames=[dict(mode=m,phase='a.'+m,base_q=[1.,2.,.3],arm_q=[.1]*7,gripper_q=[-.02,.02]) for m in ['hold','lower','release','retreat']]
    original=copy.deepcopy(frames);out=core.translated_destination_suffix(frames,[.7,-.2,0])
    assert [f['mode'] for f in out]==['lower','release','retreat']
    for f in out:
        assert np.allclose(f['base_q'],[1.7,1.8,.3]) and f['arm_q']==frames[0]['arm_q']
    out[0]['arm_q'][0]=99
    assert frames==original

def test_ground_precheck_rejects_before_any_transport_search(tmp_path,monkeypatch):
    from types import SimpleNamespace
    cfg={'active_object_id':'cup'};world={'robot':{},'supports':{'cup':'table'},'poses':{'cup':np.eye(4).tolist()}}
    scene=SimpleNamespace(plant=SimpleNamespace(SetFreeBodyPose=lambda *a:None),plant_context=None,cup_body=None)
    monkeypatch.setattr(core,'world_config',lambda *a:cfg)
    monkeypatch.setattr(core,'build_scene',lambda **k:scene)
    monkeypatch.setattr(core,'destination_occupancy',lambda *a:{'satisfied':True})
    monkeypatch.setattr(core,'plan_action',lambda *a:([{'mode':'hold'}],{}))
    def fail(*a):raise AssertionError('transport must not run')
    monkeypatch.setattr(core,'complete_transport_candidate',fail)
    def reject(*a):raise core.TransportRejected('fixed_destination_suffix_passage_rejected',{})
    with pytest.raises(core.TransportRejected,match='fixed_destination_suffix'):
        core.ground_action(cfg,world,{'object_id':'cup','source_support':'table','id':'a','pickup_base':[0,0,0]},[1,1,0],{'transport_posture_search':{}},tmp_path/'branch',destination_precheck=reject)

def test_passing_precheck_never_accepts_action_without_full_planning(tmp_path,monkeypatch):
    from types import SimpleNamespace
    cfg={};world={'robot':{},'supports':{'cup':'table'},'poses':{'cup':np.eye(4).tolist()}}
    scene=SimpleNamespace(plant=SimpleNamespace(SetFreeBodyPose=lambda *a:None),plant_context=None,cup_body=None)
    monkeypatch.setattr(core,'world_config',lambda *a:cfg);monkeypatch.setattr(core,'build_scene',lambda **k:scene)
    monkeypatch.setattr(core,'destination_occupancy',lambda *a:{'satisfied':True});monkeypatch.setattr(core,'plan_action',lambda *a:([],{}))
    calls=[]
    def complete(*a):calls.append('full');raise core.TransportRejected('full_stage_failed',{})
    monkeypatch.setattr(core,'complete_transport_candidate',complete)
    with pytest.raises(core.TransportRejected,match='full_stage_failed'):
        core.ground_action(cfg,world,{'object_id':'cup','source_support':'table','id':'a','pickup_base':[0,0,0]},[1,1,0],{},tmp_path/'branch',destination_precheck=lambda *a:calls.append('precheck'))
    assert calls==['precheck','full']
