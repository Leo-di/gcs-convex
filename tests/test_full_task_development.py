import json
from types import SimpleNamespace
import pytest
from src.benchmark_full_task_development import contact_worker


@pytest.mark.parametrize('contact_ok,checkpoint_ok,target_error,expected',[
    (True,True,.01,True), (False,True,.01,False),
    (True,False,.01,False), (True,True,.03,False),
])
def test_complete_success_requires_physical_contact_checkpoint_and_target(
        tmp_path,monkeypatch,contact_ok,checkpoint_ok,target_error,expected):
    out=tmp_path/'cell'/'product_gcs';out.mkdir(parents=True)
    ref=out/'reference';ref.mkdir()
    config={'delivery_task':{'place_cup_center':[0,0,1]},
            'inspection_sites':[{'name':'site','xy':[0,0]}]}
    (ref/'scene_config_snapshot.json').write_text(json.dumps(config))
    (ref/'delivery_trace.json').write_text('[]')
    (out/'record.json').write_text(json.dumps({'reference':str(ref),'planning_success':True}))
    report={'satisfied':contact_ok,'duration_s':2.,'trace':[{'time_s':2.,
        'cup_pose':[[1,0,0,target_error],[0,1,0,0],[0,0,1,1],[0,0,0,1]]}]}
    monkeypatch.setattr('src.execute_delivery_dynamics.execute',lambda *a,**kw:(report,None))
    monkeypatch.setattr('src.semantic_gcs_experiment.contact_checkpoint_monitor',
                        lambda *a,**kw:{'satisfied':checkpoint_ok})
    contact_worker(SimpleNamespace(output=tmp_path,cell='cell',method='product_gcs'))
    result=json.loads((out/'record.json').read_text())
    assert result['end_to_end_success'] is expected
    assert result['planning_success']  # Planning success cannot hide execution failure.
