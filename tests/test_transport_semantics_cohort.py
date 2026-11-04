import copy
import numpy as np
from scripts.run_transport_semantics_cohort import scenarios, SCOPE_ZONE
from src.stateful_transport_grounding import validate_task


def test_frozen_local_scenes_are_reproducible_and_distinct():
    a,b=scenarios(),scenarios()
    assert a==b
    assert len({tuple(r['parameters']['cup']) for r in a})==6
    for row in a:
        spec=row['task'];validate_task(spec)
        assert spec['actions'][0]['pickup_base']==spec['entry_navigation']['handoff_base']
        assert spec['object_exclusions'][1]==SCOPE_ZONE
        assert spec['start_base'][0]<0
        assert np.isfinite(spec['grasp_rpy']).all()


def test_partial_constraint_baseline_keeps_physical_scene_and_task():
    task=scenarios(1)[0]['task'];baseline=copy.deepcopy(task)
    baseline['object_exclusions']=[];baseline['object_corridors']=[]
    validate_task(baseline)
    for key in task:
        if key not in ('object_exclusions','object_corridors'):
            assert task[key]==baseline[key]
