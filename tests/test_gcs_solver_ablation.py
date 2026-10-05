import json
from pathlib import Path

import pytest

from src import ablate_gcs_solver as ablation
from src.arm_branch_gcs import make_families,lateral_bounds,audit,region_specs
from src.build_scene import build_scene


@pytest.fixture(scope='module')
def experiment():
    fixture=json.loads((Path(__file__).parent/'fixtures/arm_boundary.json').read_text())
    families=make_families(build_scene(config=fixture['config'],visualize=False),fixture)
    covers={}
    for name,family in families.items():
        covers[name]=[dict(index=i,alpha=[i*.05,(i+1)*.05],accepted=True,
                          support_upper_m=family.support_upper_bound(i*.05,(i+1)*.05),
                          lateral=lateral_bounds(family,i*.05,(i+1)*.05)) for i in range(9)]
    return families,covers


@pytest.mark.parametrize('gate',[-.6,-.5])
def test_true_non_gcs_baseline_and_refinement(experiment,monkeypatch,gate):
    families,covers=experiment
    def forbidden(*args,**kwargs):raise AssertionError('Non-GCS baseline called GCS')
    monkeypatch.setattr(ablation,'solve_branches',forbidden)
    case=dict(start_x=-1.3,gate_x=gate,lateral_limits={'negative_y':.2})
    discrete=ablation.portal_search(families,covers,case,3)
    refined=ablation.refine_sequence(families,discrete)
    assert audit(families,discrete,case)[0]['accepted']
    assert audit(families,refined,case)[0]['accepted']
    assert ablation.metric_length(refined)<=ablation.metric_length(discrete)+1e-7


def test_gate_face_contains_only_semantically_compatible_cells(experiment):
    families,covers=experiment
    case=dict(start_x=-1.3,gate_x=-.6,lateral_limits={})
    specs=region_specs(families,covers,case,list(families),True)
    for spec in specs:
        if spec['mode']=='transit' and spec['rect'][0]<=case['gate_x']:
            cell=covers[spec['branch']][spec['index']]
            assert cell['support_upper_m']+.001<=-case['gate_x']
