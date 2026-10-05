import json
from pathlib import Path

import numpy as np
import pytest

from src.arm_boundary_gcs import PostureFamily, solve, audit_path
from src.build_scene import build_scene
from src.execute_arm_boundary_gcs import reference_from_path


@pytest.fixture(scope='module')
def family():
    fixture=json.loads((Path(__file__).parent/'fixtures/arm_boundary.json').read_text())
    return PostureFamily(build_scene(config=fixture['config'],visualize=False),
                         fixture['poses']['safe']['state'][3:])


def test_affine_regions_preserve_arm_coupling(family):
    region=family.polytope(-1.4,-.6,.25,.35)
    q=family.state(-.8,.3)
    assert region.PointInSet(q)
    q[7]+=.01
    assert not region.PointInSet(q)
    assert not region.PointInSet(family.state(-.5,.3))


def test_support_envelope_and_infeasible_gate(family):
    lo,hi=family.support_bounds(.25,.35)
    for alpha in np.random.default_rng(829).uniform(.25,.35,100):
        support=-family.oracle.query(family.state(0,alpha))['residual_m']
        assert lo<=support<=hi
    # Even the lower bound on support exceeds the 0.6 m gate offset:
    # no alpha in [0,.15] can satisfy arm_x<=0 at x=-.6.
    lower,_=family.support_bounds(0,.15)
    assert lower>.6


def test_semantics_change_gcs_and_negative_rejected(family):
    cells=[dict(index=i,alpha=[i*.05,(i+1)*.05],accepted=True,
                support_upper_m=family.support_upper_bound(i*.05,(i+1)*.05)) for i in range(9)]
    common=dict(start_x=-1.3,gate_x=-.6,max_alpha=.45)
    baseline=solve(family,cells,aware=False,**common)
    semantic=solve(family,cells,aware=True,**common)
    assert baseline['status']==semantic['status']=='solved'
    base_audit,_=audit_path(family,baseline,**common)
    arm_audit,_=audit_path(family,semantic,**common)
    assert base_audit['minimum_collision_distance_m']>0
    assert not base_audit['accepted']
    assert arm_audit['accepted']
    assert arm_audit['max_alpha']>.2
    negative=solve(family,cells,start_x=-1.3,gate_x=-.6,max_alpha=.15,aware=True)
    assert negative['status']=='unreachable_in_cover'


def test_execution_retiming_preserves_vertices_and_speed_limits(family):
    a,b,c=family.state(-1.3,0),family.state(-.6,.3),family.state(.6,0)
    path=[dict(start=a,end=b,mode='room'),dict(start=b,end=c,mode='transit')]
    rows=reference_from_path(path,[-.055,.055])
    q=np.array([r['base_q']+r['arm_q'] for r in rows])
    assert any(np.max(np.abs(row-b))<1e-12 for row in q)
    np.testing.assert_allclose(q[0],a)
    np.testing.assert_allclose(q[-1],c)
    assert np.all(np.abs(np.diff(q,axis=0))/.15<=np.array([.2,.2,.3]+[.3]*7)+1e-12)
