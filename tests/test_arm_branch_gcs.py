import json
from pathlib import Path

import pytest
import numpy as np

from src.arm_branch_gcs import make_families, lateral_bounds, solve_branches, audit
from src.arm_boundary_semantics import ArmBoundaryOracle
from src.build_scene import build_scene
from src.execute_arm_branch_gcs import audit_execution


@pytest.fixture(scope='module')
def experiment():
    fixture=json.loads((Path(__file__).parent/'fixtures/arm_boundary.json').read_text())
    scene=build_scene(config=fixture['config'],visualize=False);families=make_families(scene,fixture)
    covers={}
    for branch,family in families.items():
        covers[branch]=[dict(index=i,alpha=[i*.05,(i+1)*.05],accepted=True,
                            support_upper_m=family.support_upper_bound(i*.05,(i+1)*.05),
                            lateral=lateral_bounds(family,i*.05,(i+1)*.05)) for i in range(9)]
    return families,covers


@pytest.mark.parametrize('side,expected',[('positive_y','negative_y'),('negative_y','positive_y')])
def test_policy_switches_selected_branch(experiment,side,expected):
    families,covers=experiment
    case=dict(start_x=-1.3,gate_x=-.6,lateral_limits={side:.2})
    result=solve_branches(families,covers,case,list(families))
    assert result['status']=='solved' and result['selected_branch']==expected
    assert audit(families,result,case)[0]['accepted']
    wrong=solve_branches(families,covers,case,[side])
    assert wrong['status']=='unreachable_in_cover'


def test_both_branches_can_be_rejected(experiment):
    families,covers=experiment
    case=dict(start_x=-1.3,gate_x=-.6,lateral_limits={'positive_y':.2,'negative_y':.2})
    result=solve_branches(families,covers,case,list(families))
    assert result['status']=='unreachable_in_cover'


def test_lateral_bounds_cover_interior_postures(experiment):
    families,covers=experiment
    for branch,family in families.items():
        for cell in covers[branch]:
            alpha=.37*cell['alpha'][0]+.63*cell['alpha'][1]
            for side,normal in [('positive_y',[0,1,0]),('negative_y',[0,-1,0])]:
                support=-ArmBoundaryOracle(family.scene,normal).query(family.state(-.8,alpha))['residual_m']
                bounds=cell['lateral'][side]
                assert bounds['lower']<=support<=bounds['upper']


def test_execution_audit_uses_actual_arm_for_lateral_rule(experiment):
    families,_=experiment;family=families['negative_y'];scene=family.scene
    states=[family.state(-1.3,0),family.state(-.6,.3),family.state(.6,0)]
    raw=dict(trace=[dict(base_q=q[:3].tolist(),arm_q=q[3:].tolist(),
                        gripper_q=scene.gripper_open_positions.tolist(),
                        cup_pose=scene.cup_initial_pose.GetAsMatrix4().tolist()) for q in states],
             unexpected_contact_count=0,actuator_effort_limits_satisfied=True,
             base_workspace_bounds_satisfied=True,maximum_arm_tracking_error_rad=0.,
             maximum_base_position_error_m=0.,duration_s=2.)
    case=dict(start_x=-1.3,gate_x=-.6,lateral_limits={'positive_y':.2})
    assert audit_execution(raw,scene,case,states[-1])['accepted']
    case['lateral_limits']={'negative_y':.2}
    wrong=audit_execution(raw,scene,case,states[-1])
    assert wrong['minimum_semantic_residuals_m']['room_x']>0
    assert wrong['minimum_semantic_residuals_m']['negative_y']<0
    assert not wrong['accepted']
