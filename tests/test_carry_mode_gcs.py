import json
from pathlib import Path

import numpy as np
import pytest

from src.carry_mode_gcs import CarryOracle
from src.scene_spec import load_scene_config


@pytest.fixture(scope="module")
def fixture():
    anchor=json.loads((Path(__file__).parent/"fixtures/carry_anchor.json").read_text())
    return CarryOracle(load_scene_config(),anchor),anchor


def test_switch_candidate_at_obstacle_is_rejected(fixture):
    oracle,anchor=fixture
    posture=np.array([anchor["base_q"][2],*anchor["arm_q"]])
    assert oracle.switch([1.6,.65],posture,posture) is None


def test_switch_keeps_cup_attached_in_world(fixture):
    oracle,anchor=fixture
    posture=np.array([anchor["base_q"][2],*anchor["arm_q"]])
    for angle in [0.,.1,.2]:
        changed=posture.copy();changed[0]+=angle
        gripper,cup=oracle.state(anchor["base_q"][:2],changed)
        np.testing.assert_allclose((gripper.inverse()@cup).GetAsMatrix4(),oracle.X_GC.GetAsMatrix4(),atol=1e-12)


def test_safe_identity_switch_reports_samples(fixture):
    oracle,anchor=fixture
    posture=np.array([anchor["base_q"][2],*anchor["arm_q"]])
    result=oracle.switch(anchor["base_q"][:2],posture,posture)
    assert result is not None and result["samples"]>=3
    assert result["minimum_clearance"]>=.002
