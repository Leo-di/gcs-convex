from pathlib import Path
from src.scene_spec import load_scene_config
import json

import numpy as np
import pytest

from src.animate_delivery_demo import plan_delivery_candidate
from src.build_scene import build_scene
from src.candidate_planner import CandidateRejected, CandidatesExhausted, select_candidate


def test_full_validation_failure_retries_and_restores_scene(tmp_path):
    scene = build_scene(visualize=False)
    scene.config["pick_task"]["candidate_source"] = "manual"
    scene.config["pick_task"]["candidates"] = [
        {"name": "bad_trace", "base_q": [1.1, -0.1, 0]},
        {"name": "good_trace", "base_q": [1.2, -0.1, 0]},
        {"name": "unused", "base_q": [1.3, -0.1, 0]},
    ]
    original = scene.plant.GetPositions(scene.plant_context).copy()
    original_time = scene.root_context.get_time()
    calls = []

    def evaluate(scene, q, attempt):
        # Exercise the retry contract at the final validation stage, not just IK.
        np.testing.assert_allclose(scene.plant.GetPositions(scene.plant_context), original)
        assert scene.root_context.get_time() == original_time
        calls.append(attempt["name"])
        changed = original.copy()
        changed[0] += 0.1
        scene.plant.SetPositions(scene.plant_context, changed)
        scene.root_context.SetTime(17)
        attempt["stage"] = "sampled_validation"
        if attempt["name"] == "bad_trace":
            raise CandidateRejected("collision", {"first_frame": 25})
        return "validated_result"

    path = tmp_path / "attempts.json"
    result, report = select_candidate(scene, evaluate, report_path=path)
    assert result == "validated_result"
    assert calls == ["bad_trace", "good_trace"]
    assert report["selected_candidate"] == "good_trace"
    assert report["attempts"][0]["details"] == {"first_frame": 25}
    assert json.loads(path.read_text()) == report
    np.testing.assert_allclose(scene.plant.GetPositions(scene.plant_context), original)
    assert scene.root_context.get_time() == original_time


def test_real_solver_failures_exhaust_candidates_without_claiming_infeasibility(tmp_path):
    scene = build_scene(visualize=False, config=load_scene_config(Path(__file__).parent / "fixtures/legacy_grasp.yaml"))
    scene.config["pick_task"]["candidate_source"] = "manual"
    scene.config["pick_task"]["candidates"] = scene.config["pick_task"]["candidates"][:2]
    scene.config["pick_task"]["grasp_offset"] = [0, 0, 3.0]
    path = tmp_path / "attempts.json"
    with pytest.raises(CandidatesExhausted, match="not a proof") as caught:
        select_candidate(scene, plan_delivery_candidate, report_path=path)
    report = caught.value.report
    assert report["status"] == "exhausted"
    assert report["selected_candidate"] is None
    assert len(report["attempts"]) == 2
    assert all(item["stage"] == "pickup_ik" for item in report["attempts"])
    assert all(item["error_type"] == "IKSolveError" for item in report["attempts"])
    assert all(item["details"]["stage"] == "grasp" for item in report["attempts"])
    assert json.loads(path.read_text()) == report


def test_proximity_failure_is_logged_before_optimization(tmp_path):
    scene = build_scene(visualize=False)
    scene.config["pick_task"]["candidate_source"] = "manual"
    scene.config["pick_task"]["candidates"] = [{"name": "far", "base_q": [0.25, -0.45, 0]}]
    with pytest.raises(CandidatesExhausted) as caught:
        select_candidate(scene, plan_delivery_candidate, report_path=tmp_path / "attempts.json")
    attempt = caught.value.report["attempts"][0]
    assert attempt["stage"] == "pickup_proximity"
    assert attempt["details"]["near_pickup_residual"] < 0


def test_unexpected_errors_are_not_silently_retried(tmp_path):
    scene = build_scene(visualize=False)
    scene.config["pick_task"]["candidate_source"] = "manual"
    path = tmp_path / "attempts.json"

    def broken(scene, q, attempt):
        raise TypeError("implementation bug")

    with pytest.raises(TypeError, match="implementation bug"):
        select_candidate(scene, broken, report_path=path)
    report = json.loads(path.read_text())
    assert report["status"] == "error"
    assert len(report["attempts"]) == 1


@pytest.mark.parametrize("candidates", [
    [],
    [{"name": "bad", "base_q": [float("nan"), 0, 0]}],
    [{"name": "bad", "base_q": [0, 0]}],
    [{"name": "same", "base_q": [0, 0, 0]}] * 2,
])
def test_malformed_candidates_fail_before_planning(candidates):
    scene = build_scene(visualize=False)
    scene.config["pick_task"]["candidate_source"] = "manual"
    scene.config["pick_task"]["candidates"] = candidates
    with pytest.raises(ValueError):
        select_candidate(scene, lambda *args: pytest.fail("Must not plan malformed candidates"))
