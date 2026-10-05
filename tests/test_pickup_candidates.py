import numpy as np
import pytest

from src.candidate_planner import CandidatesExhausted, select_candidate
from src.build_scene import build_scene
from src.pickup_candidates import generate_pickup_candidates
from src.scene_spec import load_scene_config
from src.task_semantics import near_pickup_residual


def test_generation_is_deterministic_bounded_and_ignores_manual_points():
    config = load_scene_config()
    candidates, report = generate_pickup_candidates(config)
    config["pick_task"]["candidates"] = [{"name": "bad", "base_q": [-99, -99, -99]}]
    config["reference_path"][-1] = [-99, -99, -99]
    again, _ = generate_pickup_candidates(config)
    assert candidates == again
    assert len(candidates) == config["pick_task"]["candidate_generation"]["max_candidates"]
    assert report["truncated"]
    assert len({tuple(c["base_q"]) for c in candidates}) == len(candidates)
    region = next(r for r in config["gcs_base_planning"]["regions"] if r["name"] == "room2_pickup_lane")
    for item in candidates:
        q = np.array(item["base_q"])
        assert np.all(q[:2] >= region["lower"]) and np.all(q[:2] <= region["upper"])
        assert region["yaw_bounds"][0] <= q[2] <= region["yaw_bounds"][1]
        assert near_pickup_residual(q, config) >= 0


def test_ranking_changes_when_cup_moves():
    config = load_scene_config()
    before, _ = generate_pickup_candidates(config)
    config["objects"]["cup"]["center"][0] += 0.05
    after, _ = generate_pickup_candidates(config)
    assert before[0]["base_q"] != after[0]["base_q"]


def test_yaw_filter_and_overlapping_regions_do_not_duplicate_candidates():
    from copy import deepcopy

    config = load_scene_config()
    spec = config["pick_task"]["candidate_generation"]
    spec["yaw_samples_rad"] = [0.0, 0.0, 2.0]
    original, _ = generate_pickup_candidates(config)
    duplicate = deepcopy(next(r for r in config["gcs_base_planning"]["regions"]
                              if r["name"] == "room2_pickup_lane"))
    duplicate["name"] = "overlap"
    config["gcs_base_planning"]["regions"].append(duplicate)
    config["semantic_product_planning"]["transitions"][0]["candidate_regions"].append("overlap")
    combined, _ = generate_pickup_candidates(config)
    assert combined == original
    assert all(item["base_q"][2] == 0 for item in combined)


def test_empty_generated_domain_is_reported_as_exhausted(tmp_path):
    config = load_scene_config()
    config["pick_task"]["candidate_generation"]["boundary_margin_m"] = 10
    scene = build_scene(visualize=False, config=config)
    with pytest.raises(CandidatesExhausted) as error:
        select_candidate(scene, lambda *args: pytest.fail("No candidates should be evaluated"),
                         report_path=tmp_path / "attempts.json")
    assert error.value.report["attempts"] == []
    assert error.value.report["generation"]["returned_count"] == 0


@pytest.mark.parametrize("key,value", [
    ("xy_step_m", 0), ("xy_step_m", float("nan")),
    ("max_candidates", 0), ("max_candidates", 1.5),
    ("yaw_samples_rad", []), ("boundary_margin_m", -1),
])
def test_invalid_generator_settings_are_not_solver_failures(key, value):
    config = load_scene_config()
    config["pick_task"]["candidate_generation"][key] = value
    with pytest.raises(ValueError):
        generate_pickup_candidates(config)
