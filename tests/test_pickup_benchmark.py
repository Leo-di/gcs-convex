from src.benchmark_pickup import scenario_config


def test_paired_methods_share_scene_and_fixed_baseline_is_original_success():
    fixed = scenario_config("cup_x_plus_5cm", "fixed")
    auto = scenario_config("cup_x_plus_5cm", "auto")
    assert fixed["pick_task"]["candidates"][0]["base_q"] == [1.2, -0.1, 0.0]
    assert fixed["pick_task"].pop("candidate_source") == "manual"
    assert auto["pick_task"].pop("candidate_source") == "auto"
    fixed["pick_task"].pop("candidates")
    auto["pick_task"].pop("candidates")
    assert fixed == auto
    nominal = scenario_config("nominal", "auto")
    assert abs(auto["objects"]["cup"]["center"][0] - nominal["objects"]["cup"]["center"][0] - 0.05) < 1e-12
