import numpy as np

from src.scene_spec import load_scene_config
from src.semantic_product_planner import plan_semantic_product


def test_product_plan_discovers_pick_then_place() -> None:
    config = load_scene_config()
    plan = plan_semantic_product(
        config,
        np.array(config["robot"]["start"], dtype=float),
        np.array(config["delivery_task"]["transfer_path"][-1], dtype=float),
    )

    assert plan.actions == ["pick_cup", "place_cup"]
    assert plan.regions_in_mode("seeking_pickup") == [
        "room1_central_lane",
        "doorway_lane",
        "room2_pickup_lane",
    ]
    assert plan.regions_in_mode("carrying_cup") == [
        "room2_pickup_lane",
        "doorway_lane",
        "room1_central_lane",
        "room1_upper_lane",
    ]
    assert plan.nodes[-1].mode == "delivered"


def test_product_plan_fails_when_pick_predicate_is_unavailable() -> None:
    config = load_scene_config()
    config["semantic_product_planning"]["region_predicates"][
        "room2_pickup_lane"
    ] = ["base_in_room2"]

    try:
        plan_semantic_product(
            config,
            config["robot"]["start"],
            config["delivery_task"]["transfer_path"][-1],
        )
    except RuntimeError as error:
        assert "No semantic product path" in str(error)
    else:
        raise AssertionError("Expected the missing pickup predicate to make the task infeasible.")
