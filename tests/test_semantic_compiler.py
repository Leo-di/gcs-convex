import copy
import numpy as np
import pytest
from src.build_scene import build_scene
from src.scene_spec import load_scene_config
from src.semantic_dsl import parse
from src.semantic_compiler import compile_predicates
from src.whole_robot_room import WholeRobotRoomOracle


@pytest.fixture(scope="module")
def compiled():
    cfg = load_scene_config("configs/scene_doorway_table.yaml")
    scene = build_scene(config=cfg, visualize=False)
    document = dict(
        version=1,
        regions=dict(
            room=dict(
                type="box",
                frame="world",
                lower=[0.1, -1.4, -0.1],
                upper=[2.13, 1.4, 2.5],
            )
        ),
        predicates=[
            dict(id="inside", op="Inside", entity="robot", region="room", margin_m=0.02)
        ],
    )
    return compile_predicates(
        document, scene, base=[1.4, -0.8, 0], gripper=scene.gripper_open_positions
    ), document


def test_compiled_fk_matches_independent_full_geometry_oracle(compiled):
    c, _ = compiled
    r = c.regions["room"]
    oracle = WholeRobotRoomOracle(c.scene, r.normals, r.offsets, margin=0.02)
    rng = np.random.default_rng(14)
    for _ in range(20):
        q = np.array([0, -0.5, 0, -1.4, 0, 0, 0]) + rng.uniform(-0.15, 0.15, 7)
        assert c.g(q).max() == pytest.approx(
            -oracle.query(np.r_[c.base, q], c.gripper)["residual_m"], abs=1e-10
        )


def test_autodiff_gradient_matches_finite_differences(compiled):
    c, _ = compiled
    q = np.array([0.23, -0.71, 0.14, -1.28, 0.17, 0.21, -0.13])
    eps = 1e-6
    finite = np.column_stack(
        [
            (c.g(q + np.eye(7)[i] * eps) - c.g(q - np.eye(7)[i] * eps)) / (2 * eps)
            for i in range(7)
        ]
    )
    assert np.all(np.isfinite(c.jacobian(q)))
    np.testing.assert_allclose(c.jacobian(q), finite, atol=2e-6, rtol=2e-5)


def test_box_bound_dominates_sampled_joint_combinations(compiled):
    c, _ = compiled
    q = np.array([0.2, -0.6, 0.1, -1.4, 0.1, 0.2, -0.1])
    h = np.array([0.02, 0.03, 0.01, 0.02, 0.01, 0.02, 0.01])
    bound = c.box_upper_bound(q - h, q + h)
    rng = np.random.default_rng(16)
    for x in rng.uniform(q - h, q + h, (200, 7)):
        assert np.all(c.g(x) <= bound + 1e-10)
    base_rows = [i for i, v in enumerate(c.labels) if v["body"] == "mobile_base"]
    assert base_rows and np.all(c.lipschitz[base_rows] == 0)


def test_identical_program_and_point_evaluator_and_active_counterexample(compiled):
    c, _ = compiled
    p = c.to_program()
    binding = p.GetAllConstraints()[0]
    for angle in [-0.5, 0, 0.5]:
        q = np.array([angle, -0.3, 0, -1.4, 0, 0, 0])
        np.testing.assert_allclose(binding.evaluator().Eval(q), c.g(q))
        assert (c.g(q).max() > 0) == (angle == 0)


def test_unknown_primitive_frame_and_entity_are_rejected(compiled):
    _, document = compiled
    for key, value in [("op", "Above"), ("entity", "unknown")]:
        d = copy.deepcopy(document)
        d["predicates"][0][key] = value
        with pytest.raises(ValueError):
            parse(d)
    d = copy.deepcopy(document)
    d["regions"]["room"]["frame"] = "moving_cup"
    with pytest.raises(ValueError):
        parse(d)


def test_scaled_polyhedron_preserves_meter_margin_and_predicate_conjunction(compiled):
    c, document = compiled
    d = copy.deepcopy(document)
    region = c.regions["room"]
    scales = np.arange(1, 7)
    d["regions"]["room"] = dict(
        type="polyhedron",
        frame="world",
        A=(region.normals * scales[:, None]).tolist(),
        b=(region.offsets * scales).tolist(),
    )
    regions, _ = parse(d)
    np.testing.assert_allclose(regions["room"].normals, region.normals)
    np.testing.assert_allclose(regions["room"].offsets, region.offsets)
    d["predicates"].append(
        dict(id="arm_only", op="Inside", entity="arm", region="room", margin_m=0.02)
    )
    both = compile_predicates(d, c.scene, base=c.base, gripper=c.gripper)
    q = np.array([0.1, -0.5, 0.1, -1.4, 0, 0, 0])
    np.testing.assert_allclose(both.g(q)[: len(c.labels)], c.g(q))
    assert len(both.labels) > len(c.labels)
    assert {r["predicate"] for r in both.labels} == {"inside", "arm_only"}
