"""Mode-specific Inside compilation and independent whole-task auditing."""

import copy
import numpy as np
from .semantic_compiler import compile_predicates
from .whole_robot_room import shape_support
from .build_scene import set_robot_configuration
from .joint_approach_gcs import interpolate_q10


def attach_compilers(modes, constraints, interface):
    if interface not in ("conservative", "nonlinear"):
        raise ValueError("Unknown Inside interface")
    for mode in modes:
        document = copy.deepcopy(constraints["dsl"])
        if mode.held:
            document["predicates"].extend(constraints["held_predicates"])
        mode.compiled_inside = compile_predicates(
            document, mode.scene, gripper=mode.grip
        )
        mode.inside_interface = interface


def independent_values(compiled, q):
    """Evaluate separately composed transforms and scalar support geometry."""
    scene = compiled.scene
    set_robot_configuration(scene, q[:3], q[3:], compiled.gripper)
    values = []
    for index, shape, offset, predicate, region in compiled.items:
        from pydrake.multibody.tree import BodyIndex

        pose = (
            scene.plant.EvalBodyPoseInWorld(
                scene.plant_context, scene.plant.get_body(BodyIndex(index))
            )
            @ offset
        )
        for n, b in zip(region.normals, region.offsets):
            values.append(
                n @ pose.translation()
                + shape_support(shape, pose.rotation().matrix().T @ n)
                + predicate.margin_m
                - b
            )
    return np.asarray(values)


def audit_modes(modes, segments=None):
    rows = []
    for mode in modes:
        if segments is None:
            points = mode.guide
        else:
            selected = [s for s in segments if s["mode"] == mode.name]
            points = np.array([selected[0]["a"]] + [s["b"] for s in selected])
        dense = interpolate_q10(points, base_step=0.01, joint_step=0.01)
        maximum = -np.inf
        bad = []
        for i, q in enumerate(dense):
            g = mode.compiled_inside.g(q)
            independent = independent_values(mode.compiled_inside, q)
            if not np.allclose(g, independent, atol=1e-9, rtol=0):
                raise RuntimeError("Independent Inside evaluators disagree")
            maximum = max(maximum, float(g.max()))
            if g.max() > 1e-7 and len(bad) < 8:
                j = int(g.argmax())
                bad.append(
                    dict(
                        index=i,
                        q=q.tolist(),
                        g_max_m=float(g[j]),
                        constraint=mode.compiled_inside.labels[j],
                    )
                )
        rows.append(
            dict(
                mode=mode.name,
                dimension=10,
                samples=len(dense),
                maximum_g_m=maximum,
                satisfied=not bad,
                counterexamples=bad,
                payload_predicate=mode.held,
            )
        )
    return dict(satisfied=all(r["satisfied"] for r in rows), modes=rows)
