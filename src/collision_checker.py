"""Collision predicates and signed-distance residuals for the Drake scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .build_scene import SceneHandles, set_robot_configuration


@dataclass(frozen=True)
class CollisionStatus:
    """Collision result at one robot configuration."""

    collision_free: bool
    minimum_signed_distance: float
    residual: float
    closest_geometry_a: str | None
    closest_geometry_b: str | None


def evaluate_collision_status(
    scene: SceneHandles,
    q: Iterable[float],
    *,
    arm_q: Iterable[float] | None = None,
    gripper_q: Iterable[float] | None = None,
    safety_margin: float = 0.0,
    influence_distance: float = 10.0,
) -> CollisionStatus:
    """Evaluate collision clearance at planar configuration ``q``.

    The residual is ``minimum_signed_distance - safety_margin``. Therefore a
    nonnegative residual satisfies the requested clearance.
    """

    q_array = np.asarray(tuple(q), dtype=float)
    if q_array.shape != (3,):
        raise ValueError("Expected planar configuration q=(x, y, yaw).")
    if safety_margin < 0.0:
        raise ValueError("safety_margin must be nonnegative.")

    set_robot_configuration(
        scene,
        q_array,
        None if arm_q is None else np.asarray(tuple(arm_q), dtype=float),
        None if gripper_q is None else np.asarray(tuple(gripper_q), dtype=float),
    )
    scene_graph_context = scene.scene_graph.GetMyContextFromRoot(
        scene.root_context
    )
    query = scene.scene_graph.get_query_output_port().Eval(scene_graph_context)
    pairs = query.ComputeSignedDistancePairwiseClosestPoints(influence_distance)

    if not pairs:
        return CollisionStatus(
            collision_free=True,
            minimum_signed_distance=float("inf"),
            residual=float("inf"),
            closest_geometry_a=None,
            closest_geometry_b=None,
        )

    closest = min(pairs, key=lambda pair: pair.distance)
    inspector = query.inspector()
    minimum_distance = float(closest.distance)
    residual = minimum_distance - safety_margin

    return CollisionStatus(
        collision_free=residual >= 0.0,
        minimum_signed_distance=minimum_distance,
        residual=residual,
        closest_geometry_a=inspector.GetName(closest.id_A),
        closest_geometry_b=inspector.GetName(closest.id_B),
    )
