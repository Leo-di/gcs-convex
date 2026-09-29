"""Semantic predicates and their signed residuals."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .scene_spec import Region2D


def base_inside_region(q: Iterable[float], region: Region2D) -> bool:
    """Return whether the planar base origin lies inside ``region``."""

    q_array = np.asarray(tuple(q), dtype=float)
    if q_array.shape != (3,):
        raise ValueError("Expected planar configuration q=(x, y, yaw).")
    return region.contains_xy(float(q_array[0]), float(q_array[1]))


def base_inside_region_residual(q: Iterable[float], region: Region2D) -> float:
    """Return signed room-membership margin for the planar base origin."""

    q_array = np.asarray(tuple(q), dtype=float)
    if q_array.shape != (3,):
        raise ValueError("Expected planar configuration q=(x, y, yaw).")
    return region.residual_xy(float(q_array[0]), float(q_array[1]))


def evaluate_region_trace(
    path: Iterable[Iterable[float]],
    regions: dict[str, Region2D],
) -> list[dict[str, object]]:
    """Evaluate all region predicates at every path waypoint."""

    trace: list[dict[str, object]] = []
    for index, q in enumerate(path):
        q_array = np.asarray(tuple(q), dtype=float)
        values = {
            name: {
                "satisfied": base_inside_region(q_array, region),
                "residual": base_inside_region_residual(q_array, region),
            }
            for name, region in regions.items()
        }
        trace.append(
            {
                "index": index,
                "q": q_array.tolist(),
                "regions": values,
            }
        )
    return trace
