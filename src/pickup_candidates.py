"""Deterministic geometric sampling of pickup base configurations.

Candidates are proposals, not reachability/collision certificates. Ranking
prefers a configurable cup standoff, then shorter base displacement.
"""

from itertools import product
import math

import numpy as np

from .task_semantics import near_pickup_residual


def generate_pickup_candidates(config: dict) -> tuple[list[dict], dict]:
    spec = config["pick_task"]["candidate_generation"]
    step = float(spec["xy_step_m"])
    margin = float(spec["boundary_margin_m"])
    standoff = float(spec["preferred_standoff_m"])
    limit = spec["max_candidates"]
    yaws = np.asarray(spec["yaw_samples_rad"], dtype=float)
    if not (np.isfinite([step, margin, standoff]).all() and step > 0 and margin >= 0 and standoff > 0):
        raise ValueError("Candidate grid parameters must be finite, with positive step/standoff and nonnegative margin")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("max_candidates must be a positive integer")
    if yaws.ndim != 1 or not len(yaws) or not np.isfinite(yaws).all():
        raise ValueError("yaw_samples_rad must be a nonempty finite list")
    transitions = config["semantic_product_planning"]["transitions"]
    action = next(item for item in transitions if item["name"] == "pick_cup")
    region_names = action["candidate_regions"]
    regions = {item["name"]: item for item in config["gcs_base_planning"]["regions"]}
    unknown = set(region_names) - regions.keys()
    if unknown:
        raise ValueError(f"Unknown pickup candidate regions: {sorted(unknown)}")
    cup = np.asarray(config["objects"]["cup"]["center"][:2], dtype=float)
    start = np.asarray(config["robot"]["start"], dtype=float)
    proposals = {}
    sampled = 0
    for name in region_names:
        region = regions[name]
        low = np.asarray(region["lower"]) + margin
        high = np.asarray(region["upper"]) - margin
        axes = [range(math.ceil(a / step - 1e-9), math.floor(b / step + 1e-9) + 1)
                for a, b in zip(low, high)]
        if len(axes[0]) * len(axes[1]) * len(yaws) > 100_000:
            raise ValueError("Candidate grid exceeds 100000 samples per region")
        for ix, iy, yaw in product(*axes, yaws):
            sampled += 1
            q = np.round([ix * step, iy * step, yaw], 12)
            if not region["yaw_bounds"][0] <= yaw <= region["yaw_bounds"][1]:
                continue
            if near_pickup_residual(q, config) < -1e-9:
                continue
            score = (abs(float(np.linalg.norm(q[:2] - cup)) - standoff),
                     float(np.linalg.norm(q[:2] - start[:2])), abs(float(q[2] - start[2])))
            proposals[tuple(q)] = score
    ranked = sorted(proposals, key=lambda q: (*proposals[q], *q))
    candidates = [{"name": f"auto_{index:03d}", "base_q": [float(value) for value in q],
                   "ranking_cost": list(proposals[q])}
                  for index, q in enumerate(ranked[:limit])]
    return candidates, {
        "source": "geometric_grid", "settings": dict(spec),
        "ranking": "standoff_error_then_base_displacement_then_yaw_change",
        "sampled_count": sampled, "eligible_count": len(ranked),
        "returned_count": len(candidates), "truncated": len(ranked) > limit,
        "candidates": candidates,
        "guarantee": "Action-region bounds and pickup proximity only; full planning still required",
    }
