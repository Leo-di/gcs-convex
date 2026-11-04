"""Finite task-order search with explicit precedence and shared candidate ranking.

The distance score only orders trials. It is not a feasibility test, lower
bound, or task-optimality certificate. Every admitted action is still grounded
against the complete propagated world by the normal motion backend.
"""

import numpy as np


def validate_order(spec):
    policy = spec.get("task_order", {"mode": "fixed"})
    if not isinstance(policy, dict) or set(policy) - {"mode", "precedence", "ranking"}:
        raise ValueError("Unknown task_order fields")
    if policy.get("mode", "fixed") not in ("fixed", "search"):
        raise ValueError("task_order mode must be fixed or search")
    if policy.get("ranking", "input") not in ("input", "nearest_pickup"):
        raise ValueError("Unknown task-order ranking")
    ids = [a["id"] for a in spec["actions"]]
    edges = policy.get("precedence", [])
    if not isinstance(edges, list) or any(
        not isinstance(e, list)
        or len(e) != 2
        or e[0] not in ids
        or e[1] not in ids
        or e[0] == e[1]
        for e in edges
    ):
        raise ValueError("Precedence edges require two distinct known action IDs")
    remaining = set(ids)
    done = set()
    while remaining:
        ready = {a for a in remaining if all(u in done for u, v in edges if v == a)}
        if not ready:
            raise ValueError("Cyclic action precedence")
        remaining -= ready
        done |= ready
    if policy.get("mode", "fixed") == "fixed" and any(
        ids.index(u) >= ids.index(v) for u, v in edges
    ):
        raise ValueError("Fixed input order contradicts precedence")
    return policy


def available_actions(spec, selected, world):
    policy = validate_order(spec)
    done = {r["action"] for r in selected}
    actions = [a for a in spec["actions"] if a["id"] not in done]
    if policy.get("mode", "fixed") == "fixed":
        return actions[:1]
    actions = [
        a
        for a in actions
        if all(u in done for u, v in policy.get("precedence", []) if v == a["id"])
    ]
    if policy.get("ranking", "input") == "nearest_pickup":
        base = np.array(world["robot"]["base_q"][:2])

        # Stable ties retain input order, identically for all semantic policies.
        def score(a):
            candidates = a.get("pickup_candidates", [{"base_q": a["pickup_base"]}])
            return min(
                float(np.linalg.norm(np.array(c["base_q"][:2]) - base))
                for c in candidates
            )

        actions.sort(key=score)
    return actions
