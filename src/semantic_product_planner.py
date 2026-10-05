"""Discrete semantic planning on a product of regions and task modes.

The continuous motion inside and between regions is delegated to GCS.  This
module decides which copy of each convex region is active for each task mode
and where discrete actions such as picking and placing may occur.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any

import numpy as np


@dataclass(frozen=True, order=True)
class ProductNode:
    region: str
    mode: str


@dataclass(frozen=True)
class ProductStep:
    source: ProductNode
    target: ProductNode
    kind: str
    name: str
    cost: float


@dataclass(frozen=True)
class SemanticProductPlan:
    nodes: list[ProductNode]
    steps: list[ProductStep]
    total_cost: float
    expanded_nodes: int

    @property
    def actions(self) -> list[str]:
        return [step.name for step in self.steps if step.kind == "action"]

    def regions_in_mode(self, mode: str) -> list[str]:
        """Return the de-duplicated spatial route followed in one mode."""

        result: list[str] = []
        for node in self.nodes:
            if node.mode == mode and (not result or result[-1] != node.region):
                result.append(node.region)
        return result


def _region_specs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(spec["name"]): spec
        for spec in config["gcs_base_planning"]["regions"]
    }


def _contains(spec: dict[str, Any], xy: np.ndarray) -> bool:
    lower = np.asarray(spec["lower"], dtype=float)
    upper = np.asarray(spec["upper"], dtype=float)
    return bool(np.all(xy >= lower - 1e-9) and np.all(xy <= upper + 1e-9))


def _overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    lower = np.maximum(
        np.asarray(a["lower"], dtype=float),
        np.asarray(b["lower"], dtype=float),
    )
    upper = np.minimum(
        np.asarray(a["upper"], dtype=float),
        np.asarray(b["upper"], dtype=float),
    )
    return bool(np.all(lower <= upper + 1e-9))


def _center(spec: dict[str, Any]) -> np.ndarray:
    return 0.5 * (
        np.asarray(spec["lower"], dtype=float)
        + np.asarray(spec["upper"], dtype=float)
    )


def _movement_adjacency(
    regions: dict[str, dict[str, Any]],
) -> dict[str, list[tuple[str, float]]]:
    adjacency: dict[str, list[tuple[str, float]]] = {
        name: [] for name in regions
    }
    names = list(regions)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            if not _overlap(regions[first], regions[second]):
                continue
            cost = float(np.linalg.norm(_center(regions[first]) - _center(regions[second])))
            adjacency[first].append((second, cost))
            adjacency[second].append((first, cost))
    return adjacency


def _neighbors(
    node: ProductNode,
    *,
    adjacency: dict[str, list[tuple[str, float]]],
    region_predicates: dict[str, set[str]],
    transitions: list[dict[str, Any]],
) -> list[tuple[ProductNode, ProductStep]]:
    result: list[tuple[ProductNode, ProductStep]] = []
    for next_region, cost in adjacency[node.region]:
        target = ProductNode(next_region, node.mode)
        result.append(
            (
                target,
                ProductStep(node, target, "move", "move", cost),
            )
        )

    predicates = region_predicates.get(node.region, set())
    for transition in transitions:
        if transition["source"] != node.mode:
            continue
        # An action site is a candidate, not a region-wide geometric guard.
        if node.region not in transition.get("candidate_regions", region_predicates):
            continue
        required = set(transition.get("requires", []))
        if not required.issubset(predicates):
            continue
        target = ProductNode(node.region, str(transition["target"]))
        result.append(
            (
                target,
                ProductStep(
                    node,
                    target,
                    "action",
                    str(transition["name"]),
                    float(transition.get("cost", 0.0)),
                ),
            )
        )
    return result


def plan_semantic_product(
    config: dict[str, Any],
    start_q: np.ndarray | list[float],
    goal_q: np.ndarray | list[float],
) -> SemanticProductPlan:
    """Find the lowest-cost path through ``region x task-mode`` space."""

    start_q = np.asarray(start_q, dtype=float)
    goal_q = np.asarray(goal_q, dtype=float)
    if start_q.shape != (3,) or goal_q.shape != (3,):
        raise ValueError("Expected planar configurations q=(x, y, yaw).")

    regions = _region_specs(config)
    product_spec = config["semantic_product_planning"]
    start_regions = [name for name, spec in regions.items() if _contains(spec, start_q[:2])]
    goal_regions = [name for name, spec in regions.items() if _contains(spec, goal_q[:2])]
    if not start_regions:
        raise ValueError("The start is outside every convex region.")
    if not goal_regions:
        raise ValueError("The goal is outside every convex region.")

    initial_mode = str(product_spec["initial_mode"])
    goal_mode = str(product_spec["goal_mode"])
    start_nodes = [ProductNode(region, initial_mode) for region in start_regions]
    goal_nodes = {ProductNode(region, goal_mode) for region in goal_regions}
    adjacency = _movement_adjacency(regions)
    region_predicates = {
        name: set(predicates)
        for name, predicates in product_spec["region_predicates"].items()
    }
    transitions = list(product_spec["transitions"])

    distances: dict[ProductNode, float] = {}
    previous: dict[ProductNode, tuple[ProductNode, ProductStep]] = {}
    queue: list[tuple[float, ProductNode]] = []
    for node in start_nodes:
        distances[node] = 0.0
        heapq.heappush(queue, (0.0, node))

    expanded = 0
    reached: ProductNode | None = None
    while queue:
        distance, node = heapq.heappop(queue)
        if distance > distances.get(node, float("inf")) + 1e-12:
            continue
        expanded += 1
        if node in goal_nodes:
            reached = node
            break
        for next_node, step in _neighbors(
            node,
            adjacency=adjacency,
            region_predicates=region_predicates,
            transitions=transitions,
        ):
            candidate = distance + step.cost
            if candidate + 1e-12 >= distances.get(next_node, float("inf")):
                continue
            distances[next_node] = candidate
            previous[next_node] = (node, step)
            heapq.heappush(queue, (candidate, next_node))

    if reached is None:
        raise RuntimeError(
            f"No semantic product path reaches mode {goal_mode!r} at the goal."
        )

    nodes = [reached]
    steps: list[ProductStep] = []
    cursor = reached
    while cursor not in start_nodes:
        parent, step = previous[cursor]
        nodes.append(parent)
        steps.append(step)
        cursor = parent
    nodes.reverse()
    steps.reverse()
    return SemanticProductPlan(
        nodes=nodes,
        steps=steps,
        total_cost=distances[reached],
        expanded_nodes=expanded,
    )


def plan_to_dict(plan: SemanticProductPlan) -> dict[str, Any]:
    return {
        "nodes": [
            {"region": node.region, "mode": node.mode}
            for node in plan.nodes
        ],
        "steps": [
            {
                "kind": step.kind,
                "name": step.name,
                "source": {"region": step.source.region, "mode": step.source.mode},
                "target": {"region": step.target.region, "mode": step.target.mode},
                "cost": step.cost,
            }
            for step in plan.steps
        ],
        "actions": plan.actions,
        "total_cost": plan.total_cost,
        "expanded_nodes": plan.expanded_nodes,
    }
