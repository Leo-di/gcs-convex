"""Typed spatial predicates and world-frame convex regions."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Inside:
    id: str
    entity: str
    region: str
    margin_m: float


@dataclass(frozen=True)
class ConvexRegion:
    normals: np.ndarray
    offsets: np.ndarray


def parse(document):
    if (
        set(document) != {"version", "regions", "predicates"}
        or document["version"] != 1
    ):
        raise ValueError("Expected DSL version 1 with regions and predicates")
    regions = {}
    for name, spec in document["regions"].items():
        if spec.get("frame") != "world":
            raise ValueError("Inside currently requires a world-frame region")
        if spec.get("type") == "box" and set(spec) == {
            "type",
            "frame",
            "lower",
            "upper",
        }:
            lo, hi = np.array(spec["lower"], float), np.array(spec["upper"], float)
            if (
                lo.shape != (3,)
                or hi.shape != (3,)
                or not np.all(np.isfinite([lo, hi]))
                or np.any(lo >= hi)
            ):
                raise ValueError("Finite ordered three-dimensional box bounds required")
            A, b = np.vstack([-np.eye(3), np.eye(3)]), np.r_[-lo, hi]
        elif spec.get("type") == "polyhedron" and set(spec) == {
            "type",
            "frame",
            "A",
            "b",
        }:
            A, b = np.array(spec["A"], float), np.array(spec["b"], float)
        else:
            raise ValueError("Unsupported region specification")
        if (
            A.ndim != 2
            or A.shape[1] != 3
            or b.shape != (len(A),)
            or not len(A)
            or not np.all(np.isfinite(A))
            or not np.all(np.isfinite(b))
        ):
            raise ValueError("Finite 3D halfspaces required")
        lengths = np.linalg.norm(A, axis=1)
        if np.any(lengths <= 0):
            raise ValueError("Region normals must be nonzero")
        regions[name] = ConvexRegion(A / lengths[:, None], b / lengths)
    predicates = []
    ids = set()
    for spec in document["predicates"]:
        if (
            set(spec) != {"id", "op", "entity", "region", "margin_m"}
            or spec["op"] != "Inside"
        ):
            raise ValueError(
                "Supported primitive: Inside(id, entity, region, margin_m)"
            )
        if (
            spec["entity"] not in ("robot", "arm", "gripper", "base", "payload")
            or spec["region"] not in regions
        ):
            raise ValueError("Known entity and region required")
        if not isinstance(spec["id"], str) or not spec["id"] or spec["id"] in ids:
            raise ValueError("Distinct nonempty predicate IDs required")
        if (
            isinstance(spec["margin_m"], bool)
            or not np.isfinite(spec["margin_m"])
            or spec["margin_m"] < 0
        ):
            raise ValueError("Finite nonnegative margin required")
        ids.add(spec["id"])
        predicates.append(
            Inside(spec["id"], spec["entity"], spec["region"], float(spec["margin_m"]))
        )
    if not predicates:
        raise ValueError("At least one predicate required")
    return regions, predicates
