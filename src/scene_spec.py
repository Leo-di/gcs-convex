"""Configuration loading and small geometry data types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_CONFIG = PROJECT_ROOT / "configs" / "scene.yaml"


@dataclass(frozen=True)
class Region2D:
    """An axis-aligned semantic region in the world xy plane."""

    name: str
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def contains_xy(self, x: float, y: float, tolerance: float = 0.0) -> bool:
        return (
            self.x_min - tolerance <= x <= self.x_max + tolerance
            and self.y_min - tolerance <= y <= self.y_max + tolerance
        )

    def residual_xy(self, x: float, y: float) -> float:
        """Positive inside, zero on the boundary, and negative outside."""

        return min(
            x - self.x_min,
            self.x_max - x,
            y - self.y_min,
            self.y_max - y,
        )


def load_scene_config(path: Path | str = DEFAULT_SCENE_CONFIG) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError("Scene configuration must be a YAML mapping.")
    return data


def semantic_regions(config: dict[str, Any]) -> dict[str, Region2D]:
    result: dict[str, Region2D] = {}
    for name, spec in config["semantic_regions"].items():
        x_min, x_max = spec["x_bounds"]
        y_min, y_max = spec["y_bounds"]
        result[name] = Region2D(
            name=name,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
        )
    return result
