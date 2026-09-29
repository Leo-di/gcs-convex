"""Animate a reference path and report semantic predicate residuals."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .build_scene import build_scene, set_robot_configuration
from .scene_spec import PROJECT_ROOT, semantic_regions
from .semantic_predicates import evaluate_region_trace


def interpolate_waypoints(
    waypoints: np.ndarray,
    samples_per_segment: int = 20,
) -> np.ndarray:
    samples: list[np.ndarray] = []
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        for alpha in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            samples.append((1.0 - alpha) * start + alpha * end)
    samples.append(waypoints[-1])
    return np.vstack(samples)


def main() -> None:
    scene = build_scene()
    waypoints = np.asarray(scene.config["reference_path"], dtype=float)
    path = interpolate_waypoints(waypoints)
    regions = semantic_regions(scene.config)
    trace = evaluate_region_trace(path, regions)

    output_path = PROJECT_ROOT / "outputs" / "results" / "semantic_trace.json"
    output_path.write_text(json.dumps(trace, indent=2), encoding="utf-8")

    scene.meshcat.StartRecording()
    for index, q in enumerate(path):
        set_robot_configuration(scene, q)
        scene.root_context.SetTime(index * 0.05)
        scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()

    final_regions = trace[-1]["regions"]
    print(f"Meshcat URL: {scene.meshcat.web_url()}")
    print(f"Saved semantic trace: {output_path}")
    print("Final predicate values:", final_regions)
    input("Press Enter to close the animation... ")


if __name__ == "__main__":
    main()
