"""Smoke-test the four mode-specific manipulation phases end to end.

Builds the scene, runs the existing IK grasp/delivery planners to obtain
the reference keyframes, then grows IRIS regions and solves the 10-D GCS
for each manipulation mode. Prints per-mode statistics and writes a JSON
summary. Run:

LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.smoke_manipulation_gcs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .build_scene import build_scene
from .delivery_planner import plan_delivery
from .geometry_outbound_gcs import select_navigation_arm
from .grasp_planner import cartesian_arm_path, plan_pick
from .manipulation_gcs import phase_plan_to_dict, plan_manipulation_phase
from .scene_spec import PROJECT_ROOT, load_scene_config


def _interpolate(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    alphas = np.linspace(0.0, 1.0, count)
    return np.asarray([(1.0 - a) * start + a * end for a in alphas])


def _with_base(base_q: np.ndarray, arm_path: np.ndarray) -> np.ndarray:
    base = np.asarray(base_q, dtype=float)
    return np.asarray(
        [np.concatenate([base, arm]) for arm in arm_path], dtype=float
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs" / "results" / "manipulation_gcs_smoke.json")
    args = parser.parse_args()

    config = load_scene_config()
    scene = build_scene(config=config, visualize=False)
    start_q = np.asarray(config["robot"]["start"], dtype=float)
    pickup_q = np.asarray(config["pick_task"]["candidates"][0]["base_q"], dtype=float)

    navigation_arm = select_navigation_arm(scene, [start_q, pickup_q])
    print(f"navigation_arm = {np.round(navigation_arm, 3).tolist()}", flush=True)

    from .grasp_planner import IKSolveError

    pick_plan = None
    for candidate in config["pick_task"]["candidates"]:
        base = np.asarray(candidate["base_q"], dtype=float)
        try:
            pick_plan = plan_pick(scene, base)
            pickup_q = base
            print(f"pick candidate {candidate['name']} accepted", flush=True)
            break
        except IKSolveError as error:
            print(f"pick candidate {candidate['name']} rejected: {error}", flush=True)
    if pick_plan is None:
        raise SystemExit("No pickup candidate solved IK")
    delivery_plan = plan_delivery(scene, pick_plan)
    print(
        f"pick base={np.round(pick_plan.base_q, 2).tolist()} "
        f"dropoff base={np.round(delivery_plan.base_q, 2).tolist()}",
        flush=True,
    )

    pregrasp_reference = np.vstack(
        [
            _interpolate(navigation_arm, pick_plan.pregrasp_q, 45),
            _interpolate(pick_plan.pregrasp_q, pick_plan.grasp_q, 35)[1:],
        ]
    )
    lift_reference = _interpolate(pick_plan.grasp_q, pick_plan.lift_q, 45)
    lower_reference = cartesian_arm_path(
        scene, delivery_plan.base_q, delivery_plan.preplace_q,
        delivery_plan.place_q, 35, stage="lower",
    )
    retreat_reference = cartesian_arm_path(
        scene, delivery_plan.base_q, delivery_plan.place_q,
        delivery_plan.preplace_q, 35, stage="retreat",
    )

    scene_cache: dict = {}
    phases = {
        "pregrasp": dict(
            reference_q10=_with_base(pick_plan.base_q, pregrasp_reference),
        ),
        "grasp": dict(
            reference_q10=_with_base(pick_plan.base_q, lift_reference),
            X_GC=delivery_plan.X_GC,
        ),
        "release": dict(
            reference_q10=_with_base(delivery_plan.base_q, lower_reference),
            X_GC=delivery_plan.X_GC,
        ),
        "retreat": dict(
            reference_q10=_with_base(delivery_plan.base_q, retreat_reference),
            final_cup_pose=delivery_plan.place_cup_pose,
        ),
    }

    summary = {}
    for mode, kwargs in phases.items():
        print(f"[{mode}] planning...", flush=True)
        begin = time.perf_counter()
        try:
            plan = plan_manipulation_phase(
                mode=mode, config=config, scene_cache=scene_cache, **kwargs
            )
            data = phase_plan_to_dict(plan)
            data["wall_seconds"] = time.perf_counter() - begin
            data["status"] = "ok"
            print(
                f"[{mode}] OK regions={data['number_of_regions']} "
                f"seeds={data['seed_count']} bridged={data['bridged']} "
                f"iris={data['iris_solve_seconds']:.2f}s gcs={data['gcs_solve_seconds']:.2f}s "
                f"travel={data['arm_joint_travel']:.3f}rad "
                f"audit={data['trajectory_audit']['collisions']}/{data['trajectory_audit']['samples']} "
                f"wall={data['wall_seconds']:.1f}s",
                flush=True,
            )
        except RuntimeError as error:
            data = {
                "status": "failed",
                "error": str(error),
                "wall_seconds": time.perf_counter() - begin,
            }
            print(f"[{mode}] FAILED: {error}", flush=True)
        summary[mode] = data

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {args.output}")
    if any(item["status"] != "ok" for item in summary.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
