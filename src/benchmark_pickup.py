"""Paired deterministic scene checks for fixed versus generated pickup bases.

This small diagnostic suite is not a statistical performance benchmark.
Each method uses identical scene geometry and the same full validation.
"""

import argparse
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .scene_spec import load_scene_config


SCENARIOS = {"nominal": 0.0, "cup_x_minus_5cm": -0.05, "cup_x_plus_5cm": 0.05}
METHODS = ("fixed", "auto")


def scenario_config(scenario: str, method: str) -> dict:
    config = deepcopy(load_scene_config())
    config["objects"]["cup"]["center"][0] += SCENARIOS[scenario]
    # Preserve the original known-good fixed base, not the previously failing x=1.0.
    config["pick_task"]["candidate_source"] = "manual" if method == "fixed" else "auto"
    if method == "fixed":
        config["pick_task"]["candidates"] = [{"name": "fixed_original", "base_q": [1.2, -0.1, 0.0]}]
    return config


def run_worker(directory: Path, scenario: str, method: str):
    from .animate_delivery_demo import plan_delivery_candidate
    from .build_scene import build_scene
    from .candidate_planner import CandidatesExhausted, select_candidate
    from .iris_region_generator import cover_to_dict

    config = scenario_config(scenario, method)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "scene_config_snapshot.json").write_text(json.dumps(config, indent=2))
    started = time.perf_counter()
    scene = build_scene(config=config, visualize=False)
    try:
        selected, report = select_candidate(scene, plan_delivery_candidate,
                                            report_path=directory / "candidate_attempts.json")
    except CandidatesExhausted as error:
        result = {"status": "exhausted", "attempts": len(error.report["attempts"])}
    else:
        (directory / "delivery_trace.json").write_text(json.dumps(selected["trace"], indent=2))
        (directory / "semantic_monitor.json").write_text(json.dumps(selected["monitor_report"], indent=2))
        cover = cover_to_dict(selected["carrying_iris_cover"])
        result = {"status": "success", "attempts": len(report["attempts"]),
                  "selected_candidate": report["selected_candidate"],
                  "selected_base_q": report["attempts"][-1]["base_q"],
                  "frames": len(selected["trace"]),
                  "collision_free": selected["collision_free"],
                  "semantic_monitor_satisfied": selected["monitor_report"]["satisfied"],
                  "grasp_geometry_satisfied": selected["grasp_audit"]["satisfied"],
                  "physical_grasp_validated": False,
                  "minimum_signed_distance": selected["minimum_distance"],
                  "cover_audit": cover["independent_holdout_audit"]}
    result.update(scenario=scenario, method=method, wall_seconds=time.perf_counter() - started)
    (directory / "result.json").write_text(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=240)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--scenario", choices=SCENARIOS, default="nominal")
    parser.add_argument("--method", choices=METHODS, default="auto")
    args = parser.parse_args()
    if args.worker:
        run_worker(args.output_dir, args.scenario, args.method)
        return
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Use an empty output directory to avoid mixing experiment runs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {"scope": "Three deterministic scenes, one run per method; not a success-rate estimate",
               "methods": {"fixed": "Original base (1.2,-0.1,0), one candidate",
                           "auto": "Geometric grid, at most configured max_candidates"},
               "python": sys.version, "drake": version("drake"),
               "timeout_seconds_per_run": args.timeout_seconds, "runs": []}
    for scenario in SCENARIOS:
        for method in METHODS:
            directory = args.output_dir / scenario / method
            directory.mkdir(parents=True)
            began = time.perf_counter()
            command = [sys.executable, "-m", "src.benchmark_pickup", "--worker",
                       "--output-dir", str(directory), "--scenario", scenario, "--method", method]
            with (directory / "run.log").open("w") as log:
                try:
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             timeout=args.timeout_seconds,
                                             env={**os.environ, "LCM_DEFAULT_URL": "memq://"})
                except subprocess.TimeoutExpired:
                    result = {"status": "timeout"}
                else:
                    result = (json.loads((directory / "result.json").read_text()) if process.returncode == 0
                              else {"status": "error", "exit_code": process.returncode})
            result.update(scenario=scenario, method=method, process_wall_seconds=time.perf_counter() - began)
            summary["runs"].append(result)
            (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
