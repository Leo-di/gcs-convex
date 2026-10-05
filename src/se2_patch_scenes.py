"""Re-run falsely-failed scenes (startup kills / load timeouts) then republish.

Usage: LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 \
       nohup .venv/bin/python -m src.se2_patch_scenes > /tmp/se2_patch.log 2>&1 &
"""
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .benchmark_semantic_baselines import atomic_json
from .transfer_scene_experiment import run_process
from . import navigation_refresh_regression as reg

ROOT = Path("outputs/results/se2_navigation_refresh_v3")
SCENE_TIMEOUT_S = 21600
PARALLEL = 5


def false_failures():
    names = []
    for name in sorted(json.loads((ROOT / "design.json").read_text())["scenes"]):
        prep = ROOT / "libraries" / name / "preparation.json"
        if not prep.exists():
            names.append(name)
            continue
        d = json.loads(prep.read_text())
        # A scene counts as falsely failed only when the recorded failure is a
        # timeout/program error from machine contention, not a real result.
        if d.get("failure") in ("timeout", "program_error", "preparation_timeout",
                                "preparation_program_error"):
            names.append(name)
    return names


def main():
    todo = false_failures()
    print("PATCH_SCENES", todo, flush=True)
    began = time.perf_counter()
    cpus = sorted(__import__("os").sched_getaffinity(0))
    slots = [cpus[i] for i in [4, 12, 20, 28, 36]]

    def worker(i):
        for k, name in enumerate(todo[i::PARALLEL]):
            dest = ROOT / "libraries" / name
            dest.mkdir(parents=True, exist_ok=True)
            print("PATCH_START", name, flush=True)
            code, expired = run_process(
                ["taskset", "-c", str(slots[i]), sys.executable,
                 "-m", "src.navigation_refresh_regression", "--worker", name],
                dest / "prepare.log", SCENE_TIMEOUT_S,
            )
            if code or expired:
                progress = dest / "preparation_progress.json"
                d = json.loads(progress.read_text()) if progress.exists() else {"attempts": [], "goal_attempts": []}
                d.update(candidate_count=0, goal_count=0,
                         failure="timeout" if expired else "program_error",
                         return_code=code, seconds=time.perf_counter() - began)
                atomic_json(dest / "preparation.json", d)
            print("PATCH_DONE", name, code, flush=True)

    with ThreadPoolExecutor(PARALLEL) as pool:
        list(pool.map(worker, range(PARALLEL)))
    reg.publish(True)
    print("PATCH_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
