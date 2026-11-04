"""Single entry point for cold finite task search + whole-task joint GCS."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--method", choices=["early", "posthoc"], default="early")
    p.add_argument("--order-mode", choices=["fixed", "search"], default="search")
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Install telemetry before importing project modules and their Drake aliases.
    from scripts.run_with_solver_telemetry import install

    telemetry = install(out / "native_solvers.jsonl")
    from src.cold_joint_task_search import search
    from src.stateful_transport_grounding import make_config
    from src.whole_task_execution import execute_global_task
    from scripts.verify_service_cafe import verify

    spec = json.loads(a.task.read_text())
    original_scene = Path(spec["scene"])
    if not original_scene.is_absolute():
        original_scene = ROOT / original_scene
    (out / "inputs").mkdir()
    shutil.copy2(original_scene, out / "inputs/scene.json")
    spec["scene"] = str(out / "inputs/scene.json")
    spec["task_order"] = dict(spec.get("task_order", {}), mode=a.order_mode)
    (out / "inputs/task.json").write_text(json.dumps(spec, indent=2))
    files = [
        *sorted((ROOT / "src").glob("*.py")),
        Path(__file__).resolve(),
        ROOT / "scripts/run_with_solver_telemetry.py",
        ROOT / "scripts/verify_service_cafe.py",
    ]
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (out / "protocol.json").write_text(
        json.dumps(
            dict(
                spec=spec,
                method=a.method,
                source_sha256=hashes,
                input_task_sha256=hashlib.sha256(a.task.read_bytes()).hexdigest(),
                input_scene_sha256=hashlib.sha256(
                    original_scene.read_bytes()
                ).hexdigest(),
                scope="Cold task inputs, finite discrete search with feedback from complete joint optimization and reference admission",
            ),
            indent=2,
        )
    )
    began = time.monotonic()
    try:
        result, accepted = search(make_config(spec), spec, out, method=a.method)
        summary = dict(
            reference_satisfied=result["satisfied"],
            satisfied=False,
            planning_wall_seconds=time.monotonic() - began,
            selected_task_order=result["selected_task_order"],
        )
        if accepted:
            run = accepted["run"]
            summary["accepted_joint"] = str(run)
            if a.execute:
                frames = json.loads((run / "reference.json").read_text())
                plan = json.loads((run / "semantic_plan.json").read_text())
                execution_spec = dict(accepted["spec"], time_scale=4.5)
                physical = execute_global_task(
                    accepted["config"],
                    execution_spec,
                    frames,
                    plan,
                    dict(selected=accepted["selection"]),
                    run,
                )
                if not physical["satisfied"]:
                    raise RuntimeError(
                        "Complete contact execution failed; unchanged acceptance thresholds"
                    )
                audit = verify(run, run / "independent_passage.json")
                summary["satisfied"] = audit["human_access_satisfied"]
        summary["total_wall_seconds"] = time.monotonic() - began
        summary["source_changes_during_run"] = [
            f
            for f, h in hashes.items()
            if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h
        ]
        (out / "result.json").write_text(json.dumps(summary, indent=2))
        if summary["source_changes_during_run"]:
            raise RuntimeError("Source files changed during the frozen experiment")
        if not summary["reference_satisfied"] or (
            a.execute and not summary["satisfied"]
        ):
            raise RuntimeError("Task did not pass requested acceptance")
    except Exception as exc:
        (out / "pipeline_failure.json").write_text(
            json.dumps(
                dict(error=str(exc), wall_seconds=time.monotonic() - began), indent=2
            )
        )
        raise
