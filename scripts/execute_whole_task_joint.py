import argparse, hashlib, json, sys, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.whole_task_execution import frames_from_global_solution, execute_global_task
from src.whole_task_joint_gcs import write

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    p.add_argument("--time-scale", type=float)
    a = p.parse_args()
    run = a.run.resolve()
    try:
        read = lambda p: json.loads(p.read_text())
        protocol = read(run / "protocol.json")
        source = Path(protocol["source"])
        config = (
            read(run / "planning_config.json")
            if (run / "planning_config.json").exists()
            else read(source / "config.json")
        )
        spec = read(source / "protocol.json")["spec"]
        reference = read(source / "reference.json")
        search = read(source / "search.json")
        files = [
            ROOT / "src/whole_task_execution.py",
            ROOT / "src/stateful_transport_grounding.py",
            ROOT / "src/execute_delivery_dynamics.py",
            Path(__file__),
        ]
        write(
            run / "execution_protocol.json",
            dict(
                source_sha256={
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files
                },
                scope="No stage motion solve or boundary IK after the global solve. Only global trajectory reconstruction, prescribed finger transitions, validation and contact execution.",
            ),
        )
        frames, plan = frames_from_global_solution(config, spec, reference, run)
        print("GLOBAL_REFERENCE_ACCEPTED", len(frames), flush=True)
        if a.execute:
            if a.time_scale is not None:
                if a.time_scale <= 0:
                    raise ValueError("Execution time scale must be positive")
                write(
                    run / "execution_timing.json",
                    dict(
                        original_time_scale=spec["time_scale"],
                        execution_time_scale=a.time_scale,
                        scope="Uniform execution retiming of the unchanged globally optimized spatial path; no new motion solve.",
                    ),
                )
                spec["time_scale"] = a.time_scale
            result = execute_global_task(config, spec, frames, plan, search, run)
            print("COMPLETE_TASK_ACCEPTANCE", result["satisfied"], flush=True)
            if not result["satisfied"]:
                raise RuntimeError(
                    "Global joint task failed contact execution acceptance"
                )
    except Exception as e:
        write(
            run / "execution_failure.json",
            dict(error=str(e), traceback=traceback.format_exc()),
        )
        raise
