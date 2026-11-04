"""Run an existing CLI with native IRIS/GCS invocation telemetry.

Install before importing project modules so imported Drake aliases are also
instrumented. Solver arguments/results are passed through unchanged.
"""

import argparse
import functools
import json
import runpy
import sys
import time
import traceback
from pathlib import Path


def install(path):
    import pydrake.planning as planning

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open("x", buffering=1)
    counter = 0

    def wrap(name, fun):
        @functools.wraps(fun)
        def measured(*args, **kwargs):
            nonlocal counter
            counter += 1
            index = counter
            wall = time.perf_counter()
            cpu = time.process_time()
            caller = [
                dict(file=f.filename, line=f.lineno, function=f.name)
                for f in traceback.extract_stack(limit=8)[:-1]
            ]
            stream.write(
                json.dumps(dict(event="start", id=index, solver=name, caller=caller))
                + "\n"
            )
            status = "returned"
            try:
                return fun(*args, **kwargs)
            except BaseException as exc:
                status = type(exc).__name__ + ": " + str(exc)
                raise
            finally:
                stream.write(
                    json.dumps(
                        dict(
                            event="end",
                            id=index,
                            solver=name,
                            status=status,
                            wall_seconds=time.perf_counter() - wall,
                            cpu_seconds=time.process_time() - cpu,
                        )
                    )
                    + "\n"
                )

        return measured

    planning.IrisNp2 = wrap("Drake.IrisNp2", planning.IrisNp2)
    cls = planning.GcsTrajectoryOptimization
    cls.SolvePath = wrap("Drake.GcsTrajectoryOptimization.SolvePath", cls.SolvePath)
    return stream


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--telemetry", type=Path, required=True)
    p.add_argument("--module")
    p.add_argument("--script", type=Path)
    p.add_argument("arguments", nargs=argparse.REMAINDER)
    a = p.parse_args()
    if bool(a.module) == bool(a.script):
        p.error("Choose one module or script")
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    stream = install(a.telemetry)
    args = a.arguments[1:] if a.arguments[:1] == ["--"] else a.arguments
    sys.argv = [a.module or str(a.script), *args]
    if a.module:
        runpy.run_module(a.module, run_name="__main__")
    else:
        runpy.run_path(str(a.script), run_name="__main__")
