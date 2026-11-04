"""Matched Inside interfaces in one complete base/arm task optimization."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.whole_task_joint_gcs import make_modes, grow_mode_cover, optimize_task, write
from src.inside_cafe import attach_compilers, audit_modes
from src.whole_task_execution import frames_from_global_solution, execute_global_task
from scripts.verify_service_cafe import verify
from scripts.audit_inside_cafe_execution import audit as audit_measured_inside


def run(guide, constraints_file, output, methods, execute):
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    inputs.mkdir()
    for name in ("config.json", "protocol.json", "reference.json", "search.json"):
        shutil.copy2(guide / name, inputs / name)
    constraints = json.loads(constraints_file.read_text())
    write(inputs / "inside.json", constraints)
    read = lambda name: json.loads((inputs / name).read_text())
    cfg, reference = read("config.json"), read("reference.json")
    spec = read("protocol.json")["spec"]
    cfg["whole_task_planning"] = constraints["planning"]
    write(inputs / "planning_config.json", cfg)
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").glob("*.py"))
    }
    for p in (
        Path(__file__),
        ROOT / "scripts/audit_inside_cafe_execution.py",
        ROOT / "scripts/verify_service_cafe.py",
    ):
        source_hashes[str(p.relative_to(ROOT))] = hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
    protocol = dict(
        task_order=[a["id"] for a in spec["actions"]],
        guide_sha256=hashlib.sha256(
            (inputs / "reference.json").read_bytes()
        ).hexdigest(),
        constraints=constraints,
        source_sha256=source_hashes,
        scope="Shared staged guide; fresh covers and one task-wide 10D solve per interface; prescribed task order and fingers",
    )
    protocol["matched_input_sha256"] = hashlib.sha256(
        json.dumps(protocol, sort_keys=True).encode()
    ).hexdigest()
    write(output / "protocol.json", protocol)
    results = []
    for method in methods:
        folder = output / method
        folder.mkdir()
        write(
            folder / "protocol.json",
            dict(
                source=str(inputs.resolve()),
                inside_interface=method,
                task_order=protocol["task_order"],
                matched_input_sha256=protocol["matched_input_sha256"],
            ),
        )
        write(folder / "planning_config.json", cfg)
        began = time.monotonic()
        try:
            modes, contacts = make_modes(copy.deepcopy(cfg), reference, spec["actions"])
            attach_compilers(modes, constraints, method)
            write(folder / "guide_inside_audit.json", audit_modes(modes))
            if not json.loads((folder / "guide_inside_audit.json").read_text())[
                "satisfied"
            ]:
                raise RuntimeError("Shared complete-task guide violates Inside")
            for i, mode in enumerate(modes):
                cover = folder / "covers" / mode.name
                grow_mode_cover(mode, cover, seed=20261104 + i * 1000, max_seconds=1800)
                write(cover / "compiled_inside.json", mode.compiled_inside.metadata())
            prepare = time.monotonic() - began
            answer = optimize_task(modes, contacts, folder, max_solver_seconds=900)
            segments = json.loads((folder / "global_segments.json").read_text())
            audit = audit_modes(modes, segments)
            write(folder / "joint_inside_audit.json", audit)
            if not audit["satisfied"]:
                raise RuntimeError("Global solution violates Inside")
            frames, plan = frames_from_global_solution(cfg, spec, reference, folder)
            summary = dict(
                method=method,
                matched_input_sha256=protocol["matched_input_sha256"],
                reference_satisfied=True,
                execution_satisfied=None,
                prepare_seconds=prepare,
                global_gcs_seconds=answer[-1],
                regions=sum(len(m.regions) for m in modes),
                iris_regions=sum(
                    r["kind"] == "official_IRIS_NP2_10D"
                    for m in modes
                    for r in m.records
                ),
                contact_patches=sum(
                    r["contact_nonlinear_required"] for m in modes for r in m.records
                ),
                region_growth_and_holdout_seconds=sum(
                    r["growth_seconds"] for m in modes for r in m.records
                ),
                joint_inside_audit=audit,
            )
            write(folder / "planning_result.json", summary)
            if execute:
                execution_spec = dict(
                    spec, time_scale=constraints["execution_time_scale"]
                )
                physical = execute_global_task(
                    cfg, execution_spec, frames, plan, read("search.json"), folder
                )
                if not physical["satisfied"]:
                    raise RuntimeError("Contact execution failed")
                summary["measured_inside_audit"] = audit_measured_inside(
                    folder, constraints
                )
                passage = verify(folder, folder / "independent_passage.json")
                summary["execution_satisfied"] = (
                    physical["satisfied"] and passage["human_access_satisfied"]
                )
                summary["actions"] = physical["actions"]
                summary["unexpected_contacts"] = physical["unexpected_contacts"]
                summary["human_access_satisfied"] = passage["human_access_satisfied"]
            summary["total_seconds"] = time.monotonic() - began
            summary["source_changes_during_run"] = [
                name
                for name, digest in source_hashes.items()
                if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
            ]
            if summary["source_changes_during_run"]:
                raise RuntimeError("Frozen source files changed during run")
            results.append(summary)
            write(folder / "summary.json", summary)
            write(output / "results.json", results)
        except Exception as exc:
            write(
                folder / "failure.json",
                dict(
                    error=str(exc),
                    traceback=traceback.format_exc(),
                    seconds=time.monotonic() - began,
                ),
            )
            raise
    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--guide-run", type=Path, required=True)
    p.add_argument(
        "--constraints", type=Path, default=ROOT / "configs/inside_cafe.json"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--method", choices=["both", "conservative", "nonlinear"], default="both"
    )
    p.add_argument("--execute", action="store_true")
    a = p.parse_args()
    run(
        a.guide_run,
        a.constraints,
        a.output,
        ("conservative", "nonlinear") if a.method == "both" else (a.method,),
        a.execute,
    )
