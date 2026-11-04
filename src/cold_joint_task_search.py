"""Coupled finite task search and task-wide joint trajectory admission.

Staged plans provide seeds for the full 10D task graph. Each complete candidate
requires a successful joint solve and independent reference checks. A failed
candidate returns to task search with the propagated world state.
"""

import copy
from functools import partial
import json
from pathlib import Path
import time

from . import stateful_transport_grounding as core
from .build_scene import build_scene
from .destination_suffix import check_destination_suffix
from .service_cafe_study import ordered_postures, reference_passage_audit
from .whole_task_joint_gcs import make_modes, grow_mode_cover, optimize_task, write
from .whole_task_execution import frames_from_global_solution


def search(
    config,
    spec,
    root,
    *,
    method="early",
    domain_policy=None,
    base_margin=0.015,
    rounded_paths=5,
):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = copy.deepcopy(config)
    cfg["whole_task_planning"] = dict(base_tracking_margin_m=base_margin)
    planning = copy.deepcopy(cfg)
    if method == "posthoc":
        planning["human_passage"]["enforce"] = False
    accepted = []
    attempts = []
    began = time.monotonic()

    def posthoc(answer, world, action, folder):
        frames, _, _ = answer
        scene = build_scene(
            config=core.world_config(cfg, world, action["object_id"]), visualize=False
        )
        audit = reference_passage_audit(scene, frames, stop_on_failure=True)
        write(folder / "posthoc_passage.json", audit)
        if not audit["satisfied"]:
            raise core.TransportRejected("posthoc_passage_rejected", audit)

    ground = partial(
        core.ground_action,
        pickup_cache={},
        posture_selector=ordered_postures,
        posthoc_accept=posthoc if method == "posthoc" else None,
        destination_precheck=check_destination_suffix if method == "early" else None,
    )

    def admit(frames, world, selected):
        index = len(attempts)
        folder = root / f"task_candidate_{index:03d}"
        folder.mkdir()
        candidate_spec = copy.deepcopy(spec)
        by_id = {a["id"]: a for a in spec["actions"]}
        candidate_spec["actions"] = [by_id[r["action"]] for r in selected]
        write(folder / "config.json", cfg)
        write(folder / "reference.json", frames)
        write(
            folder / "protocol.json",
            dict(
                spec=candidate_spec,
                scope="Freshly generated geometric guide, not the final task trajectory",
            ),
        )
        write(folder / "search.json", dict(selected=selected))
        row = dict(
            index=index,
            order=[r["action"] for r in selected],
            input_world_sha256=core.digest(world),
            status="running",
            planning_attempts=[],
        )
        attempts.append(row)
        write(root / "complete_task_attempts.json", attempts)
        remaining = lambda: float(spec["time_budget_s"] - (time.monotonic() - began))
        for retry in range(2 if method == "posthoc" else 1):
            tick = time.monotonic()
            out = folder / f"joint_{retry}"
            out.mkdir()
            (out / "covers").mkdir()
            policy = domain_policy or (
                "geometry_cap"
                if method == "posthoc" and retry == 0
                else "slack_adaptive"
            )
            write(
                out / "protocol.json",
                dict(
                    source=str(folder.resolve()),
                    task_order=row["order"],
                    domain_policy=policy,
                    base_tracking_margin_m=base_margin,
                    scope="One full task model; all base/arm states and six contact gates jointly variable",
                ),
            )
            write(out / "planning_config.json", cfg)
            trial = dict(folder=str(out), policy=policy, status="running")
            row["planning_attempts"].append(trial)
            try:
                if remaining() <= 0:
                    raise core.TransportRejected(
                        "whole_task_search_budget_exhausted", {}
                    )
                modes, contacts = make_modes(cfg, frames, candidate_spec["actions"])
                for i, mode in enumerate(modes):
                    if remaining() <= 0:
                        raise core.TransportRejected(
                            "whole_task_search_budget_exhausted", {}
                        )
                    grow_mode_cover(
                        mode,
                        out / "covers" / mode.name,
                        seed=20261102 + i * 1000,
                        max_seconds=min(1200, remaining()),
                        domain_policy=policy,
                    )
                optimize_task(
                    modes,
                    contacts,
                    out,
                    max_solver_seconds=min(900, max(1, remaining() / rounded_paths)),
                    contact_radius=0.04,
                    rounded_paths=rounded_paths,
                )
                frames_from_global_solution(cfg, candidate_spec, frames, out)
                if remaining() <= 0:
                    raise core.TransportRejected(
                        "whole_task_search_budget_exhausted_after_admission", {}
                    )
                trial.update(
                    status="accepted_reference", wall_seconds=time.monotonic() - tick
                )
                row.update(status="accepted_reference", accepted_joint=str(out))
                write(root / "complete_task_attempts.json", attempts)
                accepted.append(
                    dict(run=out, spec=candidate_spec, selection=selected, config=cfg)
                )
                return dict(satisfied=True, run=str(out))
            except Exception as exc:
                trial.update(
                    status="rejected",
                    wall_seconds=time.monotonic() - tick,
                    reason=str(exc),
                    stage=getattr(exc, "stage", "joint_solver_or_reference_audit"),
                )
                write(out / "admission_failure.json", trial)
                write(root / "complete_task_attempts.json", attempts)
                if remaining() <= 0:
                    break
        row["status"] = "rejected"
        write(root / "complete_task_attempts.json", attempts)
        return dict(satisfied=False, candidate=index, attempts=row["planning_attempts"])

    result = core.search(planning, spec, root, ground=ground, complete_accept=admit)
    write(root / "search.json", {k: v for k, v in result.items() if k != "frames"})
    if not result["satisfied"]:
        return result, None
    return result, accepted[-1]
