"""Bounded, ordered candidate retry with full-validation acceptance.

Solver failure rejects an attempt, not the underlying physical task. The
supplied evaluator must validate the complete motion before returning.
"""

import json
from pathlib import Path
import time
from typing import Callable, Any

import numpy as np

from .pickup_candidates import generate_pickup_candidates


class CandidateRejected(RuntimeError):
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class CandidatesExhausted(RuntimeError):
    def __init__(self, report: dict):
        super().__init__("All configured pickup candidates failed; this is not a proof of task infeasibility")
        self.report = report


def select_candidate(scene, evaluator: Callable, *, report_path: Path | None = None) -> tuple[Any, dict]:
    source = scene.config["pick_task"].get("candidate_source", "manual")
    if source == "auto":
        candidates, generation = generate_pickup_candidates(scene.config)
    elif source == "manual":
        candidates = scene.config["pick_task"]["candidates"]
        generation = {"source": "manual"}
    else:
        raise ValueError(f"Unknown candidate_source: {source}")
    if not isinstance(candidates, list) or (not candidates and source == "manual"):
        raise ValueError("pick_task.candidates must be a nonempty list")
    names = set()
    normalized = []
    for item in candidates:
        name = item["name"]
        q = np.asarray(item["base_q"], dtype=float)
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Candidate names must be nonempty and unique")
        if q.shape != (3,) or not np.isfinite(q).all():
            raise ValueError("Candidate base_q must contain three finite values")
        names.add(name)
        normalized.append((name, q))

    report = {"policy": "first_fully_validated_in_candidate_order", "status": "running",
              "selected_candidate": None, "attempts": [], "candidate_count": len(candidates),
              "generation": generation,
              "failure_scope": "Generated/configured candidates and current solver settings only"}
    initial_positions = scene.plant.GetPositions(scene.plant_context).copy()
    initial_time = scene.root_context.get_time()
    started = time.perf_counter()

    def save():
        report["elapsed_seconds"] = time.perf_counter() - started
        if report_path is not None:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    for name, q in normalized:
        attempt = {"name": name, "base_q": q.tolist(), "stage": "start", "status": "running"}
        report["attempts"].append(attempt)
        save()
        began = time.perf_counter()
        try:
            value = evaluator(scene, q, attempt)
        except RuntimeError as error:
            attempt.update(status="rejected", reason=str(error), error_type=type(error).__name__,
                           details=getattr(error, "details", {}))
        except Exception as error:
            # Programming/configuration errors must not masquerade as geometry failures.
            attempt.update(status="error", reason=str(error), error_type=type(error).__name__)
            report["status"] = "error"
            raise
        else:
            attempt["status"] = "accepted"
            report.update(status="success", selected_candidate=name)
            return value, report
        finally:
            # Includes cup free-body pose; a rejected carry must not contaminate the next IK.
            scene.plant.SetPositions(scene.plant_context, initial_positions)
            scene.root_context.SetTime(initial_time)
            attempt["elapsed_seconds"] = time.perf_counter() - began
            save()

    report["status"] = "exhausted"
    save()
    raise CandidatesExhausted(report)
