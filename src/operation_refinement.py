"""Bounded counterexample-driven repair of official operation IRIS regions.

Only failed regions are replaced. Shared guide configurations, scene geometry,
collision filters and semantic margins stay fixed. The final cover needs a
fresh audit, actual intersection edges and the usual GCS/trajectory checks.
Finite audits and a failed bounded repair are not safety/infeasibility proofs.
"""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np


class OperationRefinementError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def failed_regions(audit, *, semantic):
    keys = ("collisions", "raw_grasp_violations") + (
        ("semantic_violations",) if semantic else ()
    )
    return [item for item in audit["region_failures"] if any(item[key] for key in keys)]


def split_domain(record, counterexample, *, shrink_factor=0.5):
    """Split along the guide near a witness, and narrow its box padding.

    This deliberately restricts the search domain. It never claims that the
    retained boxes are collision free; each is sent through official IRIS.
    """
    if not 0 < shrink_factor < 1:
        raise ValueError("Domain shrink factor must be between zero and one")
    lo, hi = np.asarray(record["lower"], float), np.asarray(record["upper"], float)
    a = np.asarray(record.get("guide_start", record.get("seed")), float)
    b = np.asarray(record.get("guide_end", record.get("seed")), float)
    witness = np.asarray(counterexample, float)
    if (
        lo.ndim != 1
        or any(x.shape != lo.shape for x in (hi, a, b, witness))
        or not all(np.all(np.isfinite(x)) for x in (lo, hi, a, b, witness))
        or np.any(lo >= hi)
        or np.any(np.minimum(a, b) < lo - 1e-10)
        or np.any(np.maximum(a, b) > hi + 1e-10)
    ):
        raise ValueError(
            "Finite nonempty domain and contained guide endpoints required"
        )
    padding = (
        np.maximum(0.0, np.minimum(np.minimum(a, b) - lo, hi - np.maximum(a, b)))
        * shrink_factor
    )
    delta = b - a
    if float(delta @ delta) < 1e-20:
        pairs = [(a, b)]
    else:
        alpha = float(np.clip((witness - a) @ delta / (delta @ delta), 0.25, 0.75))
        mid = a + alpha * delta
        pairs = [(a, mid), (mid, b)]
    children = []
    for start, end in pairs:
        lower = np.maximum(lo, np.minimum(start, end) - padding)
        upper = np.minimum(hi, np.maximum(start, end) + padding)
        if np.any(upper - lower <= 1e-12):
            raise RuntimeError("No positive-volume local refinement domain")
        children.append(
            {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
                "guide_start": start.tolist(),
                "guide_end": end.tolist(),
                "refinement_depth": int(record.get("refinement_depth", 0)) + 1,
                "refinement_witness": witness.tolist(),
            }
        )
    return children


def refine_cover(
    initial,
    audit_cover,
    regrow,
    *,
    seed,
    max_rounds=2,
    max_regions=512,
    time_budget_s=60.0,
    semantic=True,
    artifact=None,
):
    """Run local repair with injected real IRIS/audit operations.

    A failed validation batch becomes repair evidence. Every subsequent
    validation uses a new seed; successful construction samples are never
    reused as the final validation batch.
    """
    if (
        isinstance(max_rounds, bool)
        or int(max_rounds) != max_rounds
        or max_rounds < 0
        or max_regions < 1
        or not np.isfinite(time_budget_s)
        or time_budget_s <= 0
    ):
        raise ValueError(
            "Nonnegative integer rounds and positive finite limits required"
        )
    began = time.monotonic()
    regions, records, reach = initial
    report = {
        "status": "auditing",
        "rounds": [],
        "max_rounds": int(max_rounds),
        "max_regions": int(max_regions),
        "time_budget_s": float(time_budget_s),
        "final_audit": None,
        "repaired_regions": 0,
        "scope": "Local guide-domain refinement; official IRIS regeneration; fresh finite-sample validation. No scene/goal changes, route completeness or continuous-time certificate.",
    }

    def save():
        report["elapsed_seconds"] = time.monotonic() - began
        if artifact is not None:
            Path(artifact).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fail(reason):
        report.update(status="rejected", reason=reason)
        save()
        raise OperationRefinementError(reason, report)

    def check_time():
        if time.monotonic() - began >= time_budget_s:
            fail("Operation refinement time budget exhausted; no infeasibility claim")

    def audit(records, audit_seed):
        check_time()
        try:
            result = audit_cover(records, audit_seed)
        except Exception as exc:
            fail(f"Operation region audit failed: {exc}")
        return result

    for iteration in range(int(max_rounds) + 1):
        if len(regions) > max_regions:
            fail("Operation refinement region budget exhausted; no infeasibility claim")
        discovery = audit(records, int(seed) + 2 * iteration)
        failed = failed_regions(discovery, semantic=semantic)
        row = {
            "iteration": iteration,
            "region_count": len(regions),
            "discovery": discovery,
        }
        report["rounds"].append(row)
        save()
        check_time()
        if not failed:
            validation = audit(records, int(seed) + 2 * iteration + 1)
            row["validation"] = validation
            failed = failed_regions(validation, semantic=semantic)
            save()
            check_time()
            if not failed:
                report.update(status="accepted", final_audit=validation)
                save()
                return regions, records, reach, report
        if iteration == max_rounds:
            fail(
                "Operation region audit rejected after bounded refinement; no infeasibility claim"
            )
        replacements = {item["region_index"]: item for item in failed}
        next_regions, next_records = [], []
        for index, (region, record) in enumerate(zip(regions, records)):
            if index not in replacements:
                next_regions.append(region)
                next_records.append(record)
                continue
            check_time()
            failure = replacements[index]
            try:
                children = split_domain(record, failure["counterexamples"][0]["arm_q"])
                if (
                    len(next_regions) + len(children) + len(regions) - index - 1
                    > max_regions
                ):
                    fail(
                        "Operation refinement region budget exhausted; no infeasibility claim"
                    )
                grown, child_records, _ = regrow(children)
            except OperationRefinementError:
                raise
            except Exception as exc:
                fail(f"Local IRIS refinement failed: {exc}")
            row.setdefault("replacements", []).append(
                {
                    "region_index": index,
                    "original": record,
                    "children": child_records,
                    "counterexamples": failure["counterexamples"],
                }
            )
            next_regions.extend(grown)
            next_records.extend(child_records)
            report["repaired_regions"] += 1
            save()
        regions, records = next_regions, next_records


def build_audited_cover(
    scene, base, reference, gripper, *, X_GC=None, semantic=True, artifact=None
):
    from .green_room_iris import grow_cover
    from .green_room_gcs import region_holdout

    spec = scene.config["green_room"].get("operation_refinement", {})
    initial = grow_cover(scene, base, reference, gripper, X_GC=X_GC, semantic=semantic)

    def audit(records, seed):
        return region_holdout(scene, base, records, gripper, X_GC=X_GC, seed=seed)

    def regrow(records):
        return grow_cover(
            scene,
            base,
            reference,
            gripper,
            X_GC=X_GC,
            semantic=semantic,
            domain_records=records,
            complete=False,
        )

    return refine_cover(
        initial,
        audit,
        regrow,
        seed=int(scene.config["green_room"]["random_seed"]),
        max_rounds=spec.get("max_rounds", 2),
        max_regions=spec.get("max_regions", 512),
        time_budget_s=spec.get("time_budget_s", 60.0),
        semantic=semantic,
        artifact=artifact,
    )
