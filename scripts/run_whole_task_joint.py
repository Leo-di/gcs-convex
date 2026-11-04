"""Build one complete-task optimization and retain every failed attempt."""

import argparse, hashlib, json, shutil, sys, time, traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.whole_task_joint_gcs import make_modes, grow_mode_cover, optimize_task, write
from pydrake.geometry.optimization import HPolyhedron


def run(
    out,
    reuse=None,
    source=None,
    domain_policy="guide_local",
    contact_radius=None,
    rounded_paths=1,
    base_margin=0.0,
    add_base_guard=False,
):
    out.mkdir(parents=True, exist_ok=False)
    source = (
        source
        or ROOT
        / "outputs/results/multibranch_layout_study_20261101_v1/cases/asymmetric_chairs/branch_early"
    )
    read = lambda p: json.loads(p.read_text())
    cfg = read(source / "config.json")
    spec = read(source / "protocol.json")["spec"]
    reference = read(source / "reference.json")
    sources = [
        source / "config.json",
        source / "protocol.json",
        source / "reference.json",
        ROOT / "src/whole_task_joint_gcs.py",
        ROOT / "src/semantic_domain_expansion.py",
        ROOT / "src/task_order_search.py",
        ROOT / "src/base_clearance_domains.py",
        Path(__file__),
    ]
    protocol = dict(
        source=str(source),
        task_order=[a["id"] for a in spec["actions"]],
        optimization="One global 10D mode graph with six variable exact-FK contact gates",
        fixed_internal_robot_configurations=0,
        domain_policy=domain_policy,
        contact_radius=contact_radius,
        rounded_paths=rounded_paths,
        reuse_covers=str(reuse) if reuse else None,
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    )
    write(out / "protocol.json", protocol)
    snapshot = out / "source_snapshot"
    snapshot.mkdir()
    for f in sources:
        if f.suffix == ".py":
            shutil.copyfile(f, snapshot / f.name)
    cfg["whole_task_planning"] = dict(base_tracking_margin_m=base_margin)
    write(out / "planning_config.json", cfg)
    print("TASK_MODEL_BUILD", flush=True)
    modes, contacts = make_modes(cfg, reference, spec["actions"])
    (out / "covers").mkdir()
    for i, mode in enumerate(modes):
        folder = out / "covers" / mode.name
        cached = reuse / "covers" / mode.name if reuse else None
        if cached and (cached / "complete.json").exists():
            info = read(cached / "complete.json")
            assert info.get("domain_policy", "guide_local") == domain_policy
            assert info.get("base_tracking_margin_m", 0.0) == base_margin
            assert (
                info["guide_sha256"] == hashlib.sha256(mode.guide.tobytes()).hexdigest()
            )
            shutil.copytree(cached, folder)
            mode.records = read(folder / "regions.json")
            mode.regions = [
                HPolyhedron(np.array(r["A"]), np.array(r["b"])) for r in mode.records
            ]
            print("REUSED", mode.name, len(mode.regions), flush=True)
            if add_base_guard:
                from src.base_clearance_domains import intersect_cached

                pairs = [
                    intersect_cached(mode, record, base_margin)
                    for record in mode.records
                ]
                mode.regions = [p[0] for p in pairs]
                mode.records = [p[1] for p in pairs]
                write(folder / "regions.json", mode.records)
        else:
            grow_mode_cover(
                mode, folder, seed=20261102 + i * 1000, domain_policy=domain_policy
            )
    print("ALL_COVERS_READY", flush=True)
    answer = optimize_task(
        modes, contacts, out, contact_radius=contact_radius, rounded_paths=rounded_paths
    )
    print("WHOLE_TASK_SOLVED", answer[-1], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reuse-covers", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument(
        "--domain-policy",
        choices=["guide_local", "slack_adaptive", "geometry_cap"],
        default="guide_local",
    )
    p.add_argument("--contact-radius", type=float)
    p.add_argument("--rounded-paths", type=int, default=1)
    p.add_argument("--base-margin", type=float, default=0.0)
    p.add_argument("--add-base-guard", action="store_true")
    a = p.parse_args()
    try:
        run(
            a.output.resolve(),
            a.reuse_covers.resolve() if a.reuse_covers else None,
            a.source.resolve() if a.source else None,
            a.domain_policy,
            a.contact_radius,
            a.rounded_paths,
            a.base_margin,
            a.add_base_guard,
        )
    except Exception as e:
        if a.output.exists():
            write(
                a.output / "failure.json",
                dict(error=str(e), traceback=traceback.format_exc()),
            )
        raise
