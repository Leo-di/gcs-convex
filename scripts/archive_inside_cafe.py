"""Archive compact numerical evidence without meshes or large recordings."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def archive(run, report, destination):
    read = lambda p: json.loads(p.read_text())
    protocol = read(run / "protocol.json")
    results = read(run / "results.json")
    if len(results) != 2 or not all(r["execution_satisfied"] for r in results):
        raise RuntimeError("Both methods must have passed execution")
    mismatches = [
        name
        for name, digest in protocol["source_sha256"].items()
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest
    ]
    if mismatches:
        raise RuntimeError("Source hashes differ: " + str(mismatches))
    destination.mkdir(parents=True, exist_ok=False)
    for source in (
        run / "protocol.json",
        run / "results.json",
        report / "summary.json",
        report / "comparison.png",
        report / "comparison.svg",
        report / "synchronized_state.png",
    ):
        shutil.copy2(source, destination / source.name)
    manifest = dict(
        input_files_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((run / "inputs").glob("*.json"))
        },
        physics_sha256={},
        parent_commit="71e6d2545274757f70b37058a5988c9ef21258ea",
        scope="Fresh full-task Inside comparison; large measured traces and Meshcat assets retained locally",
    )
    covers = {}
    for method in ("conservative", "nonlinear"):
        source, target = run / method, destination / method
        target.mkdir()
        for name in (
            "global_model.json",
            "solve_status.json",
            "optimized_contacts.json",
            "global_segments.json",
            "contact_transition_graph.json",
            "joint_inside_audit.json",
            "measured_inside_audit.json",
            "independent_passage.json",
            "result.json",
        ):
            shutil.copy2(source / name, target / name)
        manifest["physics_sha256"][method] = hashlib.sha256(
            (source / "physics.json").read_bytes()
        ).hexdigest()
        covers[method] = {
            folder.name: {
                name: read(folder / name)
                for name in ("compiled_inside.json", "regions.json", "complete.json")
            }
            for folder in sorted((source / "covers").iterdir())
            if folder.is_dir()
        }
        if method == "nonlinear":
            for mode, data in covers[method].items():
                candidates = [
                    r for r in data["regions.json"] if r["inside_active_nonlinear_rows"]
                ]
                if candidates:
                    row = candidates[0]
                    example = dict(
                        mode=mode,
                        region=row,
                        active_constraints=[
                            data["compiled_inside.json"]["constraints"][i]
                            for i in row["inside_active_nonlinear_rows"]
                        ],
                        scope="One actual nominal domain and returned IRIS region with nonlinear Inside rows",
                    )
                    (destination / "nonlinear_region_example.json").write_text(
                        json.dumps(example, indent=2) + "\n"
                    )
                    break
    raw = json.dumps(covers, allow_nan=False, separators=(",", ":")).encode()
    (destination / "covers.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    archive(a.run, a.report, a.output)
