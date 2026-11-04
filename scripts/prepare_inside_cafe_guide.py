"""Unpack the frozen, portable common guide; this does not solve a task."""

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    with gzip.open(ROOT / "configs/examples/inside_cafe_guide.json.gz", "rt") as f:
        bundle = json.load(f)
    for name, value in bundle["files"].items():
        if name not in (
            "config.json",
            "reference.json",
            "protocol.json",
            "search.json",
        ):
            raise ValueError("Unknown guide member")
        (a.output / name).write_text(value)
    print(bundle["scope"])
