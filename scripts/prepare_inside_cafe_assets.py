"""Convert only the three furniture variants needed by the Inside cafe guide."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.replicacad import prepare_furniture, REVISION


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, default=ROOT / "data/replicacad")
    a = p.parse_args()
    variants = [
        ("frl_apartment_table_01", "cafe/frl_apartment_table_01", [1.2, 0.8, 0.75]),
        ("frl_apartment_chair_04", "cafe/frl_apartment_chair_04", [0.48, 0.48, 0.88]),
        ("frl_apartment_table_01", "human_passage/service_island", [1.6, 0.4, 0.9]),
    ]
    for template, relative, size in variants:
        folder = ROOT / "data/replicacad_drake" / relative
        if (folder / "asset.json").exists():
            if (
                json.loads((folder / "asset.json").read_text())["normalized_size"]
                != size
            ):
                raise ValueError("Existing asset has a different scale: " + relative)
        else:
            prepare_furniture(a.assets.resolve(), template, folder.resolve(), size=size)
        print(relative)
    print("ReplicaCAD source revision:", REVISION)
