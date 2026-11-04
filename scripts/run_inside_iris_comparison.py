import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inside_iris_comparison import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compare conservative domains and direct Inside constraints in IRIS"
    )
    p.add_argument(
        "--problem", type=Path, default=ROOT / "configs/inside_compiler_demo.json"
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--method", choices=["both", "conservative", "nonlinear"], default="both"
    )
    a = p.parse_args()
    run(
        a.problem,
        a.output,
        methods=("conservative", "nonlinear") if a.method == "both" else (a.method,),
    )
