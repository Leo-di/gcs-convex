"""Inspect one fixed-base semantic domain and, optionally, its IRIS region."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.build_scene import build_scene
from src.green_room_gcs import make_operation_cover
from src.green_room_iris import grow_cover
from src.scene_spec import load_scene_config


def run(config_path, output, base, run_iris):
    config = load_scene_config(config_path)
    config['green_room']['corridor_padding_rad'] = .03
    scene = build_scene(config=config, visualize=False)
    arm = np.array(config['pick_task']['navigation_arm_q'], float)
    guide = np.array([arm, arm + np.r_[.002, np.zeros(6)]])
    _, records, reach = make_operation_cover(
        scene, np.array(base), guide, scene.gripper_open_positions, semantic=True
    )
    report = dict(
        example='Local joint perturbation at a fixed base; domain inspection only',
        config=str(config_path), base_q=base, guide=guide.tolist(),
        configured_padding_rad=.03, reach_bound_m=reach, semantic_domains=records,
        iris_run=False,
    )
    if run_iris:
        regions, iris_records, _ = grow_cover(
            scene, np.array(base), guide, scene.gripper_open_positions,
            semantic=True, domain_records=records, complete=False,
        )
        report.update(
            iris_run=True, iris_regions=iris_records,
            region_matrices=[dict(A=r.A().tolist(), b=r.b().tolist()) for r in regions],
        )
    output.mkdir(parents=True, exist_ok=False)
    (output/'inspection.json').write_text(json.dumps(report, indent=2))
    for i, record in enumerate(records):
        print(f"domain {i}: padding={record['corridor_padding_rad']:.6f} rad, "
              f"semantic lower bound={record['whole_box_semantic_lower_bound_m']:.6f} m")
    print(output/'inspection.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'configs/scene_doorway_table.yaml')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', type=float, nargs=3, default=[1.4, -.8, 0.])
    p.add_argument('--run-iris', action='store_true')
    args = p.parse_args()
    run(args.config, args.output, args.base, args.run_iris)
