"""Uniform measured-body evaluation for all doorway scope methods."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.build_scene import build_scene
from src.green_room_gcs import room_oracle
from src.doorway_scope_study import METHODS


def verify(root, methods=METHODS):
    root = Path(root); rows = []
    configs = [json.loads((root/'configs'/f'{m}.json').read_text()) for m in METHODS]
    for key in ('world','objects','robot','arm','gripper','green_room','semantic_refinement'):
        assert all(c[key] == configs[0][key] for c in configs), key
    protocol = json.loads((root/'protocol.json').read_text())
    for method in methods:
        search = json.loads((root/method/'operation_first/search_result.json').read_text())
        screen = json.loads((root/method/'operation_first/candidate_screen.json').read_text())
        assert [r['id'] for r in screen] == [c['id'] for c in protocol['candidates']]
        row = dict(method=method, success=search.get('success', False), selected=search.get('selected'),
                   search_wall_seconds=search.get('total_wall_seconds'), attempts=len(search.get('attempts', [])))
        if not row['success']:
            row['failures'] = search.get('attempts', []); rows.append(row); continue
        folder = root/method/'operation_first'/search['selected']
        result = json.loads((folder/'candidate_result.json').read_text())
        physical = json.loads((folder/'execution/execution_grounded_candidate/physics.json').read_text())
        scene = build_scene(config=json.loads((folder/'scene_config_snapshot.json').read_text()), visualize=False)
        oracle = room_oracle(scene); samples = []
        for f in physical['trace']:
            r = oracle.query(np.r_[f['base_q'], f['arm_q']], f['gripper_q'])
            samples.append(dict(time_s=f['time_s'], phase=f['phase'],
                                base_residual_m=r['base_residual_m'], room_residual_m=r['residual_m'],
                                arm_gripper_residual_m=r['arm_gripper_residual_m'], worst=r['worst']))
        active = [s for s in samples if s['phase'] != 'navigate_to_pickup']
        assert all(result[k]['accepted'] for k in ('bridge','operation','physical'))
        assert result['physical']['unexpected_contacts'] == 0
        accepted = json.loads((folder/'delivery_trace.json').read_text())
        executed = json.loads((folder/'execution/execution_grounded_candidate/delivery_trace.json').read_text())
        assert accepted == executed
        row.update(base_q=result['base_q'], minimum_base_residual_m=min(s['base_residual_m'] for s in active),
                   minimum_room_residual_m=min(s['room_residual_m'] for s in active),
                   minimum_arm_gripper_residual_m=min(s['arm_gripper_residual_m'] for s in active),
                   minimum_raw_room_residual_m=min(s['room_residual_m'] for s in active)+scene.config['green_room']['margin_m'],
                   worst_sample=min(active, key=lambda s:s['room_residual_m']),
                   initial_operation_room_residual_m=active[0]['room_residual_m'],
                   base_violating_samples=sum(s['base_residual_m'] < -1e-8 for s in active),
                   raw_room_violating_samples=sum(s['room_residual_m']+scene.config['green_room']['margin_m'] < -1e-8 for s in active),
                   violating_samples=sum(s['room_residual_m'] < -1e-8 for s in active), operation_samples=len(active),
                   unexpected_contacts=result['physical']['unexpected_contacts'],
                   supplied_plan_satisfied=result['physical']['supplied_plan_satisfied'],
                   navigation_path_length_m=result['bridge']['path_length_m'],
                   arm_path_length_rad=result['operation']['path_length_rad'],
                   physical_duration_s=physical['trace'][-1]['time_s'],
                   reference_exactly_matches_execution=True,
                   replay=str((folder/'execution/execution_grounded_candidate/scene.html').relative_to(root)))
        # Uniform evaluation must retain violations, including in the full method.
        row['base_satisfied'] = row['minimum_base_residual_m'] >= -1e-8
        row['whole_body_satisfied'] = row['minimum_room_residual_m'] >= -1e-8
        row['raw_whole_body_satisfied'] = row['minimum_raw_room_residual_m'] >= -1e-8
        (folder/'room_scope_time_series.json').write_text(json.dumps(samples))
        rows.append(row)
    manifest=json.loads((root/'source_manifest.json').read_text())
    matches=all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in manifest.items())
    assert matches
    report=dict(rows=rows, same_scene_and_candidates=True, source_manifest_matches=True,
                scope='Measured 50 ms samples; manipulation only; 20 mm room margin subtracted; independent docking selections. No success-rate inference.')
    (root/'verification.json').write_text(json.dumps(report, indent=2));print(json.dumps(report,indent=2));return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);verify(p.parse_args().root)
