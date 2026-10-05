"""Replay seven-joint GCS reference trajectories in the original 3D scene."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from pydrake.geometry import Box,Rgba
from pydrake.math import RigidTransform

from .arm_cspace_research import OUTPUT,WEB,BASE
from .joint_cspace_research import make_scene,write_json
from .arm_boundary_semantics import ArmBoundaryOracle
from .build_scene import set_robot_configuration
from .collision_checker import evaluate_collision_status


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=OUTPUT)
    parser.add_argument('--web',type=Path,default=WEB)
    args=parser.parse_args()
    protocol=json.loads((args.source/'protocol.json').read_text())
    summary=json.loads((args.source/'summary.json').read_text())
    scene=make_scene(json.loads((args.source/'config.json').read_text()),visualize=True)
    oracle=ArmBoundaryOracle(scene)
    scene.meshcat.SetObject('semantic_boundary',Box(.004,2.2,2.2),Rgba(1,.12,.12,.17))
    scene.meshcat.SetTransform('semantic_boundary',RigidTransform([0,0,1.1]))
    scene.meshcat.SetCameraPose([-.6,-.2,3.8],[-.6,0,.7])
    scene.meshcat.StartRecording(frames_per_second=30.,set_visualizations_while_recording=False)
    frame=0;modes={}
    for label in ['collision_only','semantic']:
        plan_path=args.source/f'{label}_plan.json'
        regions_path=args.source/f'{label}_regions.json'
        plan=json.loads(plan_path.read_text()) if plan_path.exists() else dict(success=False,pending=True)
        records=json.loads(regions_path.read_text()) if regions_path.exists() else []
        if label not in summary:
            summary[label]=dict(pending=True,accepted=False,accepted_regions=sum(r['accepted'] for r in records),attempts=len(records))
        modes[label]=dict(summary=summary[label],regions=[{k:v for k,v in r.items() if k not in ('A','b')} for r in records],frames=[])
        if not plan.get('success') or 'trajectory_samples' not in plan:
            continue
        indices=np.unique(np.linspace(0,len(plan['sample_times'])-1,301).astype(int))
        for index in indices:
            q=np.array(plan['trajectory_samples'][index])
            c=evaluate_collision_status(scene,BASE,arm_q=q)
            r=oracle.query(np.r_[BASE,q])
            scene.root_context.SetTime(frame/30.)
            set_robot_configuration(scene,BASE,q,scene.gripper_open_positions)
            scene.diagram.ForcedPublish(scene.root_context)
            modes[label]['frames'].append(dict(q=q.tolist(),time=frame/30.,path_time=plan['sample_times'][index],
                                               semantic_residual_m=r['residual_m'],worst_body=r['worst']['body'],
                                               clearance_m=c.minimum_signed_distance,collision_free=c.collision_free))
            frame+=1
    scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
    args.web.mkdir(parents=True,exist_ok=True)
    (args.web/'scene.html').write_text(scene.meshcat.StaticHtml())
    coverage_path=args.source/'reference_coverage.json'
    coverage=json.loads(coverage_path.read_text()) if coverage_path.exists() else None
    write_json(args.web/'data.json',dict(protocol=protocol,modes=modes,coverage=coverage))
    write_json(args.web/'summary.json',summary)
    for name in ['protocol.json','semantic_regions.json','collision_only_regions.json']:
        if (args.source/name).exists():shutil.copy2(args.source/name,args.web/name)
        else:write_json(args.web/name,[])
    for name in ['reference_coverage.json','coefficient_audit.json']:
        if (args.source/name).exists():shutil.copy2(args.source/name,args.web/name)
    (args.web/'index.html').write_text(Path(__file__).with_name('arm_cspace_research_view.html').read_text())
    print(f'{frame} recorded configurations. http://localhost:7001/arm_cspace/',flush=True)


if __name__=='__main__':main()
