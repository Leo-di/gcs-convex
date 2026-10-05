"""Export synchronized original-scene FK snapshots and research planner results."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from pydrake.geometry import Box, Rgba
from pydrake.math import RigidTransform

from .arm_boundary_semantics import ArmBoundaryOracle
from .build_scene import set_robot_configuration
from .collision_checker import evaluate_collision_status
from .joint_cspace_research import OUTPUT, WEB, make_scene, state, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=OUTPUT)
    parser.add_argument('--web', type=Path, default=WEB)
    args = parser.parse_args()
    protocol = json.loads((args.source/'protocol.json').read_text())
    summary = json.loads((args.source/'summary.json').read_text())
    scene = make_scene(json.loads((args.source/'config.json').read_text()), visualize=True)
    oracle = ArmBoundaryOracle(scene)
    scene.meshcat.SetObject('semantic_boundary', Box(.004, 2.2, 2.2), Rgba(1,.12,.12,.17))
    scene.meshcat.SetTransform('semantic_boundary', RigidTransform([0,0,1.1]))
    scene.meshcat.SetCameraPose([-.6,-.2,5.0],[-.6,0,.7])
    scene.meshcat.StartRecording(frames_per_second=30., set_visualizations_while_recording=False)
    frames = []

    def snapshot(q):
        s = state(scene,q)
        collision = evaluate_collision_status(scene,s[:3],arm_q=s[3:])
        predicate = oracle.query(s)
        index = len(frames)
        scene.root_context.SetTime(index/30.)
        set_robot_configuration(scene,s[:3],s[3:],scene.gripper_open_positions)
        scene.diagram.ForcedPublish(scene.root_context)
        frames.append(dict(q=np.asarray(q).tolist(), state=s.tolist(), time=index/30.,
                           collision_free=collision.collision_free,
                           clearance_m=collision.minimum_signed_distance,
                           semantic_residual_m=predicate['residual_m'],worst_body=predicate['worst']['body']))
        return index

    n = 41
    x = np.linspace(protocol['lower'][0],protocol['upper'][0],n)
    y = np.linspace(protocol['lower'][1],protocol['upper'][1],n)
    for yq in y:
        for xq in x:
            snapshot([xq,yq])
    modes = {}
    for label in ('collision_only','semantic'):
        plan = json.loads((args.source/f'{label}_plan.json').read_text())
        rows = json.loads((args.source/f'{label}_regions.json').read_text())
        modes[label] = dict(polygons=[r['polygon'] for r in rows if r['accepted']],frames=[])
        if plan['success']:
            indices = np.unique(np.linspace(0,len(plan['sample_times'])-1,241).astype(int))
            modes[label]['frames'] = [snapshot(plan['trajectory_samples'][i]) for i in indices]
            modes[label]['path_times'] = [plan['sample_times'][i] for i in indices]
        modes[label]['summary'] = summary[label]
    scene.meshcat.StopRecording()
    scene.meshcat.PublishRecording()
    args.web.mkdir(parents=True,exist_ok=True)
    (args.web/'scene.html').write_text(scene.meshcat.StaticHtml())
    data = dict(protocol=protocol,grid=dict(n=n,x=x.tolist(),y=y.tolist()),frames=frames,modes=modes)
    write_json(args.web/'data.json',data)
    shutil.copy2(args.source/'summary.json',args.web/'summary.json')
    shutil.copy2(args.source/'protocol.json',args.web/'protocol.json')
    template = Path(__file__).with_name('joint_cspace_research_view.html').read_text()
    (args.web/'index.html').write_text(template)
    print(f'{len(frames)} synchronized snapshots. http://localhost:7001/joint_cspace/',flush=True)


if __name__ == '__main__':
    main()
