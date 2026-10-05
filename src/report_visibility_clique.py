"""Publish actual coverage comparison and main-query 3D recordings."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from pydrake.geometry import Box,Rgba
from pydrake.math import RigidTransform
from .joint_cspace_research import make_scene,write_json,BASE
from .build_scene import set_robot_configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--web',type=Path,default=Path('outputs/results/grasp_fix_web/planar/visibility_clique'))
    parser.add_argument('--template',type=Path,default=Path(__file__).with_name('visibility_clique_view.html'))
    parser.add_argument('--query-index',type=int,default=0)
    args=parser.parse_args();source=args.source;web=args.web;web.mkdir(parents=True,exist_ok=True)
    summary=json.loads((source/'summary.json').read_text())
    scene=make_scene(json.loads((source/'config.json').read_text()),visualize=True)
    scene.meshcat.SetObject('semantic_boundary',Box(.004,2.2,2.2),Rgba(1,.12,.12,.17))
    scene.meshcat.SetTransform('semantic_boundary',RigidTransform([0,0,1.1]))
    scene.meshcat.SetCameraPose([-.6,-.2,3.8],[-.6,0,.7])
    scene.meshcat.StartRecording(frames_per_second=30,set_visualizations_while_recording=False)
    frame=0;recordings={}
    for method in ['uniform','vcc']:
        plan=json.loads((source/f'{method}_query_{args.query_index:02}.json').read_text());recordings[method]=[]
        if not plan.get('accepted'):continue
        for index in np.unique(np.linspace(0,len(plan['sample_times'])-1,301).astype(int)):
            q=plan['trajectory_samples'][index];time=frame/30.
            scene.root_context.SetTime(time);set_robot_configuration(scene,BASE,q,scene.gripper_open_positions)
            scene.diagram.ForcedPublish(scene.root_context)
            recordings[method].append(dict(time=time,path_time=plan['sample_times'][index],q=q));frame+=1
    scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
    (web/'scene.html').write_text(scene.meshcat.StaticHtml())
    write_json(web/'data.json',dict(summary=summary,recordings=recordings,display_query_index=args.query_index,cliques=json.loads((source/'cliques.json').read_text()),
        baseline=json.loads((source/'continuation_provenance.json').read_text()) if (source/'continuation_provenance.json').exists() else None,
        history=json.loads((source/'history.json').read_text()) if (source/'history.json').exists() else None))
    for file in source.glob('*.json'):shutil.copy2(file,web/file.name)
    if (source/'semantic_counterexample.html').exists():
        shutil.copy2(source/'semantic_counterexample.html',web/'semantic_counterexample.html')
    if (source/'additional_queries'/'run_status.json').exists():
        extra=web/'additional_queries';extra.mkdir(exist_ok=True)
        for file in (source/'additional_queries').glob('*.json'):
            shutil.copy2(file,extra/file.name)
    (web/'index.html').write_text(args.template.read_text())
    print(f'{frame} frames exported to {web}')

if __name__=='__main__':main()
