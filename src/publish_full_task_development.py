"""Publish every development outcome and sampled actual execution replays."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from pydrake.math import RigidTransform
from .build_scene import build_scene, set_robot_configuration
from .benchmark_full_task_development import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('outputs/results/full_task_development_v1'))
    p.add_argument('--web',type=Path,default=Path('outputs/results/grasp_fix_web/planar/full_task_baselines'))
    p.add_argument('--render',action='store_true');args=p.parse_args();out=args.web;out.mkdir(parents=True,exist_ok=True)
    data=summarize(args.source);data['protocol']=json.loads((args.source/'protocol.json').read_text())
    data['complete']=(args.source/'complete.json').exists()
    data['cases']=json.loads((args.source/'development_cases.json').read_text())
    data['preparation']={}
    for cell in data['cases']:
        bundle=json.loads((args.source/'inputs'/(cell['id']+'.json')).read_text())
        data['preparation'][cell['id']]={k:bundle[k] for k in ['preparation_seconds','primitive_rejections','known_colliding_cached_regions_removed']}
    for record in data['records']:
        directory=args.source/record['cell']/record['method'];dest=out/record['cell']/record['method'];dest.mkdir(parents=True,exist_ok=True)
        shutil.copy2(directory/'record.json',dest/'record.json')
        physical=directory/'physics.json';record['replay']=False
        if not physical.exists():continue
        raw=json.loads(physical.read_text());record['physics_metrics']={k:v for k,v in raw.items() if k not in ('trace','unexpected_contacts')}
        (dest/'trace.json').write_text(json.dumps([{'time_s':r['time_s'],'phase':r['phase']} for r in raw['trace']]))
        shutil.copy2(physical,dest/'physics.json')
        if args.render and not (dest/'scene.html').exists():
            config=json.loads((Path(record['reference'])/'scene_config_snapshot.json').read_text())
            scene=build_scene(config=config,visualize=True);scene.meshcat.StartRecording()
            chosen=raw['trace'][::3]
            if chosen[-1] is not raw['trace'][-1]:chosen.append(raw['trace'][-1])
            for row in chosen:
                set_robot_configuration(scene,row['base_q'],row['arm_q'],row['gripper_q'])
                scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.asarray(row['cup_pose'])))
                scene.root_context.SetTime(row['time_s']);scene.diagram.ForcedPublish(scene.root_context)
            scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
            (dest/'scene.html').write_text(scene.meshcat.StaticHtml())
            del scene
            print('REPLAY',record['cell'],record['method'],flush=True)
        record['replay']=(dest/'scene.html').exists()
    (out/'summary.json').write_text(json.dumps(data))
    (out/'index.html').write_text(Path('src/full_task_development_view.html').read_text())
    print(out/'index.html')


if __name__=='__main__':main()
