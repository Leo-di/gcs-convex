"""Publish complete paired results and replays of actual contact execution."""
import json
from pathlib import Path
import shutil
import numpy as np
from pydrake.math import RigidTransform
from .build_scene import build_scene,set_robot_configuration
from .analyze_unified_delivery import REG,NEW


def main():
    out=Path('outputs/results/grasp_fix_web/planar/unified_delivery');out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for label,root in [('regression',REG),('fresh',NEW)]:
        for record in json.loads((root/'summary.json').read_text())['records']:
            directory=root/record['cell']/record['method'];key=label+'_'+record['cell']+'_'+record['method']
            dest=out/key;dest.mkdir(exist_ok=True);shutil.copy2(directory/'record.json',dest/'record.json')
            row={'key':key,'set':label,'cell':record['cell'],'method':record['method'],
                 'success':record['end_to_end_success'],'planning_success':record['planning_success'],
                 'failure':record.get('failure'),'height':record.get('accepted_height',.22) if record['planning_success'] else None,
                 'bridge':record.get('accepted_bridge_stage',False),'seconds':record.get('planning_wall_seconds'),
                 'candidate':record.get('selected_candidate'),'replay':False}
            if (directory/'physics.json').exists():
                raw=json.loads((directory/'physics.json').read_text());shutil.copy2(directory/'physics.json',dest/'physics.json')
                row['physics']={k:raw[k] for k in ['complete_task_satisfied','unexpected_contact_count','held_bilateral_contact_fraction','actual_target_xy_error_m','duration_s']}
                (dest/'trace.json').write_text(json.dumps([{'time':r['time_s'],'phase':r['phase']} for r in raw['trace']]))
                if not (dest/'scene.html').exists():
                    config=json.loads((Path(record['reference'])/'scene_config_snapshot.json').read_text())
                    scene=build_scene(config=config,visualize=True);scene.meshcat.StartRecording()
                    for r in raw['trace'][::3]+[raw['trace'][-1]]:
                        set_robot_configuration(scene,r['base_q'],r['arm_q'],r['gripper_q'])
                        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(r['cup_pose'])))
                        scene.root_context.SetTime(r['time_s']);scene.diagram.ForcedPublish(scene.root_context)
                    scene.meshcat.StopRecording();scene.meshcat.PublishRecording();(dest/'scene.html').write_text(scene.meshcat.StaticHtml());del scene
                row['replay']=True
            rows.append(row);print('PUBLISHED',key,flush=True)
    (out/'data.json').write_text(json.dumps({'rows':rows,'analysis':json.loads((NEW/'analysis.json').read_text())}))
    (out/'index.html').write_text(Path('src/unified_delivery_view.html').read_text())
    print(out/'index.html')


if __name__=='__main__':main()
