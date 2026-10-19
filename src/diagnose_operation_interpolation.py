"""Diagnostic straight segments only; never used as planning/execution fallback."""
import argparse,json
from pathlib import Path
import numpy as np
from .scene_spec import load_scene_config
from .semantic_refinement import make_candidate_config
from .build_scene import build_scene
from .green_room_gcs import room_oracle
from .joint_grounding_checks import check_pick_endpoints
from .collision_checker import evaluate_collision_status


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root);protocol=json.loads((root/'protocol.json').read_text());candidate=protocol['candidates'][0];rows=[]
    tasks=[('no_screen_control',Path('configs/scene_semantic_choice_v6.yaml'))]+[(t['id'],root/'configs'/f"{t['id']}.json") for t in protocol['tasks']]
    for task,path in tasks:
        scene=build_scene(config=make_candidate_config(load_scene_config(path),candidate),visualize=False);oracle=room_oracle(scene);base=np.array(candidate['base_q']);row={'task':task,'candidate':candidate,'endpoint_pass':False}
        try:
            pick,goals=check_pick_endpoints(scene,base,oracle);row.update(endpoint_pass=True,goals=goals,edges=[])
            for name,start,end in [('tucked_to_pregrasp',np.array(scene.config['pick_task']['navigation_arm_q']),pick.pregrasp_q),('pregrasp_to_grasp',pick.pregrasp_q,pick.grasp_q)]:
                n=max(2,int(np.ceil(np.max(abs(end-start))/.00125))+1);minimum=1.;bad=0;sem_bad=0;witness=None
                for t in np.linspace(0,1,n):
                    q=(1-t)*start+t*end;inside=oracle.query(np.r_[base,q]);status=evaluate_collision_status(scene,base,arm_q=q,gripper_q=scene.gripper_open_positions,influence_distance=.2)
                    bad+=int(status.minimum_signed_distance < -1e-7);sem_bad+=int(not inside['satisfied'])
                    if status.minimum_signed_distance<minimum:
                        minimum=status.minimum_signed_distance;witness={'alpha':float(t),'arm_q':q.tolist(),'geometry_a':status.closest_geometry_a,'geometry_b':status.closest_geometry_b}
                row['edges'].append({'name':name,'samples':n,'collision_samples':bad,'semantic_violation_samples':sem_bad,'minimum_distance_m':minimum,'worst_witness':witness})
        except Exception as e:row['reason']=str(e)
        rows.append(row);(root/'interpolation_diagnostic.json').write_text(json.dumps({'scope':'Additional diagnostic only; not a planner baseline or execution path. A colliding straight segment does not imply no alternative path exists. No payload lift interpolation tested.','rows':rows},indent=2));print(task,row['endpoint_pass'],[(r['name'],r['collision_samples']) for r in row.get('edges',[])],flush=True)

if __name__=='__main__':main()
