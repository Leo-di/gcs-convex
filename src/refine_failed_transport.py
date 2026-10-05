"""Controlled refinement of the two remaining candidates; original guards retained."""
import json,copy,argparse
from pathlib import Path
import numpy as np
from .build_scene import build_scene
from .animate_delivery_demo import plan_delivery_candidate
from .audit_replay import audit_replay
from .iris_region_generator import cover_to_dict

def main():
 p=argparse.ArgumentParser();p.add_argument('case',choices=['left_stronger','right_cartesian','right_original','control_cartesian','left_cuts','right_pregrasp','control_cuts']);a=p.parse_args()
 name={'left_stronger':'cup_left','right_cartesian':'cup_right','right_original':'cup_right','control_cartesian':'nominal','left_cuts':'cup_left','right_pregrasp':'cup_right','control_cuts':'nominal'}[a.case]
 c=json.load(open(f'outputs/results/semantic_multiscene_v1/libraries/{name}/config.json'))
 if a.case=='left_stronger':c['iris_np2_cover'].update(particles=880,ray_particles=240,max_iterations=6,epsilon=.005,holdout_random_seed=19200,holdout_samples_per_region=500)
 if 'cartesian' in a.case:c['pick_task']['cartesian_approach']=True
 if a.case in ['left_cuts','control_cuts']:c['iris_np2_cover']['counterexample_refinement']=True
 if a.case=='right_pregrasp':c['pick_task']['collision_checked_pregrasp']=True
 q=np.array([1.2 if a.case=='control_cuts' else 1.15,-.1 if a.case in ['control_cartesian','control_cuts'] else .05,0])
 out=Path('outputs/results/remaining_collision_refinement')/a.case;out.mkdir(parents=True,exist_ok=True)
 (out/'scene_config_snapshot.json').write_text(json.dumps(c,indent=2));record={'case':a.case,'base_q':q.tolist()}
 try:
  d=plan_delivery_candidate(build_scene(config=copy.deepcopy(c),visualize=False),q,record)
  record['dense']=audit_replay(d['trace'],c,10);record['accepted']=record['dense']['satisfied']
  (out/'delivery_trace.json').write_text(json.dumps(d['trace']));(out/'cover.json').write_text(json.dumps(cover_to_dict(d['carrying_iris_cover'])))
 except Exception as e:record.update(accepted=False,reason=str(e),details=getattr(e,'details',{}))
 (out/'result.json').write_text(json.dumps(record,indent=2));print(a.case,record['accepted'],record.get('reason',''),flush=True)
if __name__=='__main__':main()
