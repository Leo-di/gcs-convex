import json,copy
from pathlib import Path
import numpy as np
from src.build_scene import build_scene
from src.animate_delivery_demo import plan_delivery_candidate
from src.audit_replay import audit_replay
root=Path('outputs/results/semantic_candidate_diagnosis');rows=[]
for name in ['nominal','cup_left','cup_right','cart_near_route']:
 lib=root/'inner_margin/libraries'/name;c=json.loads((lib/'config.json').read_text())
 for old in json.loads((lib/'report.json').read_text())['attempts']:
  if old['status']=='accepted' or old['stage'] not in ['bootstrap_carry_gcs','carrying_iris_cover']:continue
  row={'scene':name,'name':old['name']}
  try:
   data=plan_delivery_candidate(build_scene(config=copy.deepcopy(c),visualize=False),np.array(old['base_q']),row)
   row.update(status='accepted',dense=audit_replay(data['trace'],c,10))
  except Exception as e:row.update(status='rejected',reason=str(e),details=getattr(e,'details',{}))
  rows.append(row);(root/'transport_endpoint_retest.json').write_text(json.dumps(rows,indent=2));print(name,old['name'],row['status'],row.get('reason',''),flush=True)
