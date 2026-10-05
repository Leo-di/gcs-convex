"""Prespecified unseen combinations; no rejection sampling after outcomes."""
import copy,json
from pathlib import Path
import numpy as np

def design():
    base=json.loads(Path('outputs/results/semantic_candidates/library/config.json').read_text())
    base['pick_task'].update(collision_checked_pregrasp=True,cartesian_approach=True)
    base['iris_np2_cover']['counterexample_refinement']=True
    rng=np.random.default_rng(20261008);scenes={};changes={};cases=[]
    for i in range(16):
        name=f'heldout_{i:02d}';c=copy.deepcopy(base)
        width=float(rng.uniform(1.00,1.20));cy=float(rng.uniform(-.85,.95));dx=float(rng.uniform(-.07,.07));cup=rng.uniform([1.53,.64],[1.67,.70]).tolist()
        for w in c['world']['walls']:
            if w['name'] in ['shared_wall_upper','shared_wall_lower']:
                sign=1 if w['name'].endswith('upper') else -1;w['size'][1]=3-width/2;w['center'][1]=sign*(3+width/2)/2
            elif w['name']=='doorway_header':w['size'][1]=width
        c['objects']['mobile_cart']['center']=[-1.25,cy,.41];c['objects']['cup']['center'][:2]=cup
        for obj in ['dropoff_table','dropoff_bin']:c['objects'][obj]['center'][0]+=dx
        c['task_zones']['dropoff']['center'][0]+=dx;c['delivery_task']['place_cup_center'][0]+=dx
        for k in ['iris_np2_cover','iris_np2_doorway']:
            c[k]['random_seed']=120000+i*1000;c[k]['holdout_random_seed']=220000+i*1000
        scenes[name]=c;changes[name]={'door_width':width,'cart_y':cy,'cup_xy':cup,'dropoff_dx':dx,'contact_query':i%3}
        for j in range(3):
            cases.append({'id':f'{name}__q{j}','scene':name,'start':[*rng.uniform([-3.15,-.8],[-2.75,.8]).tolist(),0.],
              'sites':None if j==0 else [{'name':['south','north'][j-1],'xy':[-2.5,[-1.,1.][j-1]]}],
              'contact':j==i%3,'allowed':['x110','x115','x120','x125'],
              'docking':{'name':'shared_goal_domain','lower':[-2.6,1.0],'upper':[-2.1,1.5]}})
    return {'scenes':scenes,'changes':changes,'cases':cases,'pickup_x':[1.10,1.15,1.20,1.25],
      'goal_x':[-2.50,-2.35,-2.20],'goal_y':1.25,'goal_heights':[.14,.22],
      'seed':20261008,'use_known_witnesses':False,
      'scope':'16 prespecified unseen random combinations of the same two-room template; 48 correlated queries, not 48 independent scenes. Same algorithm and probe budgets, no old witnesses, no test-query feedback. Official GCS/IRIS shared bootstrap remains. Raw planning-only; screened 16 prespecified contact opportunities per method including inspection tasks. Not broad environment generalization.'}
