"""Frozen non-rigid cafe layout and cup-position variations.

Recipes are independent of planner outcomes. The existing scene and manual
control remain unchanged; each new case receives the same three methods.
"""
import copy
import numpy as np
from .transport_posture_search import DEFAULT_POLICY

METHODS=('automatic','manual','automatic_geometry')
RECIPES=[
 dict(id='staggered_tables',label='错列桌位与前后错放杯子',
      tables={'table':[.65,.80],'customer_b':[3.55,-.95],'customer_c':[.70,3.55],
              'customer_d':[3.35,3.80],'dropoff_table':[2.20,2.30]},island=[2.00,-.75],
      cup_offsets=[[-.12,.05],[.10,.10],[-.10,.12]],chairs='translated'),
 dict(id='shifted_island_chairs',label='岛台左移与非对称椅位',
      tables={'table':[.35,.90],'customer_b':[3.35,-.95],'customer_c':[.35,3.80],
              'customer_d':[3.65,3.55],'dropoff_table':[2.00,2.45]},island=[1.65,-.85],
      cup_offsets=[[.10,.18],[-.12,.18],[.12,.10]],chairs='asymmetric'),
 dict(id='cups_and_return',label='回收台右移与横向换杯位',
      tables={'table':[.60,.75],'customer_b':[3.50,-.95],'customer_c':[.45,3.65],
              'customer_d':[3.65,3.80],'dropoff_table':[2.30,2.30]},island=[2.10,-.78],
      cup_offsets=[[.18,.18],[-.18,.10],[.18,.10]],chairs='offset'),
]


def make_case(scene,manual,recipe,scene_path):
    scene=copy.deepcopy(scene);task=copy.deepcopy(manual)
    scene['scenario']['name']='replicacad_cafe_'+recipe['id']
    scene['scenario']['instruction']='Clear three customer tables while preserving the fixed front human passage.'
    centers={key:[*xy,.375] for key,xy in recipe['tables'].items()}
    for key,center in centers.items():
        support=scene['objects'][key] if key in ('table','dropoff_table') else scene['support_surfaces'][key]
        support['center']=center
    # Preserve publisher mesh geometry, scales, and floor/room/person model.
    for fixture in scene['asset_fixtures']:
        if fixture['name'] in centers:fixture['center']=copy.deepcopy(centers[fixture['name']])
        elif fixture['name']=='service_island':fixture['center']=[*recipe['island'],.45]
    chair_supports={0:('table',-1.05),1:('table',.85),4:('customer_c',-1.05),5:('customer_c',.85),6:('customer_d',-.85),7:('customer_d',.85)}
    for fixture in scene['asset_fixtures']:
        if not fixture['name'].startswith('cafe_chair_'):continue
        i=int(fixture['name'].rsplit('_',1)[1])
        if i in chair_supports:
            key,dx=chair_supports[i];fixture['center']=[centers[key][0]+dx,centers[key][1],.44]
        if recipe['chairs']=='asymmetric':
            if i in chair_supports:
                key,dx=chair_supports[i];side=1 if dx>0 else -1
                fixture['center']=[centers[key][0]+side*(.95 if side>0 or key=='customer_d' else 1.10),centers[key][1],.44]
                fixture['rpy']=[0,0,side*(1.35 if key=='table' else 1.75)]
            if i==2:fixture.update(center=[4.55,1.70,.44],rpy=[0,0,-.9])
            if i==3:fixture.update(center=[4.55,2.55,.44],rpy=[0,0,1.2])
        elif recipe['chairs']=='offset':
            if i==0:fixture['center']=[-.50,.85,.44]
            if i==1:fixture.update(center=[1.55,.85,.44],rpy=[0,0,1.35])
            if i==2:fixture['center']=[4.65,1.65,.44]
            if i==3:fixture['center']=[4.65,2.65,.44]
            if i in (6,7):
                side=-1 if i==6 else 1
                fixture.update(center=[centers['customer_d'][0]+side*.95,centers['customer_d'][1],.44],rpy=[0,0,side*1.3])
    task['scene']=str(scene_path)
    task['destination_table']=copy.deepcopy(scene['objects']['dropoff_table'])
    task['support_surfaces']=copy.deepcopy(scene['support_surfaces'])
    for key,support,offset,action in zip(['cup','cup_b','cup_c'],['table','customer_b','customer_c'],recipe['cup_offsets'],task['actions']):
        c=centers[support];task['objects'][key]=[c[0]+offset[0],c[1]+offset[1],.84]
        # Same support-relative docking rule in every case; no posture hints.
        base=[task['objects'][key][0]-.03,c[1]-.70,0.]
        action['pickup_base']=base
        action['pickup_candidates']=[dict(base_q=base.copy(),grasp_rpy=task['grasp_rpy'].copy())]
    x,y=centers['dropoff_table'][:2]
    task['actions'][0]['placements']=[[x+.05,y-.05,.84],[x-.25,y-.05,.84]]
    task['actions'][1]['placements']=[[x+.05,y-.05,.84]]
    task['actions'][2]['placements']=[[x+.35,y-.05,.84]]
    tasks={'manual':task}
    auto=copy.deepcopy(task)
    for action in auto['actions']:action.pop('transport_offset_m',None)
    auto['transport_posture_search']=copy.deepcopy(DEFAULT_POLICY)
    tasks['automatic']=auto
    geometry=copy.deepcopy(auto);geometry['human_passage']['enforce']=False;tasks['automatic_geometry']=geometry
    return scene,{name:tasks[name] for name in METHODS}


def normalized_task(task):
    task=copy.deepcopy(task);task.pop('transport_posture_search',None)
    for action in task['actions']:action.pop('transport_offset_m',None)
    task['human_passage']['enforce']=True
    return task


def assert_consistency(scene,tasks):
    canonical=normalized_task(tasks['manual'])
    assert all(normalized_task(t)==canonical for t in tasks.values())
    fixtures={f['name']:f for f in scene['asset_fixtures']}
    for key,support in {'table':scene['objects']['table'],'dropoff_table':scene['objects']['dropoff_table'],**scene['support_surfaces']}.items():
        assert np.allclose(support['center'],fixtures[key]['center'])
    for action in canonical['actions']:
        table=scene['objects'].get(action['source_support'],scene['support_surfaces'].get(action['source_support']))
        p=np.array(canonical['objects'][action['object_id']]);c=np.array(table['center']);size=np.array(table['size'])
        assert np.all(np.abs(p[:2]-c[:2])+.05<size[:2]/2)
        assert abs(p[2]-.09-(c[2]+size[2]/2))<1e-8
        assert np.allclose(np.array(action['pickup_base'])[:2],[p[0]-.03,c[1]-.70])
    assert all('transport_offset_m' not in a for a in tasks['automatic']['actions'])
    assert tasks['manual']['actions'][1]['transport_offset_m']==[0.,-.35,0.]
