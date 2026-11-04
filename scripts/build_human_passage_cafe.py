"""Build a separate two-route cafe; preserve the earlier fixed-aisle scenes."""
import copy
from pathlib import Path
import sys
import yaml
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from src.scene_spec import load_scene_config
from src.replicacad import prepare_furniture

def build():
    scene=load_scene_config(PROJECT/'configs/scene_replicacad_cafe.yaml')
    scene['scenario']['name']='replicacad_cafe_human_access'
    scene['scenario']['instruction']='Clear customer tables while preserving an upright human route from the entrance to the exit; the route may change.'
    scene['world']['floor']['size']=[7.,8.,.1]
    scene['robot']['lower_limits'][1]=-2.3
    scene['green_room']['y_bounds'][0]=-2.3
    scene['green_room']['navigation_guide']='fixed_yaw_grid'
    scene['semantic_regions']['room2']['y_bounds'][0]=-2.3
    scene['gcs_base_planning']['regions'][0]['lower'][1]=-2.3
    scene['support_surfaces']['customer_b']['center']=[3.5,-.95,.375]
    for f in scene['asset_fixtures']:
        if f['name']=='customer_b':f['center']=[3.5,-.95,.375]
        if f['name']=='cafe_chair_2':f['center']=[4.6,1.8,.44]
        if f['name']=='cafe_chair_3':f['center']=[4.6,2.6,.44]
    asset=PROJECT/'data/replicacad_drake/human_passage/service_island'
    prepare_furniture(PROJECT/'data/replicacad','frl_apartment_table_01',asset,size=[1.6,.4,.9])
    scene['asset_fixtures'].append(dict(name='service_island',center=[1.9,-.75,.45],asset_file=str((asset/'asset.json').relative_to(PROJECT))))
    scene['visualization']['global_camera']=dict(eye=[8.,-9.,9.],target=[1.8,1.,.3])
    task=load_scene_config(PROJECT/'configs/replicacad_cafe_transport.yaml')
    task['scene']='configs/scene_replicacad_human_passage.yaml'
    task['objects']['cup_b']=[3.48,-.8,.84]
    task['support_surfaces']=copy.deepcopy(scene['support_surfaces'])
    task['robot_keep_clear']=[];task['robot_keep_clear_display']=[]
    task['object_exclusions']=[];task['object_exclusion_display']=[]
    task['carry']['lower_xy']=[-1.,-2.2]
    for i,a in enumerate(task['actions']):
        if i==1:
            a['pickup_base']=[3.45,-1.65,0.]
            a['transport_offset_m']=[0.,-.35,0.]
        base=a['pickup_base']
        a['pickup_candidates']=[dict(base_q=base.copy(),grasp_rpy=task['grasp_rpy'])]
    task['human_passage']=dict(id='entrance_to_exit',lower_xy=[-1.35,-2.1],upper_xy=[5.2,.25],start_xy=[0.,-.75],goal_xy=[4.75,-.75],radius_m=.28,height_m=1.7,margin_m=.02,resolution_m=.08,enforce=True)
    (PROJECT/task['scene']).write_text(yaml.safe_dump(scene,sort_keys=False))
    (PROJECT/'configs/replicacad_human_passage_transport.yaml').write_text(yaml.safe_dump(task,sort_keys=False))
    print('Built independent two-route cafe and three-table task')
if __name__=='__main__':build()
