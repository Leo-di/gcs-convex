"""Construct a functional cafe prototype and check its initial access relations.

New doorway and static person approximations are real collision geometry.
Only an initial-state check is made: no new manipulation success is implied.
"""
import argparse,copy,json,sys
from pathlib import Path
import numpy as np
from pydrake.geometry import Box,Rgba,Sphere
from pydrake.math import RigidTransform
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT))
from src.build_scene import build_scene,set_robot_configuration
from src.human_passage import HumanPassageEvaluator
from src.scene_slice_gcs import SliceOracle

def person(name,xy,color):
    x,y=xy
    # Upright static use-site models: articulated walking is deliberately absent.
    return [dict(name=name+'_torso',shape='cylinder',radius=.18,height=.62,center=[x,y,1.12],color=color),
            dict(name=name+'_head',shape='cylinder',radius=.125,height=.23,center=[x,y,1.55],color=[.78,.60,.44,1]),
            dict(name=name+'_leg_l',shape='box',size=[.12,.16,.80],center=[x-.09,y,.40],color=[.18,.23,.30,1]),
            dict(name=name+'_leg_r',shape='box',size=[.12,.16,.80],center=[x+.09,y,.40],color=[.18,.23,.30,1])]

def build(source,out):
    out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((source/'config.json').read_text());task=json.loads((source/'protocol.json').read_text())['spec']
    cfg['scenario']['name']='cafe_service_functional_prototype'
    cfg['visualization']={'global_camera':dict(eye=[5.,-4.8,7.7],target=[1.8,1.3,.4])}
    # The front opening now has an explicit physical doorway, 1.2 m wide.
    cfg['world']['walls'] += [dict(name='entrance_left_front',size=[.9,.10,1.0],center=[-1.05,-2.35,.50]),
                             dict(name='entrance_right_front',size=[4.7,.10,1.0],center=[2.95,-2.35,.50])]
    cfg['objects']['distractors'] += person('customer',[4.9,3.05],[.12,.38,.72,1])
    cfg['objects']['distractors'] += person('staff',[2.,3.35],[.77,.36,.12,1])
    routes=[dict(id='entrance_to_customer_table',label='入口内侧 → 使用中餐桌接近区',start=[0.,-1.85],goal=[3.65,2.65],color=[.03,.65,.40,1]),
            dict(id='customer_table_to_service',label='餐桌接近区 → 回收桌外侧服务位',start=[3.65,2.65],goal=[3.05,2.45],color=[.14,.43,.78,1])]
    cfg['human_passage'].update(id=routes[0]['id'],lower_xy=[-1.35,-2.30],upper_xy=[5.15,4.9],start_xy=routes[0]['start'],goal_xy=routes[0]['goal'],enforce=True)
    # This config is a buildable prototype, not an executed task specification.
    (out/'scene_config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    scene=build_scene(config=cfg,visualize=True)
    set_robot_configuration(scene,task['start_base'],cfg['pick_task']['navigation_arm_q'],scene.gripper_open_positions)
    scene.config['human_passage']['enforce']=False
    oracle=SliceOracle(scene,cfg['pick_task']['navigation_arm_q'],scene.gripper_open_positions,max_distance=.5)
    clearance,internal,closest=oracle.query(task['start_base'][:2])
    scene.config['human_passage']['enforce']=True
    checks=[]
    for i,r in enumerate(routes):
        scene.config['human_passage'].update(id=r['id'],start_xy=r['start'],goal_xy=r['goal'])
        evaluator=HumanPassageEvaluator(scene,force=True);result=evaluator.current()
        color=Rgba(*r['color'])
        if result['satisfied']:
            for j,xy in enumerate(result['path_xy']):
                scene.meshcat.SetObject(f'functional_route/{i}/{j}',Sphere(.045),color)
                scene.meshcat.SetTransform(f'functional_route/{i}/{j}',RigidTransform([*xy,.05+i*.015]))
        for which,xy in [('start',r['start']),('goal',r['goal'])]:
            scene.meshcat.SetObject(f'functional_targets/{i}/{which}',Box(.7,.7,.008),Rgba(*r['color'][:3],.25))
            scene.meshcat.SetTransform(f'functional_targets/{i}/{which}',RigidTransform([*xy,.01+i*.01]))
        checks.append(dict(**r,initial_connected=result['satisfied'],path_xy=result['path_xy'],scope='Initial-state human-model access only; no walking or manipulation execution'))
    # Restore the saved primary rule after evaluating the secondary relation.
    scene.config['human_passage'].update(id=routes[0]['id'],start_xy=routes[0]['start'],goal_xy=routes[0]['goal'])
    scene.diagram.ForcedPublish(scene.root_context)
    for prefix in ['contact_forces','/drake/contact_forces','/meshcat/contact_forces']:scene.meshcat.Delete(prefix)
    summary=dict(status='prototype_initial_state_only',new_collision_geometry=['front doorway','static customer','static staff'],
        robot_initial_geometry_clearance_m=clearance,robot_initial_self_clearance_m=internal,closest=closest,
        checks=checks,initial_base=task['start_base'],source=str(source),
        scope='New geometry and rule targets; no full IRIS/GCS plan, pick/place execution, dynamic pedestrian, or benchmark success claimed')
    (out/'initial_check.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    statuses=' · '.join(r['label']+('：初始连通' if r['initial_connected'] else '：初始断连') for r in checks)
    banner='<div style="position:fixed;top:14px;left:18px;right:18px;padding:13px;background:#fffffff0;border-radius:9px;font:14px/1.6 system-ui;z-index:10000"><b>咖啡馆业务场景原型 · 初始状态，尚未执行清桌</b><br>前方开口为入口；蓝色人物为静态顾客，橙色为静态工作人员。绿色：入口到餐桌；蓝色：餐桌到服务位。<br>'+statuses+'<br>人物采用简化固定碰撞几何，不是行走仿真。可拖动旋转查看。</div>'
    (out/'scene.html').write_text(scene.meshcat.StaticHtml().replace('<body>','<body>'+banner,1))
    print(json.dumps(dict(initial_geometry_clearance_m=clearance,routes=[(r['id'],r['initial_connected']) for r in checks]),ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();build(a.source.resolve(),a.output.resolve())
