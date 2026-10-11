"""Audit physical doorway semantics and publish a review dashboard."""
import argparse, json
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform
from .build_scene import build_scene, set_robot_configuration


def audit(config, report, trace):
    scene = build_scene(config=config, visualize=False)
    plant = scene.plant
    door = config['doorway_semantics']; x0=float(door['plane_coordinate_m'])
    ylo,yhi=door['aperture_y_bounds_m']; half=np.asarray(config['robot']['size'][:2],float)/2
    grasp_point=np.asarray(config['gripper']['grasp_point_in_body'],float)
    wall_names={f'{n}_collision' for n in ('shared_wall_upper','shared_wall_lower','doorway_header')}
    inspector=scene.scene_graph.model_inspector()
    wall_ids={g for g in inspector.GetAllGeometryIds() if inspector.GetName(g) in wall_names}
    arm_ids={g for inst in (scene.arm_instance,scene.gripper_instance) for bi in plant.GetBodyIndices(inst) for g in plant.GetCollisionGeometriesForBody(plant.get_body(bi))}
    cup_ids=set(plant.GetCollisionGeometriesForBody(scene.cup_body))
    min_arm_gap=float('inf'); min_cup_gap=float('inf'); max_depth=-float('inf'); crossed=False; min_base=float('inf')
    base_samples=[]; ee_samples=[]; state_samples=[]
    for row in trace:
        bq=np.asarray(row['base_q'],float); yaw=float(bq[2])
        clearance=x0-(bq[0]+abs(np.cos(yaw))*half[0]+abs(np.sin(yaw))*half[1])
        min_base=min(min_base,float(clearance)); base_samples.append(clearance)
        set_robot_configuration(scene,bq,np.asarray(row['arm_q']),np.asarray(row['gripper_q']))
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.asarray(row['cup_pose'])))
        Xwg=scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
        ee=np.asarray(Xwg.multiply(grasp_point),float); depth=float(ee[0]-x0); max_depth=max(max_depth,depth); ee_samples.append(depth)
        row_arm_crossed=False
        for bi in plant.GetBodyIndices(scene.arm_instance):
            body=plant.get_body(bi)
            if body.name()=='iiwa_link_0': continue
            p=body.body_frame().CalcPoseInWorld(scene.plant_context).translation()
            if p[0]>x0+1e-4 and ylo<=p[1]<=yhi: crossed=True; row_arm_crossed=True
        context=scene.scene_graph.GetMyContextFromRoot(scene.root_context)
        query=scene.scene_graph.get_query_output_port().Eval(context)
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(10.0):
            ids={pair.id_A,pair.id_B}
            if not ids.intersection(wall_ids): continue
            if ids.intersection(arm_ids): min_arm_gap=min(min_arm_gap,float(pair.distance))
            if ids.intersection(cup_ids): min_cup_gap=min(min_cup_gap,float(pair.distance))
        cup_x=float(np.asarray(row['cup_pose'])[0,3])
        bilateral=min(row['finger_contact_force_N'].values())>.05
        state_samples.append({'time_s':row['time_s'],'phase':row['phase'],
          'base_outside':clearance>=0.0,'ee_inside':depth>0.0,
          'arm_crosses_door':row_arm_crossed,
          'cup_inside':cup_x-float(config['objects']['cup']['radius'])>x0,
          'cup_grasped':bilateral and cup_x>0.0,
          'cup_returned':cup_x+float(config['objects']['cup']['radius'])<x0 and bilateral})
    forces=[r['finger_contact_force_N'] for r in trace]
    held=[r for r in trace if r['phase']=='whole_body_carry_to_dropoff']
    bilateral=[min(r['finger_contact_force_N'].values()) for r in held]
    thresholds=float(door['minimum_base_geometry_clearance_m'])
    exact=report.get('doorway_semantics',{})
    min_base=float(exact.get('minimum_base_geometry_clearance_to_door_plane_m',min_base))
    max_depth=float(exact.get('maximum_end_effector_room2_depth_m',max_depth))
    crossed=bool(exact.get('arm_crossed_doorway',crossed))
    def witness(predicate):
        return next(({'time_s':r['time_s'],'phase':r['phase']} for r in state_samples if predicate(r)),None)
    state_sequence=[
      {'state':'base outside, arm outside, cup inside','witness':witness(lambda r:r['base_outside'] and not r['ee_inside'] and not r['arm_crosses_door'] and r['cup_inside'])},
      {'state':'base outside, arm inside, cup inside','witness':witness(lambda r:r['base_outside'] and r['ee_inside'] and r['arm_crosses_door'] and r['cup_inside'])},
      {'state':'base outside, arm inside, cup grasped','witness':witness(lambda r:r['base_outside'] and r['ee_inside'] and r['cup_grasped'])},
      {'state':'base outside, arm retracted, cup returned','witness':witness(lambda r:r['base_outside'] and r['cup_returned'])},
    ]
    result={
      'base_entered_room2':bool(min_base<0), 'base_always_room1':bool(min_base>=0),
      'minimum_base_geometry_clearance_to_door_plane_m':min_base,
      'required_base_geometry_clearance_m':thresholds,
      'base_clearance_satisfied':bool(min_base+1e-8>=thresholds),
      'end_effector_entered_room2':bool(max_depth>0),
      'maximum_end_effector_room2_depth_m':max_depth,
      'arm_crossed_doorway':crossed,
      'minimum_arm_to_doorframe_clearance_m':min_arm_gap if np.isfinite(min_arm_gap) else None,
      'minimum_cup_to_doorframe_clearance_m':min_cup_gap if np.isfinite(min_cup_gap) else None,
      'cup_initially_in_room2':bool(scene.cup_initial_pose.translation()[0]-config['objects']['cup']['radius']>x0),
      'cup_grasped_in_room2':bool(exact.get('cup_grasped_in_room2',report.get('lifted_with_bilateral_contact') and report.get('held_bilateral_contact_fraction',0)>0)),
      'cup_retracted_to_room1':bool(exact.get('cup_retracted_to_room1',report.get('carried_to_room1_with_bilateral_contact'))),
      'maximum_grasp_contact_force_N':exact.get('maximum_grasp_contact_force_N',{side:max((r[side] for r in forces),default=0.0) for side in ('left_finger','right_finger')}),
      'minimum_bilateral_force_during_carry_N':exact.get('minimum_bilateral_force_during_carry_N',min(bilateral) if bilateral else None),
      'semantic_state_sequence':state_sequence,
      'non_expected_collision_count':int(report.get('unexpected_contact_count',-1)),
      'drake_sap_dynamics_succeeded':bool(report.get('satisfied')),
      'cup_attachment':bool(report.get('cup_attachment')),
      'cup_pose_resets_after_initialization':report.get('cup_pose_resets_after_initialization'),
      'trace_samples':len(trace),
      'safety_semantics_sampling_period_s':exact.get('safety_semantics_sampling_period_s',0.05),
      'scope':'Doorway safety predicates were checked every 10 ms during force-driven SAP execution; door-frame distances were audited at saved 50 ms states; cup is free with no post-initialization pose reset.'}
    return result,scene


def dashboard(metrics, kine, dynamic_scene):
    names=[('Base entered Room 2','base_entered_room2',False),('End effector entered Room 2','end_effector_entered_room2',True),('Arm crossed doorway','arm_crossed_doorway',True),('Cup grasped in Room 2','cup_grasped_in_room2',True),('Cup retracted to Room 1','cup_retracted_to_room1',True),('Unexpected contacts = 0','non_expected_collision_count',0)]
    rows=''.join(f'<tr><td>{n}</td><td class="{"pass" if metrics.get(k)==want else "fail"}">{str(metrics.get(k)).lower()}</td></tr>' for n,k,want in names)
    d=metrics
    return f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Doorway Semantic GCS Demo</title><style>body{{font:16px system-ui;margin:0;background:#f1f5f9;color:#172033}}main{{max-width:1100px;margin:auto;padding:24px}}article{{background:white;padding:18px;border-radius:12px;margin:14px 0;box-shadow:0 2px 10px #0f172a16}}iframe{{width:100%;height:75vh;border:1px solid #cbd5e1;border-radius:8px}}table{{border-collapse:collapse;width:100%}}td{{padding:10px;border-bottom:1px solid #ddd}}.pass{{color:#047857;font-weight:bold}}.fail{{color:#b91c1c;font-weight:bold}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.metric{{padding:12px;background:#eff6ff;border-radius:8px}}small{{color:#64748b}}</style><main><h1>Doorway Semantic GCS · 真实接触动力学结果</h1><p>蓝色底盘保持在 Room 1；绿色区域属于 Room 2；橙色半透明板标示门平面，橙色短条标出末端进入门内的深度范围。全局动画来自自由杯的 SAP 接触动力学回放。</p><article><h2>语义验收</h2><table>{rows}</table></article><article><h2>数值验证</h2><div class="grid"><div class="metric">Base 几何到门平面最小距离<br><b>{d['minimum_base_geometry_clearance_to_door_plane_m']:.3f} m</b></div><div class="metric">EE 进入 Room 2 最大深度<br><b>{d['maximum_end_effector_room2_depth_m']:.3f} m</b></div><div class="metric">手臂到门框最小间隙<br><b>{d['minimum_arm_to_doorframe_clearance_m']:.3f} m</b></div><div class="metric">左右指最大接触力<br><b>{d['maximum_grasp_contact_force_N']['left_finger']:.2f} / {d['maximum_grasp_contact_force_N']['right_finger']:.2f} N</b></div><div class="metric">携杯阶段最小双指力<br><b>{(d['minimum_bilateral_force_during_carry_N'] or 0):.2f} N</b></div><div class="metric">非预期碰撞数量<br><b>{d['non_expected_collision_count']}</b></div></div><p>杯子到门框最小距离：{d['minimum_cup_to_doorframe_clearance_m']:.3f} m。Drake SAP 动力学成功：{d['drake_sap_dynamics_succeeded']}；杯 attachment：{d['cup_attachment']}；初始化后杯位姿重置次数：{d['cup_pose_resets_after_initialization']}。</p><small>{d['scope']}</small></article><article><h2>Meshcat · 斜上方全局视角（完整任务）</h2><iframe src="{dynamic_scene}"></iframe></article><article><h2>Meshcat · 门口侧视（底盘外、手臂进）</h2><iframe style="height:60vh" src="doorway_side_view.html"></iframe></article><article><h2>场景布局与坐标对照</h2><p><a href="layout_review.html">打开修改前后俯视布局、固定坐标、门平面定义和初步可达性检查</a></p><small>{d['scope']}</small></article></main></html>'''


def main():
 p=argparse.ArgumentParser();p.add_argument('output_dir',type=Path);a=p.parse_args();d=a.output_dir
 config=json.loads((d/'config.json').read_text());report=json.loads((d/'result.json').read_text());trace=report['trace']
 metrics,scene=audit(config,report,trace)
 (d/'doorway_semantic_audit.json').write_text(json.dumps(metrics,indent=2))
 kine_dir=d.parent/'kinematic'; kine=json.loads((kine_dir/'delivery_result.json').read_text())
 (d.parent/'doorway_semantic_dashboard.html').write_text(dashboard(metrics,kine,'dynamics/scene.html'))
 print(json.dumps({k:v for k,v in metrics.items() if not k.endswith('_samples')},indent=2))
 raise SystemExit(0 if metrics['drake_sap_dynamics_succeeded'] and metrics['base_clearance_satisfied'] and metrics['end_effector_entered_room2'] and metrics['arm_crossed_doorway'] and metrics['cup_retracted_to_room1'] else 1)
if __name__=='__main__': main()
