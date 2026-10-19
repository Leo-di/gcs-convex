"""Read-only geometric replay of recorded physical execution, with phase duties."""
import argparse,csv,html,json
from pathlib import Path
import numpy as np
from .build_scene import build_scene
from .green_room_gcs import room_oracle


def phase_summary(rows):
    groups=[]
    for row in rows:
        if not groups or groups[-1]['phase']!=row['phase']:
            groups.append({'phase':row['phase'],'start_s':row['time_s'],'end_s':row['time_s'],'samples':0,'whole_robot_required':row['phase']!='navigate_to_pickup','minimum_room_residual_m':float('inf'),'violations':0})
        g=groups[-1];g['samples']+=1;g['end_s']=row['time_s'];g['minimum_room_residual_m']=min(g['minimum_room_residual_m'],row['room_residual_m']);g['violations']+=int(row['room_residual_m']<0)
    return groups


def audit(candidate):
    candidate=Path(candidate);cfg=json.loads((candidate/'scene_config_snapshot.json').read_text());physics_path=candidate/'execution/execution_grounded_candidate/physics.json';physics=json.loads(physics_path.read_text());scene=build_scene(config=cfg,visualize=False);oracle=room_oracle(scene);rows=[]
    for frame in physics['trace']:
        q=np.r_[frame['base_q'],frame['arm_q']];r=oracle.query(q,frame['gripper_q'])
        rows.append({'time_s':frame['time_s'],'phase':frame['phase'],'room_residual_m':r['residual_m'],'base_residual_m':r['base_residual_m'],'arm_gripper_residual_m':r['arm_gripper_residual_m'],'worst_geometry':r['worst']['geometry'],'left_finger_N':frame['finger_contact_force_N']['left_finger'],'right_finger_N':frame['finger_contact_force_N']['right_finger']})
    if not rows:raise ValueError('No executed trace')
    phases=phase_summary(rows);nav=[r for r in rows if r['phase']=='navigate_to_pickup'];op=[r for r in rows if r['phase']!='navigate_to_pickup']
    if not nav or not op:raise ValueError('Expected both bridge and manipulation')
    first=next((r['time_s'] for r in rows if r['room_residual_m']>=0),None)
    result={'scope':'Read-only recomputation on recorded actual base/arm/gripper configurations; collision geometry containment in XY room prism, including margin; finite time samples, not a continuous-time proof. Navigation is allowed outside Room2.','samples':len(rows),'initial_residual_m':rows[0]['room_residual_m'],'first_whole_robot_inside_time_s':first,'last_navigation_residual_m':nav[-1]['room_residual_m'],'first_manipulation_time_s':op[0]['time_s'],'first_manipulation_residual_m':op[0]['room_residual_m'],'manipulation_minimum_room_residual_m':min(r['room_residual_m'] for r in op),'manipulation_violations':sum(r['room_residual_m']<0 for r in op),'original_execution_minimum_m':physics['green_room_task_audit']['room_min_m'],'phases':phases}
    result['matches_execution_audit']=bool(abs(result['manipulation_minimum_room_residual_m']-result['original_execution_minimum_m'])<1e-9)
    result['semantic_replay_pass']=bool(result['last_navigation_residual_m']>=0 and result['manipulation_violations']==0 and result['matches_execution_audit'])
    (candidate/'semantic_execution_replay.json').write_text(json.dumps(result,indent=2))
    with (candidate/'semantic_execution_timeline.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);args=p.parse_args();root=Path(args.root);results=[]
    for path in sorted(root.glob('*/operation_first/search_result.json')):
        search=json.loads(path.read_text())
        if not search.get('success'):continue
        candidate=path.parent/search['selected'];r=audit(candidate);r.update(task=search['task'],candidate=search['selected'],path=str(candidate.relative_to(root)));results.append(r);print(r['task'],r['semantic_replay_pass'],r['samples'],flush=True)
    (root/'semantic_execution_summary.json').write_text(json.dumps(results,indent=2));tables=[]
    for r in results:
        first_inside=f'{r["first_whole_robot_inside_time_s"]:.2f} s' if r['first_whole_robot_inside_time_s'] is not None else '记录中从未整机入室'
        rows=''.join(f'<tr><td>{html.escape(g["phase"])}</td><td>{g["start_s"]:.2f}–{g["end_s"]:.2f}</td><td>{g["whole_robot_required"]}</td><td>{100*g["minimum_room_residual_m"]:.3f}</td><td>{g["violations"]}</td></tr>' for g in r['phases'])
        tables.append(f'<h2>{r["task"]} / {r["candidate"]}</h2><p>整机首次满足入室余量：{first_inside}；开始操作：{r["first_manipulation_time_s"]:.2f} s；独立重算通过：{r["semantic_replay_pass"]}。</p><table><tr><th>阶段</th><th>时间 s</th><th>必须整机在 Room2</th><th>最小余量 cm（已扣 2 cm）</th><th>不满足的样本数</th></tr>{rows}</table><p><a href="{r["path"]}/semantic_execution_timeline.csv">逐帧 CSV</a> · <a href="{r["path"]}/semantic_execution_replay.json">检查结果</a></p>')
    (root/'semantic_review.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>实际执行的阶段语义</title><style>body{max-width:1200px;margin:40px auto;font:17px system-ui}table{width:100%;border-collapse:collapse}td,th{padding:12px;border-bottom:1px solid #ddd}</style><h1>教授的两阶段语义：实际执行逐帧检查</h1><p>whole-robot-inside 是导航终点条件；whole-motion-inside 是操作阶段的全程条件。导航中的负余量是允许的，不能误判为任务失败。原日志 whole_body_carry_to_dropoff 在本实验表示原地保持，不是运输。</p>'+''.join(tables)+'<p>全部是数值采样；没有声称连续时间形式化证明。这里复用同一几何 oracle 独立重算日志，不是独立软件实现。</p><a href="index.html">返回动画与完整搜索结果</a></html>')

if __name__=='__main__':main()
