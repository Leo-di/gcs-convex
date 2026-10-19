"""Publish SE(2) capability results without claiming statistical superiority."""
import argparse, html, json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    protocol=json.loads((root/'protocol.json').read_text());notice='<p style="background:#ffe2c0;padding:20px">此批已停止：旧导航 yaw 域与新候选不兼容。仅保留为诊断，不计为能力实验结果。</p>' if (root/'interrupted.json').exists() else '';rows=[];attempts=[];views=[]
    for task in protocol['tasks']:
        folder=root/task['id']/'operation_first';screen=json.loads((folder/'candidate_screen.json').read_text()) if (folder/'candidate_screen.json').exists() else []
        counts={str(y):sum(c['status']=='eligible' and c['base_q'][2]==y for c in screen) for y in [-.2,0.,.2]}
        result=json.loads((folder/'search_result.json').read_text()) if (folder/'search_result.json').exists() else {}
        row={'task':task['id'],'eligible_by_yaw':counts,'result':result,'finished':'total_wall_seconds' in result}
        if not row['finished']: result['success']=None
        for path in sorted(folder.glob('*/candidate_result.json')):
            d=json.loads(path.read_text());attempts.append({'task':task['id'],'id':path.parent.name,'base_q':d.get('base_q'),'stage':d.get('failure_stage',d.get('stage')),'reason':d.get('failure_reason') or d.get('bridge',{}).get('reason') or ('candidate time budget exhausted' if d.get('process_exit')=='timeout' else None),'physical_success':d.get('physical',{}).get('accepted',False),'process_exit':d.get('process_exit'),'record':str(path.relative_to(root))})
        if result.get('success'):
            selected=folder/result['selected'];d=json.loads((selected/'candidate_result.json').read_text());trace=json.loads((selected/'delivery_trace.json').read_text());nav=[x for x in trace if x['phase']=='navigate_to_pickup'];op=[x for x in trace if x['phase']!='navigate_to_pickup']
            row['checks']={'selected_base_q':d['base_q'],'nav_arm_max_change_rad':float(np.max(np.abs(np.array([x['arm_q'] for x in nav])-nav[0]['arm_q']))),'operation_base_max_change':float(np.max(np.abs(np.array([x['base_q'] for x in op])-op[0]['base_q']))),'bridge_to_operation_position_gap':float(np.max(np.abs(np.r_[nav[-1]['base_q'],nav[-1]['arm_q']]-np.r_[op[0]['base_q'],op[0]['arm_q']]))),'navigation_yaw_range_rad':[min(x['base_q'][2] for x in nav),max(x['base_q'][2] for x in nav)],'physical':d['physical'],'bridge_report':d['bridge'].get('report')}
            scene=selected/'execution/execution_grounded_candidate/scene.html';views.append((task['id'],str(scene.relative_to(root))))
        rows.append(row)
    navigation_description=('导航采用 OMPL 引导、基于刚体旋转位移上界的三维椭球凸区域及官方 Drake GCS；保留所生成覆盖的全部重叠边。该引导覆盖不保证全局完备性。' if protocol.get('bridge_method')=='rotation_safe_gcs' else '导航调用官方 Drake 三维 GCS；导航覆盖采用距离圆盘 × yaw 区间，尚无整个区域旋转安全证明。')
    summary={'tasks':rows,'attempts':attempts,'scope':protocol['scope']};(root/'review_summary.json').write_text(json.dumps(summary,indent=2))
    bridge_metrics=''
    if protocol.get('bridge_method')=='rotation_safe_gcs':
        metric_rows=[]
        for row in rows:
            checks=row.get('checks',{});b=checks.get('bridge_report',{});physical=checks.get('physical',{})
            if b:metric_rows.append(f'<tr><td>{row["task"]}</td><td>{b["region_count"]}</td><td>{b["edge_count"]}</td><td>{b["solve_seconds"]:.3f}</td><td>{100*b["minimum_distance_m"]:.2f}</td><td>{physical.get("unexpected_contacts")}</td></tr>')
        bridge_metrics='<h2>通过验收的执行结果</h2><table><tr><th>候选</th><th>区域</th><th>有向边</th><th>GCS 求解秒数</th><th>导航间隙 cm</th><th>非预期接触</th></tr>'+''.join(metric_rows)+'</table><p>本地开发场景验证，不代表泛化成功率。安全界仅适用于固定收臂、固定夹爪和静态世界；物理执行另行验收。</p>'
    examples=''
    example_file=root/'room_grounding_examples.json'
    if example_file.exists():
        example_data=json.loads(example_file.read_text())
        examples='<h2>同一底盘位置，朝向改变会改变整机语义</h2><table><tr><th>构型例子</th><th>房间 residual（含 2 cm 余量）</th><th>整机入室</th></tr>'+''.join(f'<tr><td>{html.escape(k)}</td><td>{v["residual_m"]:.5f} m</td><td>{v["satisfied"]}</td></tr>' for k,v in example_data.items())+'</table><p>nearest 为 [0.45,0]，farther 为 [0.65,0]；negative_yaw 为 −0.2 rad。这些只是构型检查，不代表可执行轨迹。</p>'
    semantic_link='<p><a href="semantic_review.html">实际执行的阶段语义：逐帧重算与余量表</a></p>' if (root/'semantic_review.html').exists() else ''
    table=''.join(f"<tr><td>{r['task']}</td><td>{html.escape(str(r['eligible_by_yaw']))}</td><td>{r['result'].get('success','运行中')}</td><td>{r.get('checks',{}).get('selected_base_q','—')}</td><td>{r['result'].get('total_wall_seconds','—')}</td></tr>" for r in rows)
    detail=''.join(f"<tr><td>{x['task']}/{x['id']}</td><td>{x['base_q']}</td><td>{x['stage']}</td><td>{html.escape(str(x['reason']))}</td><td>{x['process_exit']}</td><td><a href='{x['record']}'>原始记录</a></td></tr>" for x in attempts)
    buttons=''.join(f'<button data-src="{path}">{name} 实际执行</button>' for name,path in views)
    iframe=f'<iframe id="viewer" src="{views[0][1]}"></iframe>' if views else '<p>尚无通过完整物理验收的动画；不会把参考轨迹冒充执行结果。</p>'
    (root/'index.html').write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>底盘位置与朝向联合 Grounding</title><style>body{{max-width:1400px;margin:30px auto;font:17px system-ui;padding:20px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #ddd}}iframe{{width:100%;height:760px;border:0}}button{{padding:12px;margin:5px}}</style><h1>按教授的两阶段任务：底盘位置、朝向与抓取 Grounding</h1>{notice}<p>初始构型 → 导航 motion bridge（机械臂固定）→ 整机进入 Room 2 → 操作轨迹（底盘固定，整机始终在 Room 2）→ 双指抓取抬升。</p><p>协议候选集合共 {len(protocol["candidates"])} 个，底盘 yaw 集合 {sorted(set(c["base_q"][2] for c in protocol["candidates"]))} rad。固定旧布局；每场 {protocol["total_search_seconds"]} 秒、每候选 {protocol["per_candidate_seconds"]} 秒。{navigation_description}操作沿用 IRIS-NP2/GCS。仍需数值审计和物理验收。</p><p>初始状态不需要满足 Room 2 的包含条件；该条件在导航终点和整个操作阶段成立。没有把导航阶段错误限制在 Room 2 内。</p><table><tr><th>场景</th><th>各 yaw 整机目标初筛通过数</th><th>完整成功</th><th>选中底盘 [x,y,yaw]</th><th>搜索秒数</th></tr>{table}</table>{bridge_metrics}{examples}{semantic_link}<h2>所有尝试（保留拒绝及超时）</h2><table><tr><th>候选</th><th>底盘</th><th>阶段</th><th>原因</th><th>退出</th><th>证据</th></tr>{detail}</table><h2>真实接触执行动画</h2><p>播放：Open Controls → Animations → default → play；可拖动 time 回看抓取阶段。</p>{buttons}{iframe}<p>此实验检查能力边界，不与旧实验混算成功率或加速比。成功范围为抓取抬升，不是运回或放箱。连续时间安全与全局不可行均未获证明。</p><p><a href="review_summary.json">轨迹交接、固定关节与执行验收</a> · <a href="protocol.json">冻结协议</a> · <a href="source_manifest.json">运行源码</a> · <a href="../operation_obstacles_20261019/">前一轮对照与动画</a></p><script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.getElementById('viewer').src=b.dataset.src)</script></html>''')
    print(json.dumps([{'task':r['task'],'success':r['result'].get('success'),'checks':r.get('checks',{}).get('selected_base_q')} for r in rows]))

if __name__=='__main__':main()
