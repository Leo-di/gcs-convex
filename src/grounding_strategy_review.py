"""Review outer candidate selection, without claiming joint GCS optimization."""
import argparse,html,json
from pathlib import Path


def strategy_results(rows, weight=.25):
    def score(r):return r['bridge']['path_length_m']+weight*r['operation']['path_length_rad']
    first=rows[:1];back=[]
    for r in rows:
        back.append(r)
        if r['verified_task_success']:break
    results=[]
    for method,seen in [('nearest_only',first),('ordered_backtracking',back),('minimum_verified_total_cost',rows)]:
        feasible=[r for r in seen if r['verified_task_success']]
        selected=min(feasible,key=lambda r:(score(r),r['id'])) if feasible else None
        results.append({'method':method,'selected':selected['id'] if selected else None,'success':selected is not None,
                        'score':score(selected) if selected else None,'candidate_evaluations':len(seen),
                        'candidate_seconds_sum':sum(r['elapsed_seconds'] for r in seen)})
    return results


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    protocol=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'comparison.json').read_text())
    if len(rows)!=len(protocol['candidates']):raise RuntimeError('Incomplete fixed candidate experiment')
    diagnosis={r['id']:r for r in json.loads((root/'endpoint_clearance_diagnosis.json').read_text())}
    methods=strategy_results(rows,protocol['cost_arm_weight_m_per_rad'])
    summary={'methods':methods,'counts':{'semantic_valid':sum(r['semantic_goal']['satisfied'] for r in rows),'bridge_valid':sum(r['bridge']['accepted'] for r in rows),'operation_valid':sum(r['operation']['accepted'] for r in rows),'end_to_end_valid':sum(r['verified_task_success'] for r in rows)},'scope':protocol['scope'],'selection_evaluation':'Offline replay of fixed candidate outcomes; computation sums are not measured parallel wall time.'}
    (root/'strategy_results.json').write_text(json.dumps(summary,indent=2))
    from .scene_spec import load_scene_config
    config=load_scene_config(protocol['config'])
    px=lambda x:70+float(x)*470
    py=lambda y:480-float(y)*300
    layout=['<svg viewBox="0 0 900 620" role="img" aria-label="桌子杯子及八个底盘候选的俯视投影"><rect x="70" y="50" width="800" height="550" fill="#edf8ec"/><text x="80" y="30">Room 2 门内局部俯视图</text>']
    for wall in config['world']['walls']:
        x,y,_=wall['center'];w,h,_=wall['size']
        if x+w/2<-.1 or x-w/2>1.7 or y+h/2<-.4 or y-h/2>1.4:continue
        layout.append(f'<rect x="{px(x-w/2)}" y="{py(y+h/2)}" width="{w*470}" height="{h*300}" fill="#788493"/>')
    table_config=config['objects']['table'];x,y,_=table_config['center'];w,h,_=table_config['size']
    layout.append(f'<rect x="{px(x-w/2)}" y="{py(y+h/2)}" width="{w*470}" height="{h*300}" fill="#bb875c" opacity=".75"/>')
    cup=config['objects']['cup']['center'];layout.append(f'<circle cx="{px(cup[0])}" cy="{py(cup[1])}" r="9" fill="#008ccc"/><text x="{px(cup[0])+15}" y="{py(cup[1])}">杯子</text>')
    for i,r in enumerate(rows):
        x,y=r['base_q'][:2];color='#22834d' if r['verified_task_success'] else '#bf3b3b'
        layout.append(f'<circle cx="{px(x)}" cy="{py(y)}" r="8" fill="{color}"/><text x="{px(x)+10}" y="{py(y)-10}" font-size="15">{i+1}</text>')
    layout.append('<text x="85" y="570" font-size="15">数字对应下表顺序；圆点仅表示底盘候选位置，验证使用完整机器人几何。</text></svg>')
    labels={'nearest_only':'最近合法候选，仅一次','ordered_backtracking':'按距离排序，失败回溯','minimum_verified_total_cost':'评估全部候选，选最小验证代价'}
    strategies=''.join(f'<tr><td>{labels[m["method"]]}</td><td>{m["selected"] or "无"}</td><td>{m["candidate_evaluations"]}</td><td>{m["score"] if m["score"] is not None else "失败"}</td><td>{m["candidate_seconds_sum"]:.1f}</td></tr>' for m in methods)
    table=[];graph=['<svg viewBox="0 0 900 640" role="img" aria-label="各候选构型的入室与操作验证"><text x="20" y="25">初始配置</text><text x="290" y="25">整机入室构型 qᵢ</text><text x="650" y="25">各自的抓取构型 gᵢ</text><circle cx="70" cy="315" r="15" fill="#245fd0"/>']
    animations=[]
    for i,r in enumerate(rows):
        d=diagnosis[r['id']];reason=r.get('failure_reason') or r['bridge'].get('reason','')
        if d['distance_m']<.0201:reason='终点间隙低于导航要求；不是断连证明。 '+reason
        y=65+i*73;bc='#22834d' if r['bridge']['accepted'] else '#bf3b3b';oc='#22834d' if r['operation']['accepted'] else '#bf3b3b'
        graph.append(f'<path d="M85 315 L310 {y}" stroke="{bc}" fill="none"/><path d="M330 {y} L680 {y}" stroke="{oc}"/><circle cx="320" cy="{y}" r="10" fill="#607e9e"/><circle cx="690" cy="{y}" r="10" fill="{oc}"/><text x="340" y="{y-12}" font-size="14">{r["id"]}</text>')
        table.append(f'<tr><td>{r["id"]}<br>{r["base_q"][:2]}</td><td>{r["semantic_goal"]["satisfied"]}</td><td>{d["distance_m"]:.3f}</td><td>{r["bridge"]["accepted"]}</td><td>{r["operation"]["accepted"]}</td><td>{r["verified_task_success"]}</td><td>{html.escape(reason)}</td></tr>')
        scene=root/r['id']/'execution/execution_grounded_candidate/scene.html'
        if scene.exists():animations.append(f'<button data-url="{scene.relative_to(root)}">{r["id"]}</button>')
    graph.append('</svg>')
    repair_html=''
    repair_file=root/'bridge_repair_606_uncovered/result.json'
    if repair_file.exists():
        repair=json.loads(repair_file.read_text())
        accepted=repair.get('gcs_bridge_accepted','尚未完成')
        repair_html=f'<section><h2>独立诊断：候选 606 的覆盖断连</h2><p>冻结原场景、姿态与目标。OMPL 状态：{html.escape(repair["ompl_status"])}，采用 5 cm 间隙；原 GCS 要求为 2.01 cm。仅在引导路径未被原区域覆盖的位置补充 {repair.get("extra_seeds",0)} 个种子后，GCS 入室连接通过：{accepted}。</p><p>全路径补充 250 个种子的对照保留 399 个区域，达到 600 秒预算后终止（实际检查时 609 秒），未返回 GCS 解。该诊断不改写上方八候选原始结果。路径探索来自 OMPL，不能归功于 GCS 单独发现。候选 606 的操作 IK 仍未通过，所以不能记为完整任务成功。</p><p><a href="bridge_repair_606_uncovered/result.json">局部补覆盖结果与路径</a> · <a href="bridge_repair_606/result.json">全路径补种子对照</a></p></section>'
    chosen=next((m['selected'] for m in methods[::-1] if m['selected']),None)
    src=f'{chosen}/execution/execution_grounded_candidate/scene.html' if chosen else ''
    c=summary['counts']
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>语义构型与双侧连续连接</title><style>body{{font:17px system-ui;background:#f4f7fa;color:#18344a;max-width:1350px;margin:30px auto;padding:20px}}section{{background:white;padding:24px;border-radius:12px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}.note{{background:#fff1cd;padding:18px}}a{{color:#1760a0}}svg{{width:100%;max-height:640px}}iframe{{width:100%;height:700px;border:0}}button{{padding:10px;margin:5px}}</style><h1>语义目标是否能接上前后运动？</h1><p>固定两房间场景、杯子、机器人和抓取姿态。从 73 个预筛选候选中按固定排序等间隔抽取 8 个，不根据结果替换。每个候选独立运行官方 Drake 导航 GCS、操作 IRIS-NP2/GCS、严格轨迹审计及适用时的真实抓取执行。</p><p class="note">整机入室：{c['semantic_valid']}/8；入室连接通过：{c['bridge_valid']}/8；操作规划与审计通过：{c['operation_valid']}/8；完整任务执行通过：{c['end_to_end_valid']}/8。这是一个场景的候选可行性表，不是独立场景成功率。</p><section><h2>谓词与连接条件</h2><p>whole-robot-inside：底盘、机械臂和夹爪的碰撞体全部位于 Room 2 的 XY 棱柱内，保留 2 cm 房间余量。whole-motion-inside：操作全过程保持该条件，另检查杯子。导航时固定收臂姿态与底盘朝向；操作时底盘固定。</p><p>导航还要求环境间隙至少 2.01 cm。只满足房间语义且不穿透，未必满足这个导航终点条件。因此“端点未被导航覆盖”不能统一解释为真实 C-space 不连通。</p></section><section><h2>候选构型的空间位置</h2>{''.join(layout)}</section><section><h2>双侧连接验证图</h2><p>绿色：对应阶段通过；红色：本轮未通过。右侧是各候选自己的抓取构型，并非同一个 C-space 点。本图展示验证关系，不是声称已经构造联合任务 GCS。</p>{''.join(graph)}</section><section><h2>全部候选</h2><table><tr><th>候选 / 坐标</th><th>整机入室</th><th>终点几何间隙 m</th><th>入室连接</th><th>操作规划</th><th>完整任务通过</th><th>失败解释</th></tr>{''.join(table)}</table></section><section><h2>三种选择策略</h2><p>统一代价 = 导航长度（m） + 0.25 × 接近和抬升的关节路径长度（rad）。所有方法使用相同候选，每候选相同 600 秒上限，最多评估 8 个。以下按真实候选结果离线重放选择策略，未重新计为独立实验。</p><table><tr><th>策略</th><th>选择</th><th>评估候选数</th><th>验证总代价</th><th>候选耗时之和 s</th></tr>{strategies}</table><p>耗时之和不是实际并行墙钟时间。“最小验证代价”只是在这八个已评估候选中选择；不是全局最优，也不是把候选选择联合放进 GCS。</p></section>{repair_html}<section><h2>完整实际执行动画</h2>{''.join(animations)}<iframe id="viewer" src="{src}" title="候选完整执行"></iframe></section><p class="note">失败按具体求解器、覆盖、审计或执行阶段解释；有限预算失败不证明物理不可行。只有前序规划和审计全部通过的候选才进入物理执行；其余候选完整任务为 false，不表示已发生物理执行失败。本轮没有自动生成缺失语义，也没有宣称 GCS 成功率优势。碰撞与语义审计采用有限数值采样，非连续时间证书。</p><p><a href="protocol.json">固定实验方案</a> · <a href="comparison.json">全部候选原始结果</a> · <a href="strategy_results.json">策略比较</a> · <a href="endpoint_clearance_diagnosis.json">终点间隙诊断</a> · <a href="source_manifest.json">源代码快照</a></p><script>document.querySelectorAll('button[data-url]').forEach(b=>b.onclick=()=>document.getElementById('viewer').src=b.dataset.url);</script></html>'''
    (root/'index.html').write_text(text);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
