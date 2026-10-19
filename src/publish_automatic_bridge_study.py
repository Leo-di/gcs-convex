"""Publish actual automatic-repair outcomes without inferring infeasibility."""
import argparse,html,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    protocol=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'comparison.json').read_text())
    methods=protocol['methods'];summary=[];details=[];animations=[]
    for method in methods:
        subset=[r for r in rows if r['method']==method];good=[r for r in subset if r['verified_task_success']]
        ordered=sorted(good,key=lambda r:next(i for i,c in enumerate(protocol['candidates']) if c['id']==r['id']))
        summary.append({'method':method,'runs':len(subset),'bridge_pass':sum(r.get('bridge',{}).get('accepted',False) for r in subset),'operation_pass':sum(r.get('operation',{}).get('accepted',False) for r in subset),'complete_pass':len(good),'ordered_first_success':ordered[0]['id'] if ordered else None,'sum_candidate_wall_seconds':sum(r['total_wall_seconds'] for r in subset)})
    for r in sorted(rows,key=lambda r:(r['id'],r['method'])):
        sub=root/r['method']/r['id'];file=sub/'bridge_events.json';events=json.loads(file.read_text()) if file.exists() else {};trace=events.get('events',[])
        repaired=any(e['event']=='repaired_bridge_passed' for e in trace)
        failure=r.get('failure_reason') or r.get('bridge',{}).get('reason','')
        cost=(r['bridge']['path_length_m']+.25*r['operation']['path_length_rad']) if r['verified_task_success'] else None
        details.append({'method':r['method'],'id':r['id'],'repaired_bridge':repaired,'bridge':r.get('bridge',{}).get('accepted',False),'operation':r.get('operation',{}).get('accepted',False),'complete':r['verified_task_success'],'cost':cost,'failure':failure,'seconds':r['total_wall_seconds'],'events_path':str(file.relative_to(root)),'physical_attempted':'method' in r.get('physical',{})})
        scene=sub/'execution/execution_grounded_candidate/scene.html'
        if scene.exists():animations.append((f"{r['method']} / {r['id']}",str(scene.relative_to(root))))
    bridge_plot=''
    repair_file=root/'repair_gcs/candidate_606/bridge_events.json'
    if repair_file.exists():
        bridge=json.loads(repair_file.read_text())
        if bridge.get('accepted'):
            from .scene_spec import load_scene_config
            config=load_scene_config(protocol['config']);px=lambda x:60+(float(x)+4.5)*150;py=lambda y:400-float(y)*150
            pieces=['<svg viewBox="0 -100 1200 1000" style="width:100%;max-height:650px" role="img" aria-label="候选606导航修复轨迹俯视图"><rect y="-100" width="1200" height="1000" fill="#f7fafc"/>']
            projected=[w for w in config['world']['walls'] if w['center'][2]-w['size'][2]/2<.5]
            projected.extend(o for o in config['objects'].values() if isinstance(o,dict) and 'center' in o and 'size' in o)
            for box in projected:
                x,y,_=box['center'];w,h,_=box['size'];pieces.append(f'<rect x="{px(x-w/2)}" y="{py(y+h/2)}" width="{w*150}" height="{h*150}" fill="#8a98a6"/>')
            for points,color in [(bridge['guide'],'#ed9d20'),(bridge['path'],'#2764d8')]:
                coords=' '.join(f'{px(q[0])},{py(q[1])}' for q in points)
                pieces.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="4"/>')
            for event in bridge['events']:
                if event['event']=='coverage_gap_seeds':
                    for q in event['seeds']:pieces.append(f'<circle cx="{px(q[0])}" cy="{py(q[1])}" r="5" fill="#d13c3c"/>')
            pieces.append('</svg>');bridge_plot='<section><h2>候选 606 的导航覆盖修复</h2><p>橙色：OMPL 引导；蓝色：补覆盖后的 GCS 轨迹；红点：补充种子。这是底盘中心的俯视投影；高于底盘的门楣未画成封闭门洞。家具绘制外形投影，实际检查使用固定姿态下的完整机器人几何。该候选的抓取 IK 未通过。</p>'+''.join(pieces)+'</section>'
    result={'summary':summary,'details':details,'scope':protocol['scope'],'complete':len(rows)==len(methods)*len(protocol['candidates'])}
    paired={}
    for method in ['repair_gcs','repair_ompl']:
        candidate=root/method/'candidate_606'
        if (candidate/'candidate_result.json').exists() and (candidate/'bridge_events.json').exists():
            row=json.loads((candidate/'candidate_result.json').read_text());event=json.loads((candidate/'bridge_events.json').read_text())
            if row.get('bridge',{}).get('accepted'):
                paired[method]={'guide':event['guide'],'path_length_m':row['bridge']['path_length_m'],'seconds':row['bridge']['seconds'],'minimum_distance_m':row['bridge']['report']['minimum_distance_m']}
    if len(paired)==2:
        result['candidate_606_bridge_comparison']={'same_ompl_guide':paired['repair_gcs']['guide']==paired['repair_ompl']['guide'],'methods':paired,'scope':'Navigation-only comparison on one previously studied candidate; grasp fails for both.'}
    (root/'study_summary.json').write_text(json.dumps(result,indent=2))
    paired_html=''
    pair=result.get('candidate_606_bridge_comparison')
    if pair:
        entries=''.join(f'<tr><td>{name}</td><td>{v["path_length_m"]:.3f}</td><td>{v["seconds"]:.2f}</td><td>{v["minimum_distance_m"]*100:.2f}</td></tr>' for name,v in pair['methods'].items())
        paired_html=f'<section><h2>606：相同引导下的导航比较</h2><table><tr><th>方法</th><th>长度 m</th><th>阶段耗时 s</th><th>最小采样间隙 cm</th></tr>{entries}</table><p>两方法的 OMPL 引导坐标完全一致。本例 GCS 路径更长、更慢，采样间隙较大；没有证明 GCS 的成功率、效率或安全优势。两个方法都未完成该候选的抓取。</p></section>'
    table=''.join(f'<tr><td>{s["method"]}</td><td>{s["runs"]}</td><td>{s["bridge_pass"]}</td><td>{s["operation_pass"]}</td><td>{s["complete_pass"]}</td><td>{s["sum_candidate_wall_seconds"]:.1f}</td></tr>' for s in summary)
    lines=''.join(f'<tr><td>{d["id"]}</td><td>{d["method"]}</td><td>{d["bridge"]} / {d["repaired_bridge"]}</td><td>{d["operation"]}</td><td>{d["complete"]}</td><td>{d["seconds"]:.1f}</td><td>{html.escape(d["failure"])}<br><a href="{d["events_path"]}">诊断事件</a></td></tr>' for d in details)
    live_html=''
    live=root.parent/'automatic_grounding_search_20261019'
    if (live/'search_result.json').exists():
        search=json.loads((live/'search_result.json').read_text());sequence=' → '.join(html.escape(x['id'])+('（完整通过）' if x['complete'] else '（未通过，继续）') for x in search['attempts'])
        live_html=f'<section><h2>真实顺序搜索控制器</h2><p>{sequence}</p><p>已用 {search["elapsed_seconds"]:.1f} 秒，总预算 600 秒。这是另外一次真实运行；采用已知候选的距离降序来检查失败回退，不属于独立测试。</p><p><a href="../automatic_grounding_search_20261019/search_result.json">实际搜索记录</a></p></section>'
        if search['success']:
            candidate=search['selected'];animations.insert(0,(f'自动搜索选择：{candidate}',f'../automatic_grounding_search_20261019/{candidate}/execution/execution_grounded_candidate/scene.html'))
    buttons=''.join(f'<button data-src="{url}">{label}</button>' for label,url in animations)
    viewer=f'<iframe id="viewer" src="{animations[0][1]}"></iframe>' if animations else '<p>暂无完成的执行动画。</p>'
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>自动 motion bridge 修复对照</title><style>body{{max-width:1400px;margin:30px auto;padding:20px;font:16px system-ui;color:#18344a;background:#f4f7fa}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}iframe{{width:100%;height:700px;border:0}}button{{padding:9px;margin:4px}}.note{{background:#fff0cc;padding:15px}}</style><h1>自动诊断 → 局部补覆盖 → GCS → 操作与物理执行</h1><p>已完成 {len(rows)}/24 次固定候选实验。每候选 600 秒总预算；场景、候选、语义及审计标准相同。</p><p class="note">这是同一场景八个已研究候选的开发对照，不是独立测试集。失败不等于全局不可行。修复只针对导航覆盖缺口；操作规划、严格审计和物理执行仍需分别通过。</p><section><h2>三组方法</h2><p>fixed_gcs：原覆盖。repair_gcs：原覆盖失败时，调用官方 OMPL 引导、只在覆盖缺口加种子，再调用官方 GCS。repair_ompl：相同触发条件与 OMPL 引导，直接使用简化后的 OMPL 导航路径；后续机械臂仍使用相同 IRIS/GCS。</p><p>OMPL 和 GCS 使用相同 2.01 cm 导航余量。本轮区别于上一轮 5 cm 引导诊断，方案在本轮运行前冻结。并非全任务 GCS 与全任务 OMPL 的对比。</p><table><tr><th>方法</th><th>完成运行</th><th>导航通过</th><th>操作通过</th><th>完整任务通过</th><th>候选耗时之和 s</th></tr>{table}</table><p>时间为各候选墙钟时间之和，并非并行实验总耗时。按原顺序遇到首个成功就停止的选择结果仅可由该表离线重放；全候选评估不冒充在线早停。</p></section><section><h2>全部结果与失败记录</h2><table><tr><th>候选</th><th>方法</th><th>导航 / 已修复</th><th>操作</th><th>完整成功</th><th>秒</th><th>说明</th></tr>{lines}</table><p>前序未通过的候选没有物理执行；完整成功 false 不等于发生物理碰撞。</p></section>{paired_html}{bridge_plot}{live_html}<section><h2>实际执行动画</h2>{buttons}{viewer}</section><p><a href="protocol.json">冻结方案</a> · <a href="study_summary.json">汇总</a> · <a href="comparison.json">原始结果</a> · <a href="source_manifest.json">源代码快照</a></p><script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.getElementById('viewer').src=b.dataset.src)</script></html>'''
    (root/'index.html').write_text(page);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
