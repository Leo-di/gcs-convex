"""Publish a frozen local task cohort, including all unsuccessful attempts."""
import argparse,html,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);protocol=json.loads((root/'protocol.json').read_text())
    rows=json.loads((root/'comparison.json').read_text()) if (root/'comparison.json').exists() else [];details=[];animations=[]
    for task in protocol['tasks']:
        for method in protocol['methods']:
            folder=root/task['id']/method
            for path in sorted(folder.glob('*/candidate_result.json')):
                r=json.loads(path.read_text());pre=r.get('joint_endpoint_check',{});row={'task':task['id'],'method':method,'candidate':r['id'],'base_q':r['base_q'],'precheck':pre.get('accepted'),'navigation_attempted':(path.parent/'bridge_events.json').exists(),'bridge_pass':r.get('bridge',{}).get('accepted',False),'physical_pass':r.get('physical',{}).get('accepted',False),'stage':r.get('stage'),'failure':r.get('failure_reason') or r.get('bridge',{}).get('reason','')};details.append(row)
    for row in rows:
        if row['success']:
            rel=f"{row['task']}/{row['method']}/{row['selected']}/execution/execution_grounded_candidate/scene.html";animations.append((f"{row['task']} / {row['method']}",rel))
    summary=[]
    for method in protocol['methods']:
        group=[r for r in rows if r['method']==method];attempts=[r for r in details if r['method']==method]
        summary.append({'method':method,'completed_searches':len(group),'successes':sum(r['success'] for r in group),'wall_seconds_sum':sum(r['total_wall_seconds'] for r in group),'attempted_candidates':len(attempts),'precheck_rejections':sum(r['precheck'] is False for r in attempts),'navigation_calls':sum(r['navigation_attempted'] for r in attempts)})
    diagnostic_html=''
    diagnostic_file=root/'endpoint_diagnostic.json'
    if diagnostic_file.exists():
        diagnostic=json.loads(diagnostic_file.read_text());entries=[]
        for task in protocol['tasks']:
            group=[r for r in diagnostic['rows'] if r['task']==task['id']];accepted=sum(r['accepted'] for r in group)
            entries.append(f'<tr><td>{task["id"]}</td><td>{len(group)}</td><td>{accepted}</td><td>{len(group)-accepted}</td></tr>')
        diagnostic_html='<section><h2>独立诊断：全部初筛候选的操作端点</h2><p>这项额外诊断不影响上面的搜索过程，不计入搜索耗时或完整任务成功率。端点通过不保证连续轨迹存在；IK 未通过也不证明不可达。</p><table><tr><th>任务</th><th>已检查候选</th><th>端点通过</th><th>端点未通过</th></tr>'+''.join(entries)+'</table><p><a href="endpoint_diagnostic.json">完整端点诊断与 IK 构型</a></p></section>'
    result={'summary':summary,'searches':rows,'attempts':details,'complete':len(rows)==6,'scope':protocol['scope']};(root/'study_summary.json').write_text(json.dumps(result,indent=2))
    table=''.join(f'<tr><td>{r["task"]}</td><td>{r["method"]}</td><td>{r["success"]}</td><td>{r["selected"] or "无"}</td><td>{len(r["attempts"])}</td><td>{r["total_wall_seconds"]:.1f}</td></tr>' for r in sorted(rows,key=lambda r:(r['task'],r['method'])))
    stats=''.join(f'<tr><td>{r["method"]}</td><td>{r["successes"]}/{r["completed_searches"]}</td><td>{r["precheck_rejections"]}</td><td>{r["navigation_calls"]}</td><td>{r["wall_seconds_sum"]:.1f}</td></tr>' for r in summary)
    attempts=''.join(f'<tr><td>{r["task"]}/{r["method"]}</td><td>{r["candidate"]}<br>{r["base_q"]}</td><td>{r["precheck"]}</td><td>{r["navigation_attempted"]}</td><td>{r["physical_pass"]}</td><td>{html.escape(r["failure"])}</td></tr>' for r in details)
    buttons=''.join(f'<button data-src="{path}">{label}</button>' for label,path in animations);viewer=f'<iframe id="viewer" src="{animations[0][1]}"></iframe>' if animations else '<p>尚无完成的物理执行动画。</p>'
    placements=''.join(f'<li>{t["id"]}：杯子中心 {t["cup_center"]} m</li>' for t in protocol['tasks'])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>停靠与抓取端点预检查对照</title><style>body{{max-width:1400px;margin:30px auto;padding:20px;font:16px system-ui;color:#18344a;background:#f4f7fa}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}iframe{{width:100%;height:700px;border:0}}button{{padding:9px;margin:4px}}.note{{background:#fff0cc;padding:15px}}</style><h1>选择停靠点时，是否先检查后续抓取？</h1><p>已完成 {len(rows)}/6 次真实顺序搜索。三个新冻结的杯子位置，每个任务两种方法。</p><p class="note">这是原场景的局部位置变化，不是三个独立场景，也不足以得出统计推广结论。所有位置在求解前冻结，失败后不移动杯子。</p><section><h2>公平比较的内容</h2><p>两种方法使用相同底盘与抓取朝向候选、顺序、GCS/IRIS、碰撞审计和真实抓取。goal_only：整机入室筛选后先导航，再检查抓取 IK。endpoint_first：先检查预抓取、抓取、抬升的 IK 与整机房间条件，通过后才导航；已求得的 IK 会复用。</p><p>每任务 300 秒总预算，每候选最多 180 秒。36 组候选首先通过相同的整机房间和碰撞筛选，再按导航距离及固定朝向顺序尝试。端点通过不等于连续操作可行，最终仍须通过完整规划和物理执行。</p><ul>{placements}</ul></section><section><h2>汇总</h2><table><tr><th>方法</th><th>完整成功/已完成搜索</th><th>提前拒绝候选</th><th>进入导航候选</th><th>搜索耗时之和 s</th></tr>{stats}</table></section><p class="note">请区分候选诊断与搜索收益：本轮完整结果中，两方法都在首个候选上成功，没有观察到提前筛选带来的成功率或速度提升。</p><section><h2>逐任务结果</h2><table><tr><th>任务</th><th>方法</th><th>成功</th><th>选择</th><th>尝试数</th><th>总秒数</th></tr>{table}</table></section><section><h2>所有候选尝试</h2><table><tr><th>任务/方法</th><th>候选/底盘</th><th>提前检查</th><th>进入导航</th><th>物理通过</th><th>未通过原因</th></tr>{attempts}</table><p>None 表示未启用提前检查；未执行物理仿真不等于物理碰撞。有限预算或局部 IK 求解失败不证明不可行。</p></section>{diagnostic_html}<section><h2>实际执行动画</h2>{buttons}{viewer}</section><p><a href="protocol.json">冻结方案</a> · <a href="comparison.json">搜索结果</a> · <a href="study_summary.json">详细汇总</a> · <a href="source_manifest.json">源码快照</a> · <a href="reference_equivalence.json">参考轨迹一致性</a></p><script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.getElementById('viewer').src=b.dataset.src)</script></html>'''
    (root/'index.html').write_text(page);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
