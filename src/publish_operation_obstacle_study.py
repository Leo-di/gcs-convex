"""Review endpoint success versus validated continuous operation, with failures."""
import argparse,json,html
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);protocol=json.loads((root/'protocol.json').read_text());rows=json.loads((root/'comparison.json').read_text()) if (root/'comparison.json').exists() else []
    details=[];animations=[];summary=[]
    for task in protocol['tasks']:
        for method in protocol['methods']:
            folder=root/task['id']/method
            for path in sorted(folder.glob('*/candidate_result.json')):
                r=json.loads(path.read_text());op=r.get('operation',{});endpoint=bool(r.get('joint_endpoint_check',{}).get('accepted') or (op.get('goal_checks') and all(x['satisfied'] for x in op['goal_checks'].values())))
                details.append({'task':task['id'],'method':method,'candidate':r['id'],'endpoint_pass':endpoint,'operation_pass':op.get('accepted',False),'bridge_pass':r.get('bridge',{}).get('accepted',False),'physical_pass':r.get('physical',{}).get('accepted',False),'navigation_attempted':(path.parent/'bridge_events.json').exists(),'stage':r.get('failure_stage',r.get('stage')),'reason':r.get('failure_reason') or r.get('bridge',{}).get('reason','') or ('candidate wall-time limit' if r.get('process_exit')=='timeout' else ''),'exit':r.get('process_exit'),'path':str(path.relative_to(root))})
                if r.get('physical',{}).get('method') and not r['physical']['accepted']:
                    details[-1]['reason']='Physical execution checks rejected; inspect contact and tracking metrics in record'
    for row in rows:
        if row['success']:
            rel=f"{row['task']}/{row['method']}/{row['selected']}/execution/execution_grounded_candidate/scene.html";animations.append((f"实际执行：{row['task']} / {row['method']}",rel))
    for method in protocol['methods']:
        group=[r for r in rows if r['method']==method];attempts=[r for r in details if r['method']==method]
        summary.append({'method':method,'completed_searches':len(group),'successes':sum(r['success'] for r in group),'total_wall_seconds':sum(r['total_wall_seconds'] for r in group),'attempts':len(attempts),'navigation_calls':sum(r['navigation_attempted'] for r in attempts),'endpoint_pass_operation_not_pass':sum(r['endpoint_pass'] and not r['operation_pass'] for r in attempts)})
    result={'summary':summary,'searches':rows,'attempts':details,'complete':len(rows)==9,'scope':protocol['scope']};(root/'study_summary.json').write_text(json.dumps(result,indent=2))
    candidate_lookup={c['id']:c for c in protocol['candidates']}
    table=''.join(f'<tr><td>{r["task"]}</td><td>{r["method"]}</td><td>{r["success"]}</td><td>{r["selected"] or "无"}<br>RPY: {candidate_lookup.get(r["selected"],{}).get("grasp_rpy","—")}</td><td>{len(r["attempts"])}</td><td>{r["total_wall_seconds"]:.1f}</td></tr>' for r in sorted(rows,key=lambda r:(r['task'],r['method'])))
    stats=''.join(f'<tr><td>{r["method"]}</td><td>{r["successes"]}/{r["completed_searches"]}</td><td>{r["attempts"]}</td><td>{r["navigation_calls"]}</td><td>{r["endpoint_pass_operation_not_pass"]}</td><td>{r["total_wall_seconds"]:.1f}</td></tr>' for r in summary)
    attempts=''.join(f'<tr><td>{r["task"]}/{r["method"]}</td><td>{r["candidate"]}</td><td>{r["endpoint_pass"]}</td><td>{r["operation_pass"]}</td><td>{r["physical_pass"]}</td><td>{r["stage"]}: {html.escape(r["reason"])} <a href="{r["path"]}">记录</a></td></tr>' for r in details)
    layouts=[(f'静态布局：{t["id"]}',f'layouts/{t["id"]}.html') for t in protocol['tasks'] if (root/'layouts'/f'{t["id"]}.html').exists()];views=animations+layouts
    buttons=''.join(f'<button data-src="{path}">{label}</button>' for label,path in views);viewer=f'<iframe id="viewer" src="{views[0][1]}"></iframe>' if views else ''
    plots=[]
    for task in protocol['tasks']:
        panel=task['obstacle'];width=panel['size'][0];left=panel['center'][0]-width/2
        plots.append(f'<div><b>{task["id"]}：{width*100:.0f} cm</b><svg viewBox="0 0 300 320" width="260"><rect x="20" y="20" width="260" height="280" fill="#e7f4e6"/><rect x="{(0.33-.25)*300}" y="30" width="210" height="240" fill="#b58a65"/><circle cx="{(.46-.25)*300}" cy="{(1.2-.85)*300}" r="10" fill="#168ed1"/><rect x="{(left-.25)*300}" y="{(1.2-.57)*300}" width="{width*300}" height="12" fill="#dc7b27"/><text x="30" y="310">杯子蓝色；桌面棕色；隔板橙色</text></svg></div>')
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>操作障碍与连续轨迹验证</title><style>body{{max-width:1450px;margin:30px auto;padding:20px;font:16px system-ui;color:#18344a;background:#f4f7fa}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #ddd;padding:9px;text-align:left}}iframe{{width:100%;height:720px;border:0}}button{{padding:9px;margin:4px}}.note{{background:#fff0cc;padding:15px}}</style><h1>合法操作端点能否连成有效轨迹？</h1><p>完成 {len(rows)}/9 次顺序搜索；三种冻结隔板布局，三种检查顺序。每次 300 秒总预算、每候选最多 180 秒。</p><p class="note">所有方法最终都必须通过同样的连续轨迹审计和真实接触执行。区别仅是何时检查操作；不是把不检查碰撞的轨迹算成基线成功。失败或超时不证明 C-space 不连通。</p><section><h2>固定布局</h2><p>杯子 (0.46,0.85,0.84) m、桌子与机器人不变。隔板中心 (0.72,0.55,1.05) m，高 60 cm、厚 4 cm，下沿位于桌面 z=0.75 m，只改变宽度。未按求解结果移动障碍。</p><div style="display:flex;flex-wrap:wrap">{''.join(plots)}</div></section><section><h2>方法及汇总</h2><p>goal_only：先导航再操作。endpoint_first：先检查三个操作端点。operation_first：先完成操作引导、官方 IRIS/GCS 和严格审计，全部通过才调用导航。通过的操作路径直接复用。</p><table><tr><th>方法</th><th>成功/已完成</th><th>尝试数</th><th>进入导航</th><th>端点通过但操作未通过</th><th>搜索耗时之和 s</th></tr>{stats}</table><p>“操作未通过”包含引导失败、求解失败、审计拒绝和超时，具体原因见下表，不能统一称为路径不存在。统计来自相关开发场景，不能据此推断广泛性能优势。</p></section><section><h2>逐任务结果</h2><table><tr><th>布局</th><th>方法</th><th>完整成功</th><th>选择</th><th>尝试数</th><th>总秒数</th></tr>{table}</table></section><section><h2>全部尝试</h2><table><tr><th>布局/方法</th><th>候选</th><th>端点通过</th><th>操作通过</th><th>物理通过</th><th>阶段与原因</th></tr>{attempts}</table></section><section><h2>3D 布局与适用时的实际执行</h2>{buttons}{viewer}</section><p><a href="protocol.json">冻结方案</a> · <a href="study_summary.json">全部诊断</a> · <a href="interpolation_diagnostic.json">直线插值反例（非规划器）</a> · <a href="source_manifest.json">源码快照</a></p><script>document.querySelectorAll('button').forEach(b=>b.onclick=()=>document.getElementById('viewer').src=b.dataset.src)</script></html>''';(root/'index.html').write_text(page);print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
