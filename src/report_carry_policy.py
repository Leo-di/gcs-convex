import json,statistics,collections,hashlib,shutil
from pathlib import Path
r=Path('outputs/results/carry_policy_v1');d=json.loads((r/'design.json').read_text());assert (r/'complete.json').exists();summary={};records=[]
for c in d['cells']:
 for m in c['methods']:
  row=json.loads((r/c['id']/m/'record.json').read_text());row['policy_name']=c['policy'];records.append(row)
for task in ['control','central_strip','upper_door','door_closed_to_carry']:
 summary[task]={}
 for m in ['ignore_rule','first_valid','enumerate','product']:
  a=[x for x in records if x['policy_name']==task and x['method']==m];good=[x for x in a if x.get('best')]
  summary[task][m]={'accepted':len(good),'total':len(a),'median_seconds':statistics.median(x['parent_wall_seconds'] for x in a),'median_cost':statistics.median(x['best']['cost'] for x in good) if good else None,'statuses':dict(collections.Counter(x['status'] for x in a))}
contacts=json.loads((r/'contacts.json').read_text());assert contacts['completed']==12
changed=[p for p,h in d['hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed
(r/'verification.json').write_text(json.dumps({'hashes_verified':len(d['hashes']),'changed':changed,'tests':Path('/tmp/policy_tests.log').read_text()},indent=2));(r/'summary.json').write_text(json.dumps({'policies':summary,'contacts':contacts},indent=2))
lines=['# 28：空手可通行、携物禁行的状态语义对照','','固定原始 3D 场景和 v2 五个候选库，四条政策 × 三个舍入 seeds × 四种方法，共 48 次运行。方法：忽略新增规则的消融、首个有效候选、完整候选枚举、联合候选 GCS。每方法 300 秒总预算，统一最多三次重试；本轮仅重试，不在线切割，避免把原区域种子误用为切片的内部锚点。','','政策在实验前固定：无禁行；门口中央 x∈[-0.2,0.2], y∈[-0.08,0.08]；门口上部 x∈[-0.2,0.2], y∈[0,0.6]；整个门口 x∈[-0.2,0.2], y∈[-0.6,0.6]。后三种在携物状态禁止底盘中心进入，空手不受这条规则约束。它不是新增实体障碍，也不是整个机器人包络禁入。原有 3D 碰撞检查仍保留。','','将每个携物凸区域与矩形外侧四个半空间分别相交，保留非空片段，增加 1 mm 政策裕量；空手圆盘保持原状。这是凸集合的精确析取表达（额外 1 mm 保守裕量除外），不是通过删除圆心落入禁区的区域来近似。片段沿用原 IRIS 元数据，仅表示原区域子集，未重新认证。','','| 政策 | 方法 | 审计有效/运行 | 耗时中位数/秒 | 有效代价中位数 |','|---|---|---:|---:|---:|']
for t,ms in summary.items():
 for m,s in ms.items():lines.append(f"| {t} | {m} | {s['accepted']}/{s['total']} | {s['median_seconds']:.2f} | {s['median_cost']} |")
lines+=['','独立政策验证使用逐段线段—矩形相交测试，可发现两个采样端点均在外侧、但中间穿越禁区的参考轨迹。携物状态由抓取状态决定；接触执行则从实际双指接触力首次超过阈值起锁定携物状态，释放阶段解除。物理执行的位置是采样数据，线段检查不构成真实连续时间证明。','','正负结果均保留。无合法图路径只证明当前有限区域图无解，不能据此证明全空间物理不可行。禁行条件改变任务，因此忽略规则的较短轨迹不能作为任务有效解。当前仍是一个物理场景，三个 seeds 不是三个独立场景。','','接触队列预先指定每种政策 seed0、三个任务感知方法，共 12 个计划槽位：'+str(collections.Counter(x['status'] for x in contacts['runs']))+'。无有效规划的槽位记录为 skipped_no_valid_plan，不从队列中删除。接触成功还要求政策检查通过。','','共同候选库准备未重跑，计时包括本轮切片、求解、重试和审计。10 项相关测试通过，包括两采样点之间穿越禁区、空手与携物状态区别、片段排除禁区与输入隔离。冻结输入哈希验证通过。','','原始数据及源码：outputs/results/carry_policy_v1/。网页：http://localhost:7001/carry_policy/ 。']
report=Path('reports/28_carry_state_policy.md');report.write_text('\n'.join(lines)+'\n');w=Path('outputs/results/grasp_fix_web/planar/carry_policy');w.mkdir(exist_ok=True);options=[]
for x in contacts['runs']:
 if not x.get('output') or not (Path(x['output'])/'scene.html').exists():continue
 name=x['cell']+'_'+x['method'];shutil.copyfile(Path(x['output'])/'scene.html',w/(name+'.html'));options.append(f'<option value="{name}">{x["cell"]} · {x["method"]} · {x["status"]}</option>')
 cell=next(c for c in d['cells'] if c['id']==x['cell']);rows=json.loads((Path(x['output'])/'policy_trace.json').read_text())
 def xy(q):return (50+(q[0]+4)*75,30+(3-q[1])*75)
 svg=['<svg xmlns="http://www.w3.org/2000/svg" width="620" height="510" viewBox="0 0 620 510"><rect width="620" height="510" fill="#fafafa"/><text x="20" y="20">XY projection: blue=empty, orange=carrying; red=policy zone</text>']
 box=cell['box']
 if box:
  left,top=xy([box['lower'][0],box['upper'][1]]);width=(box['upper'][0]-box['lower'][0])*75;height=(box['upper'][1]-box['lower'][1])*75
  svg.append(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8858555" stroke="red"/>')
 for a,b in zip(rows,rows[1:]):
  ax,ay=xy(a['base_q']);bx,by=xy(b['base_q']);color='#e87918' if a['attached'] else '#2463dc';svg.append(f'<path d="M {ax} {ay} L {bx} {by}" stroke="{color}" stroke-width="2"/>')
 svg.append('</svg>');(w/(name+'.svg')).write_text(''.join(svg))
htmlrows=''.join(f"<tr><td>{t}</td><td>{m}</td><td>{s['accepted']}/{s['total']}</td><td>{s['median_seconds']:.2f}</td></tr>" for t,ms in summary.items() for m,s in ms.items())
(w/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>携物状态禁行</title><style>body{font:17px system-ui;margin:24px}td,th{border:1px solid #ccd;padding:7px}iframe{width:100%;height:700px}</style><h1>空手可通行，携物后禁行</h1><p>红色是任务规定的底盘中心禁入区，只有携物时生效。蓝线为空手，橙线为根据实际接触事件判断的携物轨迹。左图是同次 3D 执行的 XY 投影，未与动画时间同步。</p><table><tr><th>政策</th><th>方法</th><th>任务有效</th><th>中位耗时/秒</th></tr>'''+htmlrows+'''</table><select id="s">'''+''.join(options)+'''</select><div><img id="map" alt="实际轨迹与禁行区域投影"></div><iframe id="v" title="3D 接触执行"></iframe><p><a href="report.txt">完整报告及缺失计划</a> · <a href="summary.json">数值结果</a></p><script>let s=document.getElementById('s'),v=document.getElementById('v'),m=document.getElementById('map');s.onchange=()=>{if(s.value){v.src=s.value+'.html';m.src=s.value+'.svg'}};s.onchange()</script>''')
shutil.copyfile(report,w/'report.txt');shutil.copyfile(r/'summary.json',w/'summary.json');(w.parent/'index.html').write_text('<meta http-equiv="refresh" content="0;url=carry_policy/">')
print(json.dumps(summary,indent=2));print(collections.Counter(x['status'] for x in contacts['runs']))
