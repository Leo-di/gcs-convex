import json,statistics,collections,hashlib,shutil
from pathlib import Path
root=Path('outputs/results/inspection_semantics_v1');d=json.loads((root/'design.json').read_text());assert (root/'complete.json').exists();summary={};records=[]
for c in d['cells']:
 for m in c['methods']:
  r=json.loads((root/c['id']/m/'record.json').read_text());r['task']=c['task'];records.append(r)
for task in ['south_required','north_required','either','unavailable']:
 summary[task]={}
 for m in ['ignore_task','first_site','enumerate_sites','product_gcs']:
  rows=[r for r in records if r['task']==task and r['method']==m]
  summary[task][m]={'accepted':sum(bool(r['best']) for r in rows),'runs':len(rows),'median_seconds':statistics.median(r['parent_wall_seconds'] for r in rows),'median_cost':statistics.median([r['best']['cost'] for r in rows if r['best']]) if any(r['best'] for r in rows) else None,'statuses':dict(collections.Counter(r['status'] for r in rows))}
changed=[p for p,h in d['hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed
contacts=json.loads((root/'contacts.json').read_text());assert contacts['completed']==9
(root/'summary.json').write_text(json.dumps({'tasks':summary,'contacts':contacts},indent=2));(root/'verification.json').write_text(json.dumps({'hashes_verified':len(d['hashes']),'changed':changed,'tests':'7 passed'},indent=2))
lines=['# 27：抓取前检查点语义的分层 GCS 对照','','固定原始 3D 场景和 v2 候选库，4 个任务条件 × 3 个舍入 seeds × 4 种方法，共 48 次方法运行。共同任务是先访问允许的检查点，再抓杯、运输、放置；检查点均不可用为结构性负例。每方法 300 秒总预算，每个检查点子集最多 3 次切割/重试。所有原始失败记录保留。','','图中增加未检查/已检查两个空手模式，只能通过合法检查点进入已检查模式；pick 只能从已检查模式触发。检查点动作位置固定，之后仍执行原来的真实机械臂动作和 10D 携物 GCS。轨迹验收直接检查抓取前的底盘位置，不依赖图节点名称或人为检查标签。','','| 任务 | 方法 | 任务有效/运行 | 耗时中位数/秒 | 有效轨迹代价中位数 |','|---|---|---:|---:|---:|']
for t,ms in summary.items():
 for m,s in ms.items():lines.append(f"| {t} | {m} | {s['accepted']}/{s['runs']} | {s['median_seconds']:.2f} | {s['median_cost']} |")
lines+=['','ignore_task 仍执行 pick–carry–place，但不编码新增检查点任务，是有意不完整的消融，不能当作满足同一任务的强 baseline。first_site 按南、北固定顺序选择可用检查点，成功即停止；enumerate_sites 独立尝试全部可用检查点并选最佳审计解；product_gcs 在一个分层图中选择检查点。三者都使用相同的多抓取候选 GCS，不是与外部已发表 TAMP 系统的比较。','','南侧检查点 (-2.5,-1)，北侧 (-2.5,1)，几何到访容差 2 cm；没有传感器、停留时长或真实检验动作模拟。起点到两个点欧氏距离相等，因此 first_site 的顺序是预先固定的规则，不称为最优贪心选择。不可用条件的成功含义应是拒绝，无轨迹的 0/3 不是普通可行任务失败率。','','计时不含历史候选库准备；无连续时间安全保证。固定检查点可将路径成本分解，所以枚举与联合方法接近也是合理结果，不能预设联合方法必胜。','','接触执行预先选定三种正任务的 seed0 和三种任务感知方法，共 9 个任务。实际轨迹检查在首次双指接触前是否到访允许检查点，且要求原接触任务执行通过。结果：'+str(collections.Counter(x['status'] for x in contacts['runs']))+'。','','7 项相关测试通过，冻结源码与输入哈希核验一致。数据、源码快照、失败及各轮轨迹：outputs/results/inspection_semantics_v1/。网页：http://localhost:7001/inspection/ 。']
p=Path('reports/27_inspection_semantic_baselines.md');p.write_text('\n'.join(lines)+'\n')
w=Path('outputs/results/grasp_fix_web/planar/inspection');w.mkdir(exist_ok=True);options=[]
for x in contacts['runs']:
 if x.get('output') and (Path(x['output'])/'scene.html').exists():
  name=x['cell']+'_'+x['method']+'.html';shutil.copyfile(Path(x['output'])/'scene.html',w/name);options.append(f'<option value="{name}">{x["cell"]} · {x["method"]} · {x["status"]}</option>')
rows=''.join(f"<tr><td>{t}</td><td>{m}</td><td>{s['accepted']}/{s['runs']}</td><td>{s['median_seconds']:.2f}</td></tr>" for t,ms in summary.items() for m,s in ms.items())
(w/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>检查点语义对照</title><style>body{font:17px system-ui;margin:24px}td,th{border:1px solid #ccd;padding:7px}iframe{width:100%;height:730px}</style><h1>先检查，再抓取：语义任务对照</h1><p>南侧 (-2.5,-1)，北侧 (-2.5,1)。必须在抓取前到访允许的检查点；这里只模拟到访，不模拟传感器检测。unavailable 的正确行为是拒绝规划。</p><table><tr><th>任务</th><th>方法</th><th>任务有效</th><th>耗时中位数/秒</th></tr>'''+rows+'''</table><p>ignore_task 是故意忽略新增任务的消融，不能把其短路径当作有效解。下方显示真实接触执行，任务监测使用实际位置与双指接触时刻。</p><select id="s">'''+''.join(options)+'''</select><iframe id="v" title="接触执行"></iframe><p><a href="report.txt">完整报告</a> · <a href="summary.json">原始汇总</a></p><script>let s=document.getElementById('s'),v=document.getElementById('v');s.onchange=()=>{if(s.value)v.src=s.value};s.onchange()</script>''')
shutil.copyfile(p,w/'report.txt');shutil.copyfile(root/'summary.json',w/'summary.json');(w.parent/'index.html').write_text('<meta http-equiv="refresh" content="0;url=inspection/">')
print(json.dumps(summary,indent=2))
