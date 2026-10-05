import json,statistics,collections,hashlib,shutil
from pathlib import Path
r=Path('outputs/results/online_repair_ablation_v1');design=json.loads((r/'design.json').read_text());methods=['B1_first_valid','B2_enumerate','joint_gcs'];summary={}
for mode in design['arms']:
 progress=json.loads((r/mode/'progress.json').read_text());assert progress['completed']==36;rows=progress['runs'];summary[mode]={}
 for m in methods:
  a=[x for x in rows if x['method']==m];history=[h for x in a for t in x['attempts'] for h in t['repair_history']]
  summary[mode][m]={'accepted':sum(bool(x.get('best')) for x in a),'total':len(a),'median_wall_seconds':statistics.median(x['parent_wall_seconds'] for x in a),'solve_calls':len(history),'cut_rounds':sum(h.get('repair_status')=='cut' for h in history),'statuses':dict(collections.Counter(x['status'] for x in a)),'retried_subsets':sum(len(t['repair_history'])>1 for x in a for t in x['attempts'])}
contacts=json.loads((r/'cuts/contacts.json').read_text());assert contacts['completed']==contacts['total']==12
changed=[p for p,h in design['hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed
verification={'changed':changed,'verified_hashes':len(design['hashes']),'tests':Path('/tmp/online_tests.log').read_text()};(r/'verification.json').write_text(json.dumps(verification,indent=2))
(r/'summary.json').write_text(json.dumps({'arms':summary,'contacts':dict(collections.Counter(x['status'] for x in contacts['runs']))},indent=2))
lines=['# 26：在线重试与区域修复的公平预算消融','','复用 v2 的同一冻结候选库，四场景 × 三个 seeds × 三方法 × 三条件，共 108 次独立方法运行。每方法总墙钟限时 300 秒；每个候选子集最多四次求解。全部条件使用本轮同一代码，none 也重新运行，不直接借用旧时间。','','| 条件 | 方法 | 审计成功 | 耗时中位数/秒 | 规划调用次数 | 切割轮数 |','|---|---|---:|---:|---:|---:|']
for mode in design['arms']:
 for m,s in summary[mode].items():lines.append(f"| {mode} | {m} | {s['accepted']}/{s['total']} | {s['median_wall_seconds']:.3f} | {s['solve_calls']} | {s['cut_rounds']} |")
lines+=['','none 为一次求解；retry 只更换舍入种子；cuts 在检测到携物碰撞时尝试保留区域种子的距离梯度分离半空间，同时使用与 retry 相同的舍入种子序列 seed+101×round。未能切割时仍按相同规则重试；非携物碰撞不做这种修复。每个候选子集一旦得到通过完整审计的解就停止重试，不以重试搜寻更低代价。B2 仍继续其他候选；joint 每次重新求解多分支图。','','规划调用次数包含建图后直接判定不可达的返回，不全部是数值优化器调用。当前重试规则也会重复尝试图不可达的分支；图与区域未变化时，这类重试没有结构性收益，应在后续版本提前停止，但本轮按冻结规则保留其耗时。复制候选库后仅在当前方法与候选子集内部修改，不向其他方法或其他候选子集共享修复。总耗时包含进程启动、读取、所有重试、审计和区域切割。共同的离线候选库成本未重新测量，见报告 25。','','切割后的区域仍需最终轨迹审计；本轮没有重做整区域随机审计。updated_cover.json 中保留的离线 holdout 数值明确标记为修改前审计，不可冒充新区域测量。局部梯度切割不构成全区域安全证明。','','接触验证按事先确定的队列，仅对 cuts 条件、所有场景 seed0、所有三方法执行，共 12 个任务：'+json.dumps(dict(collections.Counter(x['status'] for x in contacts['runs'])),ensure_ascii=False)+'。其他种子仅做规划审计，不能把这 12 次接触结果推广到全部 108 次运行。','','所有失败及中间重试保留。原始数据、各轮区域与轨迹、源码快照：outputs/results/online_repair_ablation_v1/。9 项相关测试通过（含输入库隔离、重试预算、首个已审计解停止重试）。冻结输入哈希核验通过。','','网页：http://localhost:7001/online_repairs/ 。复现入口：src/benchmark_online_repairs.py（默认输出存在时拒绝覆盖）。']
report=Path('reports/26_online_repair_ablation.md');report.write_text('\n'.join(lines)+'\n')
w=Path('outputs/results/grasp_fix_web/planar/online_repairs');w.mkdir(exist_ok=True)
options=[]
for x in contacts['runs']:
 if x.get('output') and (Path(x['output'])/'scene.html').exists():
  file=x['cell']+'_'+x['method']+'.html';shutil.copyfile(Path(x['output'])/'scene.html',w/file);options.append(f'<option value="{file}">{x["cell"]} · {x["method"]} · {x["status"]}</option>')
rows=''.join(f"<tr><td>{mode}</td><td>{m}</td><td>{s['accepted']}/{s['total']}</td><td>{s['median_wall_seconds']:.2f}</td><td>{s['cut_rounds']}</td></tr>" for mode in design['arms'] for m,s in summary[mode].items())
(w/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>在线修复消融</title><style>body{font:18px system-ui;margin:24px;color:#234}td,th{padding:8px;border:1px solid #ccd}iframe{width:100%;height:740px}select{padding:12px}</style><h1>108 次规划：重试与区域修复消融</h1><p>none：不重试；retry：仅换舍入种子；cuts：区域切割加同样重试。三方法均使用相同规则与 300 秒总预算。</p><table><tr><th>条件</th><th>方法</th><th>审计成功</th><th>耗时中位数/秒</th><th>切割轮数</th></tr>'''+rows+'''</table><p>以下是预先指定的 cuts / seed0 接触执行，不能代表全部规划运行。局部切割与数值检查不是连续安全证明。</p><select id="s">'''+''.join(options)+'''</select><iframe id="v" title="接触执行"></iframe><p><a href="report.txt">完整报告</a> · <a href="summary.json">数值结果</a> · <a href="../multiscene_v2/">上一轮</a></p><script>let s=document.getElementById('s'),v=document.getElementById('v');s.onchange=()=>{if(s.value)v.src=s.value};s.onchange()</script>''')
shutil.copyfile(report,w/'report.txt');shutil.copyfile(r/'summary.json',w/'summary.json');(w.parent/'index.html').write_text('<meta http-equiv="refresh" content="0;url=online_repairs/"><a href="online_repairs/">在线修复对照</a>')
print(json.dumps(summary,indent=2))
