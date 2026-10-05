import json,statistics,collections,hashlib,shutil
from pathlib import Path
r=Path('outputs/results/semantic_multiscene_v2');spec=json.loads((r/'protocol.json').read_text());prep=json.loads((r/'preparation.json').read_text());progress=json.loads((r/'progress.json').read_text());contacts=json.loads((r/'contacts.json').read_text());runs=progress['runs'];methods=spec['methods']
assert progress['completed']==progress['total'] and contacts['completed']==contacts['total']
changed=[p for p,h in spec['input_sha256'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed
summary={'methods':{},'scenes':{},'contacts':dict(collections.Counter(x['status'] for x in contacts['runs'])),'hashes_verified':len(spec['input_sha256'])}
med=lambda xs:statistics.median(xs) if xs else None
for m in methods:
 a=[x for x in runs if x['method']==m];b=[x for x in a if x.get('best')]
 summary['methods'][m]={'accepted':len(b),'total':len(a),'median_wall_seconds':med([x['parent_wall_seconds'] for x in a]),'statuses':dict(collections.Counter(x['status'] for x in a)),'invalid_attempts':sum(t.get('solver_status')=='solved' and not t.get('accepted') for x in a for t in x['attempts']),'contact_status':dict(collections.Counter(x['status'] for x in contacts['runs'] if x['method']==m))}
for n,p in prep['scenes'].items():summary['scenes'][n]={'accepted':p['accepted'],'proposed':len(p['attempts']),'preparation_seconds':p['preparation_seconds'],'rejections':dict(collections.Counter(a.get('stage') for a in p['attempts'] if a['status']!='accepted'))}
pairs=[]
for c in spec['cells']:
 a={x['method']:x for x in runs if x['cell']==c['id']}
 if all(x.get('best') for x in a.values()):
  b=a['B2_enumerate']['best'];j=a['joint_gcs']['best'];pairs.append({'cell':c['id'],'joint_vs_B2_percent':100*(j['cost']/b['cost']-1),'same_candidate':len({x['best']['candidate'] for x in a.values()})==1,'candidates':{m:x['best']['candidate'] for m,x in a.items()}})
summary['pairs']=pairs;summary['batch_wall_seconds']={'preparation':prep['wall_seconds'],'planning':json.loads((r/'complete.json').read_text())['elapsed_seconds'],'contacts':contacts['elapsed_seconds']}
(r/'summary.json').write_text(json.dumps(summary,indent=2));(r/'verification.json').write_text(json.dumps({'changed':changed,'hashes_verified':len(spec['input_sha256']),'tests':Path('/tmp/v2_tests.log').read_text()},indent=2))
lines=['# 25：修复版本的全量多场景对照','','本轮固定相同四个物理场景、九个提案、三个 rounding seeds，重新生成全部候选，再独立执行 B1/B2/joint，共 36 次方法运行。统一开启碰撞检查预抓取路径和携物碰撞反例收紧；准备阶段采用同一套修复，不按方法或结果区别处理。源码快照在生成候选前保存。','','| 场景 | 可用候选 | 准备秒数 |','|---|---:|---:|']
for n,s in summary['scenes'].items():lines.append(f"| {n} | {s['accepted']}/{s['proposed']} | {s['preparation_seconds']:.2f} |")
lines+=['','| 方法 | 审计通过 | 运行耗时中位数（秒） | 求解有输出但审计拒绝的尝试 |','|---|---:|---:|---:|']
for m,s in summary['methods'].items():lines.append(f"| {m} | {s['accepted']}/{s['total']} | {s['median_wall_seconds']:.3f} | {s['invalid_attempts']} |")
lines+=['','上述时间包含进程启动、加载、建图、求解、完整展开和审计，不包含共同的候选库准备。B1 找到第一个审计有效解即停止；B2 独立枚举并保留最低代价；joint 使用一个多分支 GCS。三者使用相同冻结区域、代价、审计。规划每方法限时 300 秒，固定配对 CPU、单线程。准备成本是本实现全量预生成成本，不代表按需生成候选的最优实现。','','接触执行队列在实验前固定为所有场景 seed0、全部方法，共 12 个任务。结果：'+json.dumps(summary['contacts'],ensure_ascii=False)+'。失败与无有效轨迹均保留。各方法执行结果：'+json.dumps({m:s['contact_status'] for m,s in summary['methods'].items()},ensure_ascii=False)+'。','','三方法均有有效结果的配对数：%d；三者选择同一候选：%d。'%(len(pairs),sum(p['same_candidate'] for p in pairs))]
if pairs:lines+=['joint 相对 B2 的加权轨迹代价差异范围 %.6f%% 至 %.6f%%；这是同一有限候选库中解的比较，不是全局最优差距，也未做显著性检验。'%(min(p['joint_vs_B2_percent'] for p in pairs),max(p['joint_vs_B2_percent'] for p in pairs))]
lines+=['','重要范围：反例收紧只在共同候选库准备时执行。计时的三个规划器使用冻结区域，不再进行按方法的区域修复；若新求解结果违反碰撞约束，由相同审计拒绝。区域随机抽样与加密轨迹检查不是连续时间安全证明。RRT 只是携物引导路径生成，最终对照的连续路径仍由 GCS 优化。','','与 v1 相比，场景和候选网格相同，但源代码、数值处理、预抓取路径和区域准备均改变。因此可报告总体管线的前后变化，不能把变化单独归因于某一个修复。四个场景本身仍有限，三个 seeds 也不是三个独立物理场景。','','分阶段墙钟（秒）：'+json.dumps(summary['batch_wall_seconds'])+'。原始数据与源码快照：outputs/results/semantic_multiscene_v2/。','','网页：http://localhost:7001/multiscene_v2/ 。复现命令：`LCM_DEFAULT_URL=memq:// OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m src.benchmark_semantic_multiscene --repaired --root <新目录>`。']
report=Path('reports/25_repaired_multiscene_baselines.md');report.write_text('\n'.join(lines)+'\n')
w=Path('outputs/results/grasp_fix_web/planar/multiscene_v2');w.mkdir(exist_ok=True)
options=[]
for x in contacts['runs']:
 if x.get('output') and (Path(x['output'])/'scene.html').exists():
  file=x['cell']+'_'+x['method']+'.html';shutil.copyfile(Path(x['output'])/'scene.html',w/file);options.append(f'<option value="{file}">{x["cell"]} · {x["method"]} · {x["status"]}</option>')
rows=''.join(f"<tr><td>{m}</td><td>{s['accepted']}/{s['total']}</td><td>{s['median_wall_seconds']:.3f}</td><td>{s['invalid_attempts']}</td></tr>" for m,s in summary['methods'].items())
(w/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>修复版本全量对照</title><style>body{font:18px system-ui;margin:24px;color:#234}td,th{padding:10px;border:1px solid #ccd}iframe{width:100%;height:750px}select{padding:12px}</style><h1>固定修复版本：四场景全量对照</h1><p>36 个候选重新生成，36 次规划方法运行。候选可用数：'''+str({n:s['accepted'] for n,s in summary['scenes'].items()})+'''。</p><table><tr><th>方法</th><th>审计有效</th><th>耗时中位数/秒（不含准备）</th><th>审计拒绝尝试</th></tr>'''+rows+'''</table><p>预先指定的接触执行结果：'''+str(summary['contacts'])+'''。三者同选候选的配对：'''+str(sum(p['same_candidate'] for p in pairs))+'/'+str(len(pairs))+'''。</p><p>下方是实际自由杯子接触执行。区域和轨迹验证均非连续安全证明；详细失败与准备成本见报告。</p><select id="s">'''+''.join(options)+'''</select><iframe id="v" title="接触执行"></iframe><p><a href="report.txt">完整报告</a> · <a href="summary.json">数值结果</a> · <a href="../multiscene/">旧版对照</a></p><script>let s=document.getElementById('s'),v=document.getElementById('v');s.onchange=()=>{if(s.value)v.src=s.value};s.onchange()</script>''')
shutil.copyfile(report,w/'report.txt');shutil.copyfile(r/'summary.json',w/'summary.json')
(w.parent/'index.html').write_text('<meta http-equiv="refresh" content="0;url=multiscene_v2/"><a href="multiscene_v2/">修复版全量对照</a>')
print(json.dumps(summary,indent=2))
