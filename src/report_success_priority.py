import json,statistics,collections,hashlib,shutil
from pathlib import Path
root=Path('outputs/results/success_priority_v1');d=json.loads((root/'design.json').read_text());assert (root/'complete.json').exists();contacts=json.loads((root/'contacts.json').read_text());assert contacts['completed']==contacts['total']==24
methods=['first_valid','enumerate','product'];summary={};matrix=[]
for c in d['cells']:
 row={'cell':c['id'],'methods':{}}
 for m in methods:
  r=json.loads((root/c['id']/m/'record.json').read_text());contact=next(x for x in contacts['runs'] if x['cell']==c['id'] and x['method']==m)
  plan=bool(r.get('best') and r['best'].get('discovered_seconds',float('inf'))<=d['budget']);row['methods'][m]={'plan':plan,'execution':contact['status'],'end_to_end':plan and contact['status']=='passed','record_status':r['status'],'timed_out':r['timed_out'],'first_valid_seconds':min((a['discovered_seconds'] for a in r['attempts'] if a.get('physical_valid') and a.get('policy_valid')),default=None)}
 matrix.append(row)
for m in methods:
 a=[r['methods'][m] for r in matrix];summary[m]={'attempts':len(a),'planning_success':sum(x['plan'] for x in a),'end_to_end_success':sum(x['end_to_end'] for x in a),'contact_status':dict(collections.Counter(x['execution'] for x in a)),'timeouts':sum(x['timed_out'] for x in a),'median_first_valid_seconds':statistics.median([x['first_valid_seconds'] for x in a if x['plan']]) if any(x['plan'] for x in a) else None}
changed=[p for p,h in d['hashes'].items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h];assert not changed
(root/'summary.json').write_text(json.dumps({'methods':summary,'matrix':matrix,'primary_metric':d['primary_metric']},indent=2));(root/'verification.json').write_text(json.dumps({'hashes_verified':len(d['hashes']),'changed':changed,'tests':Path('/tmp/success_tests.log').read_text()},indent=2))
lines=['# 29：以任务成功率为主的多场景验证','','本轮把主指标改为：固定 300 秒规划预算内得到语义与碰撞审计有效的计划，并通过实际接触执行。没有有效计划即计为该预定任务的端到端失败，不从执行分母中删除。次指标是规划成功率、给定有效计划后的执行成功率、首次有效解时间；路径代价和求解速度不凌驾于任务成功。','','这里的端到端范围是“已有共同候选库 → 本轮规划 → 接触执行”，不含从零构建场景和候选库，也不代表硬件实验。部分任务可能在当前有限区域覆盖中无解；本指标是预定任务尝试成功率，不假设所有条件都已证明物理可行。','','预先固定四个物理场景（原场景、杯子左移、杯子右移、推车靠近）与两条携物禁行政策，形成 8 个条件；每条件三种方法，共 24 次规划及 24 个预定执行槽位。使用一个固定随机种子，不通过重复相同场景的 seeds 夸大独立场景数。政策均为 x∈[-0.2,0.2]，第一条禁止 y∈[0,0.6]，更严政策禁止 y∈[-0.08,0.6]。其他规则同报告 28；未按结果挑选或替换条件。','','| 方法 | 300 秒内规划成功 | 规划到执行成功/全部预定任务 | 规划后执行结果 | 首个有效解中位数/秒 |','|---|---:|---:|---|---:|']
for m,s in summary.items():lines.append(f"| {m} | {s['planning_success']}/{s['attempts']} | {s['end_to_end_success']}/{s['attempts']} | {s['contact_status']} | {s['median_first_valid_seconds']} |")
lines+=['','| 场景与政策 | 首个有效候选 | 枚举候选 | 联合 GCS |','|---|---|---|---|']
for r in matrix:
 vals=[('成功' if r['methods'][m]['end_to_end'] else ('规划失败/超时' if not r['methods'][m]['plan'] else '执行未通过')) for m in methods];lines.append('| '+r['cell']+' | '+' | '.join(vals)+' |')
lines+=['','## 如何解读已有成功率','','上一轮报告 28 中，三种任务感知方法在三个可行条件（含无新增规则）各为 9/9 规划通过，预先选定 seed0 的 9 条接触执行也通过；整个门口禁行的正确拒绝应单独统计，不能把其加入可行任务成功率。之前 17/36 是候选准备的接受比例，不是机器人任务成功率；单个候选被拒绝也不等于整项任务失败。','','旧实验的 9/9、12/12 接触执行多数只覆盖指定 seed0 子集，不能推广为所有舍入种子的执行成功。本轮明确执行全部有有效计划的预定任务，并保留没有计划的槽位，从分母上纠正这个混淆。','','8 个条件仍很少，且同一物理场景的两条政策相关；不要据此宣称总体统计优势，也不把三方法在同一条件的运行当作三个独立环境。无有效解不等于物理不可行，拒绝 unsafe 轨迹是必要行为，但不应被包装成正任务成功。','','所有实际动作仍为理想驱动底盘、力驱动机械臂与夹爪、自由杯子。政策监测使用实际底盘位置与首次双指接触后锁定的携物模式。离散碰撞/接触检查不是连续时间安全证明。共同候选库成本未重新计时；本轮切片、建图、所有重试和审计均在规划时限内。','','8 项相关测试通过。全部冻结输入哈希验证通过；失败、超时、执行日志、源码快照保留于 outputs/results/success_priority_v1/。网页：http://localhost:7001/success_priority/ 。']
p=Path('reports/29_success_priority_evaluation.md');p.write_text('\n'.join(lines)+'\n');w=Path('outputs/results/grasp_fix_web/planar/success_priority');w.mkdir(exist_ok=True);options=[]
for x in contacts['runs']:
 if not x.get('output') or not (Path(x['output'])/'scene.html').exists():continue
 name=x['cell']+'_'+x['method']+'.html';shutil.copyfile(Path(x['output'])/'scene.html',w/name);options.append(f'<option value="{name}">{x["cell"]} · {x["method"]} · {x["status"]}</option>')
rows=''.join(f"<tr><td>{m}</td><td>{s['planning_success']}/{s['attempts']}</td><td>{s['end_to_end_success']}/{s['attempts']}</td><td>{s['contact_status']}</td></tr>" for m,s in summary.items())
mat=''.join('<tr><td>'+r['cell']+'</td>'+''.join('<td>'+('成功' if r['methods'][m]['end_to_end'] else '失败：'+r['methods'][m]['execution'])+'</td>' for m in methods)+'</tr>' for r in matrix)
(w/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>任务成功率验证</title><style>body{font:18px system-ui;margin:24px;color:#234}td,th{border:1px solid #ccd;padding:9px}iframe{width:100%;height:750px}select{padding:12px}</style><h1>成功率优先：8 个预定任务条件</h1><p>主要指标：300 秒内得到有效计划，并通过接触执行。无计划也算入任务尝试分母。这里从共同现有候选库开始，不含冷启动建库。</p><table><tr><th>方法</th><th>规划成功</th><th>规划到执行成功</th><th>执行状态</th></tr>'''+rows+'''</table><h2>全部条件，保留失败</h2><table><tr><th>场景与政策</th><th>首个有效候选</th><th>枚举候选</th><th>联合 GCS</th></tr>'''+mat+'''</table><p>4 个物理场景、每场景 2 条相关政策，样本有限，不构成总体统计优势证明。不可行负例正确拒绝与正任务成功率分开报告。</p><select id="s">'''+''.join(options)+'''</select><iframe id="v" title="实际接触执行"></iframe><p><a href="report.txt">完整报告</a> · <a href="summary.json">数据</a> · <a href="../carry_policy/">上一轮</a></p><script>let s=document.getElementById('s'),v=document.getElementById('v');s.onchange=()=>{if(s.value)v.src=s.value};s.onchange()</script>''')
shutil.copyfile(p,w/'report.txt');shutil.copyfile(root/'summary.json',w/'summary.json');(w.parent/'index.html').write_text('<meta http-equiv="refresh" content="0;url=success_priority/">');print(json.dumps(summary,indent=2));print(json.dumps(matrix,indent=2))
