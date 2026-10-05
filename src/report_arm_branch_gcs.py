"""Render every branch-selection result and returned-plan execution."""
import html
import json
from pathlib import Path
import shutil
import statistics

from .arm_branch_gcs import ROOT, METHODS


LABELS={'base_only':'底盘语义', 'fixed_negative_y':'固定负 y 收臂',
        'enumerate':'分支枚举', 'joint_gcs':'联合 GCS'}


def main():
    web=Path('outputs/results/grasp_fix_web/planar/arm_branches');web.mkdir(parents=True,exist_ok=True)
    records=json.loads((ROOT/'summary.json').read_text())
    status=json.loads((ROOT/'execution_status.json').read_text())
    statuses={(s['case'],s['method']):s for s in status}
    manifest=json.loads((ROOT/'execution_protocol.json').read_text())
    if len(status)!=len(manifest['attempts']):raise RuntimeError('Execution manifest is incomplete')
    data={};details='';positive=[r for r in records if r['case']['expected']=='positive']
    combined=[]
    for record in records:
        case=record['case']['name'];method=record['method'];directory=ROOT/case/method/'execution'
        actual=None;url=None
        if record['case']['expected']=='positive':
            state=statuses[(case,method)]
            if state['execution_status']=='completed':
                actual=json.loads((directory/'result.json').read_text())
                url=f'{case}_{method}.html';shutil.copy2(directory/'scene.html',web/url)
        title=('两侧均允许' if case.startswith('free') else
               '避开正 y 一侧' if case.startswith('avoid_positive') else
               '避开负 y 一侧' if case.startswith('avoid_negative') else '两侧均限制')
        title+=f'，切换位置 x={record["case"]["gate_x"]}'
        branch=record['result'].get('selected_branch')
        item=dict(title=title,method=LABELS[method],url=url,branch=branch,
                  planned=record['audit']['accepted'],status=record['result']['status'],execution=actual)
        data.setdefault(case,{})[method]=item
        combined.append(dict(case=record['case'],method=method,planning=record['audit'],execution=actual,
                             objective=record['result'].get('objective'),selected_branch=branch))
        physical=('通过' if actual['accepted'] else '语义违规') if actual else (
                  '负例不执行' if record['case']['expected']!='positive' else '规划失败或执行失败')
        residual='—' if actual is None else ', '.join(f'{k}: {100*v:.2f}' for k,v in actual['minimum_semantic_residuals_m'].items())
        details+=f'<tr><td>{html.escape(title)}</td><td>{LABELS[method]}</td><td>{branch or "—"}</td><td>{record["result"]["status"]}</td><td>{physical}</td><td>{residual}</td></tr>'
    summary={};summary_table=''
    for method in METHODS:
        values=[r for r in combined if r['method']==method and r['case']['expected']=='positive']
        plans=sum(r['planning']['accepted'] for r in values)
        executed=sum(r['execution'] is not None for r in values)
        successes=sum(bool(r['execution'] and r['execution']['accepted']) for r in values)
        summary[method]=dict(conditions=len(values),planned=plans,executed=executed,successful=successes)
        summary_table+=f'<tr><td>{LABELS[method]}</td><td>{plans}/{len(values)}</td><td>{executed}</td><td>{successes}/{len(values)}</td></tr>'
    (ROOT/'combined_summary.json').write_text(json.dumps(dict(methods=summary,records=combined),indent=2))
    cases=list(dict.fromkeys(r['case']['name'] for r in positive))
    options=''.join(f'<option value="{c}">{html.escape(data[c]["joint_gcs"]["title"])}</option>' for c in cases)
    methods=''.join(f'<option value="{m}">{LABELS[m]}</option>' for m in METHODS)
    page='''<!doctype html><meta charset="utf-8"><title>多姿态分支的全臂语义 GCS</title>
<style>body{font:17px system-ui;margin:24px;max-width:1450px;color:#172b4d}p{line-height:1.55}select,button{font:inherit;padding:10px;margin:6px}iframe{width:100%;height:680px;border:1px solid #ccd}td,th{border:1px solid #ccd;padding:9px}table{border-collapse:collapse;margin:16px 0}.note{background:#eef5fc;padding:14px}#status{padding:12px;background:#f2f5f8}</style>
<h1>语义改变时 GCS 会不会换一种收臂方式</h1>
<p>在原始双房间三维场景中加入两个收臂分支。任务要求全臂先留在房间内，再穿门；可额外要求机械臂避开正 y 或负 y 一侧。透明平面表示语义约束，不是新增实体障碍。</p>
<table><tr><th>方法</th><th>正例参考轨迹通过</th><th>实际执行次数</th><th>规划并执行成功</th></tr>'''+summary_table+'''</table>
<p>固定分支的两个规划失败保留在 6 个条件的分母中。分支枚举与联合 GCS 均通过全部 6 个正例，当前没有联合方法优于枚举的成功率证据。</p>
<select id="case" onchange="show()">'''+options+'''</select><select id="method" onchange="show()">'''+methods+'''</select>
<button onclick="opposite()">切换需要避开的侧</button><div id="status"></div><iframe id="scene"></iframe>
<p>展示全部 22 次返回轨迹的力驱动实际仿真记录。红色平面：进入过门模式前，全臂须留在 x≤0。蓝色平面：全程生效的侧向手臂边界。通过 Open Controls 可重播。</p>
<p class="note">当前搜索两个二维仿射姿态分支的并集，图内保持完整 10 维构型连续，但没有任意搜索全部 7 个手臂关节自由度。六个正例来自同一场景，是构造的机制对照；尚未接回完整抓取运输流程。</p>
<h2>全部结果</h2><table><tr><th>条件</th><th>方法</th><th>选中分支</th><th>规划状态</th><th>执行结果</th><th>实际最小语义 residual cm</th></tr>'''+details+'''</table>
<p>两侧同时限制的负例在当前两分支覆盖图中不可达；不能推断完整构型空间无解。碰撞区域采用数值审核；实际状态每 50 ms 记录并检查，包含插值到切换位置的状态，不构成连续时间执行安全证明。</p>
<script>const data=__DATA__;
function show(){const c=document.getElementById('case').value,m=document.getElementById('method').value,d=data[c][m];
const b=d.branch==='positive_y'?'朝正 y 收臂':d.branch==='negative_y'?'朝负 y 收臂':'无合法分支';
document.getElementById('status').textContent=d.method+'：'+b+'。'+(d.execution?(d.execution.accepted?'实际执行通过':'实际执行违反语义'):'没有可播放的执行轨迹；该项仍计入失败');
const frame=document.getElementById('scene');frame.style.display=d.url?'block':'none';frame.src=d.url||'about:blank';}
function opposite(){const c=document.getElementById('case'),gate=c.value.slice(c.value.indexOf('_gate'));c.value=(c.value.startsWith('avoid_negative')?'avoid_positive_y':'avoid_negative_y')+gate;show();}
document.getElementById('case').value='avoid_negative_y_gate-0.6';document.getElementById('method').value='joint_gcs';show();</script>'''
    (web/'index.html').write_text(page.replace('__DATA__',json.dumps(data,ensure_ascii=False)))
    print(json.dumps(summary,indent=2))
    costs=[]
    for case in cases:
        pair={r['method']:r for r in records if r['case']['name']==case}
        costs.append(abs(pair['joint_gcs']['result']['objective']-pair['enumerate']['result']['objective']))
    print('max joint/enumeration objective difference',max(costs))
    for method in ['enumerate','joint_gcs']:
        print(method,'median solve seconds',statistics.median(r['result']['solve_seconds'] for r in positive if r['method']==method))


if __name__=='__main__':main()
