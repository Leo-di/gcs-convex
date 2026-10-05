"""Publish the entire fixed doorway experiment, including failed controls."""
import html
import json
from pathlib import Path
import shutil


def main():
    root=Path('outputs/results/arm_boundary_gcs_v1')
    web=Path('outputs/results/grasp_fix_web/planar/arm_gcs')
    records=json.loads((root/'summary.json').read_text())
    protocol=json.loads((root/'execution_protocol.json').read_text())
    executions=[];links=[]
    for item in protocol['attempts']:
        directory=root/item['case']/item['method']/'execution'
        result=json.loads((directory/'result.json').read_text())
        executions.append(result)
        name=item['case']+'_'+item['method']+'.html'
        shutil.copy2(directory/'scene.html',web/name)
        label=('全臂语义' if item['method']=='semantic_gcs' else '底盘基线')+' · '+item['case']
        links.append(f'<option value="{name}">{html.escape(label)}</option>')
    (root/'execution_summary.json').write_text(json.dumps(executions,indent=2))
    physical={(r['case'],r['method']):r for r in executions}
    table=''
    for record in records:
        case=record['case']['name'];method=record['method'];audit=record['audit']
        actual=physical.get((case,method))
        result='未执行（事后拒绝或无轨迹）' if actual is None else ('通过' if actual['accepted'] else '语义违规')
        if actual is None and record['case']['expected']!='positive':
            result='未执行（负例不在执行清单）'
        residual='—' if actual is None else f'{actual["minimum_active_semantic_residual_m"]*100:.2f} cm'
        table+=f'<tr><td>{case}</td><td>{method}</td><td>{record["result"]["status"]}</td><td>{audit["accepted"]}</td><td>{result}</td><td>{residual}</td></tr>'
    default='x-1.3_gate-0.6_semantic_gcs.html'
    (web/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>全臂语义 × GCS：执行对照</title>
<style>body{font:17px system-ui;margin:24px;color:#172b4d;max-width:1400px}select,button{font:inherit;padding:10px;margin:6px}iframe{width:100%;height:660px;border:1px solid #ccd}td,th{padding:8px;border:1px solid #ccd}table{border-collapse:collapse}p{line-height:1.55}.note{background:#eef5fc;padding:12px}</style>
<h1>原始三维场景：全臂语义让 GCS 提前收臂</h1>
<p><b>4 个正例：全臂语义规划与执行均 4/4 通过；底盘基线均违反全臂规则；事后检查全部拒绝。</b>
另有 1 个限制收臂幅度的负例，全臂语义 GCS 拒绝。</p>
<p>任务要求：底盘到达模式切换位置 x=−0.6（或 −0.5）之前，全部机械臂和夹爪几何须保持 x≤0，随后允许穿门。红平面为 x=0 语义边界。</p>
<p>下方展示力驱动控制器的<b>实际仿真记录</b>；没有逐帧设置机器人位置。可切换全部 8 次执行。Meshcat 的 Open Controls 可重播动画。</p>
<select id="choice" onchange="show(this.value)">'''+''.join(links)+f'''</select>
<button onclick="show('x-1.3_gate-0.6_base_only.html')">看基线违规</button>
<button onclick="show('{default}')">看语义 GCS 收臂</button>
<iframe id="scene" src="{default}"></iframe>
<p class="note">研究范围：同一三维场景中的过门子任务。优化变量写为完整 10 维构型，但凸区域被限制在“底盘 x + 一个耦合收臂参数”的二维仿射族内；底盘 y/yaw 固定。当前结果不是自由 10 维规划，也不是完整抓取运输任务成功率。</p>
<table><tr><th>条件</th><th>方法</th><th>GCS</th><th>参考轨迹通过</th><th>动力学执行</th><th>实际最小有效语义 residual</th></tr>'''+table+'''
</table><p>语义区域使用运动学 Lipschitz 上界收紧；碰撞区域以有限采样审核（独立审核 2400 点未发现碰撞）。实际执行按 50 ms 状态记录检查，并检查插值到达模式切换位置的状态；不构成连续时间动力学安全证明。</p>
<p>postcheck 是复用同一基线轨迹、发现违规即拒绝的控制组，没有重规划能力。四个正例是相关的机制验证，不用于声称优于现有研究方法。</p>
<script>function show(x){document.getElementById('choice').value=x;document.getElementById('scene').src=x}document.getElementById('choice').value='x-1.3_gate-0.6_semantic_gcs.html';</script>''')
    print(json.dumps({m:dict(attempts=sum(r['method']==m for r in executions),
          accepted=sum(r['method']==m and r['accepted'] for r in executions))
          for m in ('base_only','semantic_gcs')},indent=2))


if __name__=='__main__':main()
