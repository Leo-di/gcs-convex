"""Publish policy success separately from the professor's common task criteria."""
import argparse
import html
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();root=Path(a.root)
    protocol=json.loads((root/'protocol.json').read_text());rows=[];links=[];first=None
    for task in protocol['tasks']:
        name=task['id'];folder=root/name/'operation_first';f=folder/'search_result.json'
        r=json.loads(f.read_text()) if f.exists() else {}
        complete='total_wall_seconds' in r
        physical={};candidate=None
        if r.get('selected'):
            candidate=folder/r['selected']
            c=json.loads((candidate/'candidate_result.json').read_text());physical=c.get('physical',{})
        scene=candidate/'execution/execution_grounded_candidate/scene.html' if candidate else None
        if scene and scene.exists():
            url=str(scene.relative_to(root));first=first or url
            links.append(f'<button onclick="document.querySelector(\'iframe\').src={html.escape(json.dumps(url),quote=True)}">{name} 动画</button>')
        residual=physical.get('room_min_m')
        status=str(r.get('success',False)) if complete else '运行中'
        rows.append(f'<tr><td>{name}</td><td>{status}</td><td>{physical.get("semantic_task_satisfied","—")}</td><td>{len(r.get("attempts",[]))}</td><td>{r.get("selected","—")}</td><td>{f"{residual*100:.2f}" if residual is not None else "—"}</td><td>{physical.get("unexpected_contacts","—")}</td></tr>')
    attempts=[]
    for f in sorted(root.glob('*/operation_first/*/candidate_result.json')):
        r=json.loads(f.read_text());attempts.append(f'<tr><td>{f.parents[2].name}</td><td>{r.get("id")}</td><td>{html.escape(str(r.get("base_q")))}</td><td>{html.escape(str(r.get("failure_reason",r.get("stage"))))}</td></tr>')
    body=''.join(rows);buttons=' '.join(links)
    page=f"""<!doctype html><html lang="zh"><meta charset="utf-8"><title>同场景语义对照</title>
<style>body{{max-width:1400px;margin:32px auto;font:17px system-ui;color:#213547}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #ccd}}button{{padding:12px;margin:8px;cursor:pointer}}iframe{{width:100%;height:720px;border:1px solid #ddd}}p{{line-height:1.7}}</style>
<h1>同一场景：几何可行 ≠ 满足整机入室语义</h1>
<p>仅改变语义门控。相同的冻结窄屏障场景、45 个候选、排序、GCS/IRIS 实现和每组 600 秒搜索预算（每候选最多 300 秒）。未按结果移动桌子或杯子。</p>
<p><b>方法成功</b>按各组自身约束检查；<b>教授任务成功</b>统一要求导航终点整机入室、操作全程整机留在 Room 2，并通过接触动力学检查。仅几何组也完整记录语义违规，不能将方法成功当作教授任务成功。</p>
<table><tr><th>策略</th><th>方法成功</th><th>教授任务成功</th><th>已尝试</th><th>选中候选</th><th>操作最小房间余量 cm</th><th>意外接触</th></tr>{body}</table>
<p>geometry_only：不施加房间约束；goal_only：仅导航目标和操作端点施加房间约束；full_motion：额外在操作路径和 IRIS 区域中施加房间约束。</p>
{buttons}<iframe src="{first or 'about:blank'}"></iframe>
<p>动画：Open Controls → Animations → default → play。任务为导航、抓取、抬起、原地保持；不包含运输或放置。导航使用 OMPL 引导的旋转安全凸覆盖和 Drake GCS，操作使用 IRIS-NP2 / GCS。物理执行用双指接触，验证为有限时间采样。</p>
<p><a href="protocol.json">冻结协议</a> · <a href="comparison.json">完整对照结果</a> · <a href="semantic_review.html">实际执行语义逐帧复核</a></p>
<h2>全部尝试（保留失败）</h2><table><tr><th>策略</th><th>候选</th><th>底盘位姿</th><th>状态 / 原因</th></tr>{''.join(attempts)}</table>
<p>单场景诊断，不是成功率统计研究；超时、IK 或区域审计失败均不构成 C-space 不可行证明。本实验检验语义的作用，不单独证明 GCS 相对其他规划器的优势。</p></html>"""
    (root/'index.html').write_text(page)

if __name__=='__main__':main()
