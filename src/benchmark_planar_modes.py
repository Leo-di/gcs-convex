"""Run deterministic planar ablations and publish a comparison page."""
from pathlib import Path
import argparse
import json

from .planar_bubble_demo import solve


TITLES = {
    "full": "完整模型：任务顺序 + 携物尺寸",
    "no_task": "对照 A：忽略取物要求",
    "ignore_payload": "对照 B：忽略携物尺寸",
    "blocked": "负例：关闭宽门",
}
EXPLANATIONS = {
    "full": "空手经过窄门，携物经过宽门；路径必须同时满足任务顺序和实际尺寸。",
    "no_task": "仅求起点到终点的空手路径。即使到达终点，也没有取物，因此任务失败。",
    "ignore_payload": "仍然先取后放，但规划错误地沿用空手半径。橙色阶段按真实携物尺寸显示，独立检查会发现门框碰撞。该路径仅用于展示错误，不可执行。",
    "blocked": "关闭宽门后，仅剩 0.70 m 窄门。携物直径 0.84 m，无法从右房间回到左房间。此案例有独立几何不可达依据，未尝试 GCS 求解，也不显示虚构的完成轨迹。",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/results/planar_benchmark_staging"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    template = Path(__file__).with_name("planar_bubble_view.html").read_text()
    results = {}
    for case in TITLES:
        print(f"Running {case}", flush=True)
        result = solve(case)
        results[case] = result
        destination = args.output/case
        destination.mkdir(exist_ok=True)
        (destination/"result.json").write_text(json.dumps(result, indent=2))
        if result["trace"]:
            html = template.replace("__DATA__", json.dumps(result))
            (destination/"index.html").write_text(html)
        print(json.dumps({"case": case, "status": result["status"], "audit": result["audit"]}), flush=True)

    # Acceptance expectations, rather than treating any returned path as success.
    assert results["full"]["audit"]["accepted"]
    assert not results["no_task"]["audit"]["task_order"]
    assert results["ignore_payload"]["audit"]["task_order"]
    assert not results["ignore_payload"]["audit"]["actual_body_safe"]
    assert results["blocked"]["status"] == "geometrically_infeasible"
    (args.output/"benchmark.json").write_text(json.dumps(results, indent=2))
    entries = {k: {"title": TITLES[k], "explanation": EXPLANATIONS[k], **v} for k, v in results.items()}
    dashboard = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>语义 GCS 对照实验</title>
<style>body{font:16px system-ui;margin:0;background:#f1f5f9;color:#172b43}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:27px}p{line-height:1.7}select{font:inherit;padding:10px;max-width:100%}table{width:100%;border-collapse:collapse;background:white;margin:20px 0}td,th{text-align:left;padding:12px;border-bottom:1px solid #dbe4ed}iframe{width:100%;height:1420px;border:0}#notice{background:white;padding:18px;border-left:5px solid #087a85}#blocked{padding:30px;background:white}svg{width:100%;max-height:500px}a{color:#087a85}</style>
<main><h1>语义与携物几何：四组对照实验</h1><p>同一地图、固定起终点与取物点。检查“求到路径”是否真的意味着任务完成。当前是固定动作点处分解的 GCS 基线，不是通用任务图联合求解器。</p>
<table><thead><tr><th>实验</th><th>路径 / 可达性</th><th>任务顺序</th><th>实际尺寸安全</th><th>任务接受</th></tr></thead><tbody id="rows"></tbody></table>
<label>查看实验 <select id="case"></select></label><p id="notice"></p><iframe id="viewer" title="选中实验的路径和圆盘"></iframe><div id="blocked" hidden><h2>携物任务不可达</h2><p>分隔墙横截面 x=0 仅剩 y∈(-0.35,0.35) 开口。任何左右房间间连续路径都必须通过该截面，而携物圆直径 0.84 m 大于开口 0.70 m。上下边界阻止绕过分隔墙。</p><svg id="map" viewBox="-4.3 -3.3 7.6 6.6"></svg><p>图不连通通常只能说明当前覆盖未找到路径；这里额外的截面几何论证才支持“场景不可达”。没有求解或执行部分取物路径。</p></div><p><a href="benchmark.json">完整实验数据</a> · 单场景确定性实验；各组均重新求解，不代表统计性能比较。</p></main>
<script>const cases=__CASES__,select=document.getElementById('case');function check(v){return v===null?'未执行':v?'通过':'失败'}
for(const[k,v]of Object.entries(cases)){const o=document.createElement('option');o.value=k;o.textContent=v.title;select.appendChild(o);const row=document.createElement('tr');for(const t of [v.title,v.status==='solved'?'已求得路径':'几何不可达',check(v.audit.task_order),check(v.audit.actual_body_safe),v.audit.accepted?'接受':'拒绝']){const c=document.createElement('td');c.textContent=t;row.appendChild(c)}document.getElementById('rows').appendChild(row)}
function update(){const k=select.value,v=cases[k],blocked=k==='blocked';document.getElementById('notice').textContent=v.explanation;document.getElementById('viewer').hidden=blocked;document.getElementById('blocked').hidden=!blocked;document.getElementById('viewer').src=blocked?'about:blank':k+'/index.html';if(blocked){const svg=document.getElementById('map');svg.replaceChildren();function el(tag,a){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);svg.appendChild(e)}el('rect',{x:-4,y:-3,width:7,height:6,fill:'#fafbfc',stroke:'#718096','stroke-width':.03});for(const[a,b,c,d]of v.obstacles)el('rect',{x:a,y:-d,width:c-a,height:d-b,fill:'#69798c'});el('circle',{cx:v.pickup[0],cy:-v.pickup[1],r:.42,fill:'#e77820','fill-opacity':.5});el('circle',{cx:v.dropoff[0],cy:-v.dropoff[1],r:.08,fill:'#13995e'});}}
select.onchange=update;update();</script></html>'''
    (args.output/"index.html").write_text(dashboard.replace("__CASES__", json.dumps(entries)))
    print(args.output/"index.html")


if __name__ == "__main__":
    main()
