"""Publish the solver-removal pilot without hiding strong baseline results."""
import json
from pathlib import Path
import statistics

from pydrake.geometry import Box,Rgba
from pydrake.math import RigidTransform

from .ablate_gcs_solver import OUTPUT
from .arm_boundary_gcs import export_replay
from .build_scene import build_scene


def main():
    rows=json.loads((OUTPUT/'summary.json').read_text())
    positive=[r for r in rows if r['case']['expected']=='positive']
    gcs={r['case']['name']:r['result']['metric_length'] for r in positive if r['method']=='gcs'}
    table='';aggregate={}
    for method in dict.fromkeys(r['method'] for r in rows):
        values=[r for r in positive if r['method']==method]
        excess=statistics.median(100*(r['result']['metric_length']/gcs[r['case']['name']]-1) for r in values)
        seconds=statistics.median(r['result']['online_seconds'] for r in values)
        count=sum(r['audit']['accepted'] for r in values)
        aggregate[method]=dict(valid=count,total=len(values),median_length_excess_percent=excess,median_online_seconds=seconds)
        table+=f'<tr><td>{method}</td><td>{count}/{len(values)}</td><td>{excess:.4f}%</td><td>{seconds*1000:.2f} ms</td></tr>'
    (OUTPUT/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    web=Path('outputs/results/grasp_fix_web/planar/gcs_role');web.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())['config']
    scene=build_scene(config=config,visualize=True)
    scene.meshcat.SetObject('semantic_boundary',Box(.008,1.2,2.2),Rgba(1,.15,.15,.22))
    scene.meshcat.SetTransform('semantic_boundary',RigidTransform([0,0,1.1]))
    scene.meshcat.SetCameraPose([.4,-1.1,5.],[-.4,0,.9])
    for method in ['gcs','dijkstra_1','dijkstra_1_socp']:
        trace=json.loads((OUTPUT/'free_gate-0.6'/method/'trace.json').read_text())
        export_replay(scene,trace,web/f'{method}.html')
    (web/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>GCS 作用的论文核读与去除对照</title>
<style>body{font:17px system-ui;margin:24px;color:#172b4d;max-width:1400px}p{line-height:1.6}button{font:inherit;padding:12px;margin:5px}table{border-collapse:collapse}td,th{padding:10px;border:1px solid #ccd}iframe{width:100%;height:650px}.note{padding:12px;background:#eef5fc}</style>
<h1>移除 GCS 之后结果如何变化</h1>
<p>同一三维场景、同一凸区域覆盖、同一全臂语义、相同起终点和构型路径长度。对照依次为 GCS、区域交叠点的 Dijkstra 搜索、Dijkstra 后固定区域序列的连续凸优化。后两者不调用 GCS。</p>
<p class="note"><b>当前结论：成功率均为 6/6；离散搜索加连续优化能匹配 GCS 的路径质量，并且在这组小问题上更快。不能声称已经证明 GCS 优于强基线。</b></p>
<table><tr><th>方法</th><th>正例参考轨迹通过</th><th>相对 GCS 路径增长中位数</th><th>建图加求解中位数</th></tr>'''+table+'''</table>
<p>1、3、9 表示每个非退化交叠坐标的离散点数。SOCP 是固定区域序列上的连续优化，不搜索区域序列。时间不含共用覆盖生成与独立审核；GCS 额外重解松弛的诊断时间单独记录。单轮小样本时间只能作初步诊断。</p>
<button onclick="show('gcs')">GCS</button><button onclick="show('dijkstra_1')">粗离散搜索</button><button onclick="show('dijkstra_1_socp')">离散搜索加连续优化</button>
<p id="label">GCS 参考轨迹</p><iframe id="scene" src="gcs.html"></iframe>
<p>动画只展示两侧均允许、切换位置 x=−0.6 的代表条件，是规划参考轨迹回放，不是本轮新的动力学执行。仍采用两个受限收臂姿态族。</p>
<h2>论文与当前差距</h2>
<p><a href="https://arxiv.org/abs/2205.04422">Motion Planning around Obstacles with Convex Optimization</a> 提供联合区域选择和连续轨迹优化框架；<a href="https://arxiv.org/abs/2303.14737">Growing Convex Collision-Free Regions in Configuration Space using Nonlinear Programming</a> 负责凸区域生成及额外非线性约束。用户提供的导师原话已确认这两篇；当前优先深入理解 C-space，GCS 掌握基本概念，进一步研究计算方法为可选项。</p>
<p>先前的分支枚举内部也使用 GCS，因此它是联合与分开求解的对照。本轮才移除了 GCS 求解器。新的离散方法是本项目机制消融，不是论文 PRM 实现的复现。</p>
<p>已保存 GCS 的松弛与取整目标值，但存在约 10⁻⁷ 量级的相对数值倒挂，不能把它报告成严格的零最优性间隙。首次诊断暴露的切换平面边界问题已存档；所有方法共享相同的 0.1 mm 保守修正后重跑。</p>
<script>function show(x){document.getElementById('scene').src=x+'.html';document.getElementById('label').textContent=x+' 参考轨迹'}</script>''')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
