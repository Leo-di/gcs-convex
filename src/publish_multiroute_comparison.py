"""Publish measured paired results without claiming independent trial statistics."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from .compare_frozen_operation_graph import minimum_hop_route


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    rows=json.loads((root/'comparison.json').read_text());executions=json.loads((root/'execution_comparison.json').read_text())
    summary=json.loads((root/'summary.json').read_text());branches=json.loads((root/'branches.json').read_text())
    graph=json.loads((root/'query_00/reach_cover.json').read_text());edges=graph['edges']
    first=minimum_hop_route(edges,graph['sources'],graph['targets']);alternative=None
    for x,y in zip(first,first[1:]):
        cut=[e for e in edges if tuple(e) not in ((x,y),(y,x))]
        second=minimum_hop_route(cut,graph['sources'],graph['targets'])
        if second is not None:
            alternative=second;break
    summary['multiple_routes_verified']=alternative is not None
    summary['witness_routes']=[first,alternative]
    labels={'gcs':'GCS','minimum_hops_then_convex':'最少区域数 + 凸优化','center_weighted_then_convex':'几何加权选路 + 凸优化'}
    metrics=[];trs=[]
    for row in rows:
        baseline={r['method']:r for r in row['results']}
        costs=[r.get('objective',float('nan')) for r in row['results']]
        item={'query_id':row['query_id'],'methods':{}}
        for r in row['results']:
            validity=False
            if r.get('numerically_valid'):
                speed=json.loads((Path(row['input'])/'scene_config_snapshot.json').read_text())['whole_body_planning']['maximum_joint_speed']
                errors=[max(0.,float(np.max(abs(np.array(s['end'])-s['start'])))-speed*s['duration']) for s in r['segments']]
                objective_error=abs(r['objective']-r['path_length_rad']-.05*r['duration_s'])
                validity=max(errors)<1e-6 and objective_error<1e-5
                r['independent_velocity_excess_rad']=max(errors);r['objective_reconstruction_error']=objective_error
            item['methods'][r['method']]={'valid':validity,'objective':r.get('objective'),'solve_seconds':r['solve_seconds']}
            trs.append(f'<tr><td>{row["query_id"]}</td><td>{labels[r["method"]]}</td><td>{"通过" if validity else "失败"}</td><td>{r.get("objective",float("nan")):.6f}</td><td>{r["solve_seconds"]:.4f}</td></tr>')
        metrics.append(item)
    summary['queries']=metrics
    (root/'analysis.json').write_text(json.dumps(summary,indent=2));(root/'comparison.json').write_text(json.dumps(rows,indent=2))
    physical=[]
    for r in executions:
        method=r['method'];url=f'execution_{method}/scene.html'
        physical.append(f'<tr><td>{labels[method]}</td><td>{r["accepted"]}</td><td>{r.get("unexpected_contacts","—")}</td><td>{r.get("room_min_m",float("nan")):.5f}</td><td><button data-url="{url}">载入动画</button> <a href="{url}" target="_blank">独立打开</a></td></tr>')
    wins=losses=ties=0
    for row in rows:
        m={r['method']:r for r in row['results']}
        g=m['gcs']['objective'];b=m['center_weighted_then_convex']['objective']
        if abs(g-b)<=1e-5*max(1.,abs(b)):ties+=1
        elif g<b:wins+=1
        else:losses+=1
    conclusion=f'相对几何加权对照，GCS 在 {wins}/5 个查询代价更低、{losses}/5 个更高、{ties}/5 个近似相同（相对阈值 1e-5）。本轮没有证明成功率优势；GCS 求解耗时明显更高。'
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>真实机械臂多路线 GCS 对照</title><style>body{{font:17px system-ui;color:#18344a;background:#f4f7fa;max-width:1200px;margin:30px auto;padding:24px}}section{{background:white;border-radius:12px;padding:22px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #ddd}}.note{{padding:18px;background:#fff1cd}}a{{color:#1760a0}}iframe{{width:100%;height:720px;border:0}}button{{padding:8px;cursor:pointer}}</style><h1>真实 7 关节场景：多路线区域与执行对照</h1><p class="note">{conclusion}</p><p>场景、抓取目标、整机入室语义不变。新增四条预先确定的关节引导扰动，以官方 IRIS-NP2 生成区域并独立审计；三种方法共享全部合格区域。</p><section><h2>区域图</h2><p>合格新增通道 {summary['accepted_branches']}/4；共 {summary['regions']} 个区域、{summary['undirected_edges']} 条无向交叠边；多条起终点路线核查：{summary['multiple_routes_verified']}。</p><p>新增区域生成、审计及建图耗时 {summary['cover_seconds']:.2f} 秒，另复用原始覆盖的生成成本。该成本由各方法共同承担，不能将下表的求解时间说成从零端到端耗时。</p></section><section><h2>同图同约束求解</h2><p>代价 = 关节空间路径长度 + 0.05 × 规划时长。均使用官方 Drake/Clarabel；两种固定路线对照都优化连续连接点，中心仅用于几何路线评分。</p><table><tr><th>查询</th><th>方法</th><th>数值验证</th><th>目标值</th><th>求解秒</th></tr>{''.join(trs)}</table></section><section><h2>预先指定查询 0 的完整接触执行</h2><p>沿用相同导航、闭爪和抬升段，仅替换接近段。真实自由杯体 SAP 动力学，无 attachment 替代执行。</p><table><tr><th>方法</th><th>完整验收</th><th>非预期接触</th><th>最小房间 residual (m)</th><th>动画</th></tr>{''.join(physical)}</table><iframe id="viewer" src="execution_gcs/scene.html" title="执行动画"></iframe></section><p class="note">五个相关查询来自一个场景，并非独立成功率样本。新增通道是局部引导扰动，不代表发现不同绕障同伦类。当前是 GCS 选路机制消融，不是与成熟 RRT-Connect 的完整基准。部分 IRIS 内部非线性子问题发出未收敛警告，原始日志保留。独立审计通过不等于形式化区域认证；IRIS 及接触审计属于数值验证，非连续时间安全证明。执行参考经过统一重采样，规划时长不等于真实执行时长。</p><p><a href="comparison.json">完整规划数据</a> · <a href="execution_comparison.json">真实执行验收</a> · <a href="analysis.json">拓扑与数值复核</a> · <a href="branches.json">全部通道尝试及失败</a> · <a href="paired_protocol.json">预先指定的对照方案</a> · <a href="source_manifest.json">代码快照</a></p><script>document.querySelectorAll('button[data-url]').forEach(b=>b.onclick=()=>document.getElementById('viewer').src=b.dataset.url);</script></html>'''
    (root/'index.html').write_text(text)
    print(json.dumps({'regions':summary['regions'],'edges':summary['undirected_edges'],'multiple_routes':summary['multiple_routes_verified'],'queries':metrics,'execution':executions},indent=2))

if __name__=='__main__':main()
