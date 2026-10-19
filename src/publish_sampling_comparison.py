"""Publish OMPL comparison and the independently labeled RRT/IRIS/GCS hybrid."""
import json
from pathlib import Path
import numpy as np


def main():
    root=Path('outputs/results/ompl_room_fine_20261019')
    coarse=Path('outputs/results/ompl_room_comparison_20261019');old=Path('outputs/results/gcs_multiroute_20261019');hybrid=Path('outputs/results/rrt_guided_iris_refined2_20261019')
    rr=json.loads((root/'comparison.json').read_text());gcs=json.loads((old/'comparison.json').read_text())
    audits=json.loads((root/'common_fine_audit.json').read_text());coarse_audit=json.loads((coarse/'common_fine_audit.json').read_text())
    coverage=json.loads((root/'coverage_diagnosis.json').read_text())
    g=gcs[0]['results'][0];r=next(x for x in rr if x['query_id']==0 and x['seed']==100)
    h=json.loads((hybrid/'comparison.json').read_text())[0]['results'][0];hs=json.loads((hybrid/'summary.json').read_text())
    gp=json.loads((old/'execution_comparison.json').read_text())[0]
    rp=json.loads((root/'physical_execution/execution_comparison.json').read_text())[0]
    hp=json.loads((hybrid/'execution_comparison.json').read_text())[0]
    attempts=[]
    for folder,padding in [('rrt_guided_iris_20261019',.001),('rrt_guided_iris_refined_20261019',.0001),('rrt_guided_iris_refined2_20261019',.00001)]:
        audit=json.loads((Path('outputs/results')/folder/'strict_path_audit.json').read_text())[0]
        attempts.append({'folder':folder,'padding_rad':padding,'accepted':audit['accepted'],'violations':audit['violations']})
    assert attempts[-1]['accepted']
    counts={'coarse_rrt_valid':sum(x['accepted'] for x in coarse_audit if x['method']=='ompl_rrtconnect'),
            'fine_rrt_valid':sum(x['accepted'] for x in audits if x['method']=='ompl_rrtconnect'),
            'gcs_valid':sum(x['accepted'] for x in audits if x['method']=='gcs')}
    summary={'counts':counts,'coverage_diagnosis':coverage,'full_query':{'gcs':g,'rrt':r,'hybrid':h},
             'physical':{'gcs':gp,'rrt':rp,'hybrid':hp},'hybrid_cover':hs,'hybrid_attempts':attempts,
             'scope':'Single scene; 5 correlated endpoint queries, 3 RRT seeds; no statistical GCS superiority claim. Geometry and semantics shared, feasible representations differ. Hybrid uses RRT route discovery.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2))
    trs=[]
    for i in range(5):
        vals=[x for x in rr if x['query_id']==i]
        gg=next(x for x in gcs[i]['results'] if x['method']=='gcs')
        length=[x['path_length_rad'] for x in vals]
        planning=[x['setup_seconds']+x['solve_seconds']+x['simplify_seconds'] for x in vals]
        passed=sum(x['accepted'] for x in audits if x['method']=='ompl_rrtconnect' and x['query_id']==i)
        trs.append(f'<tr><td>{i}</td><td>{gg["path_length_rad"]:.4f}</td><td>{min(length):.4f}–{max(length):.4f}</td><td>{gg["build_seconds"]+gg["solve_seconds"]:.2f}</td><td>{min(planning):.2f}–{max(planning):.2f}</td><td>{passed}/3</td></tr>')
    physical=[]
    for label,record,phys,url,extra in [
        ('原引导覆盖 + GCS',g,gp,'../gcs_multiroute_20261019/execution_gcs/scene.html','共享覆盖额外约 179.23 秒，另复用原覆盖'),
        ('官方 RRT-Connect + 简化',r,rp,'physical_execution/execution_ompl_rrtconnect/scene.html',f'规划与简化合计 {r["setup_seconds"]+r["solve_seconds"]+r["simplify_seconds"]:.2f} 秒'),
        ('RRT 引导 → IRIS → GCS',h,hp,'../rrt_guided_iris_refined2_20261019/execution_gcs/scene.html',f'另加 IRIS/建图/审计 {hs["cover_seconds"]:.2f} 秒和 GCS 求解 {h["solve_seconds"]:.2f} 秒')]:
        physical.append(f'<tr><td>{label}</td><td>{record["path_length_rad"]:.4f}</td><td>{phys["accepted"]}</td><td>{phys["unexpected_contacts"]}</td><td>{extra}</td><td><button data-url="{url}">载入</button> <a href="{url}" target="_blank">打开动画</a></td></tr>')
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>成熟采样基线与 GCS 覆盖诊断</title><style>body{{font:17px system-ui;max-width:1250px;margin:32px auto;padding:20px;color:#18344a;background:#f4f7fa}}section{{padding:24px;background:white;border-radius:12px;margin:20px 0}}table{{width:100%;border-collapse:collapse}}td,th{{padding:10px;text-align:left;border-bottom:1px solid #ddd}}.note{{padding:20px;background:#fff1cd}}a{{color:#1759a0}}iframe{{width:100%;height:700px;border:0}}button{{padding:8px}}</style><h1>成熟采样基线与 GCS 区域覆盖诊断</h1><p class="note">本轮没有证明 GCS 成功率优势。官方 RRT-Connect 找到更短的合法路径并完成真实抓取；其路径约 {coverage['sampled_arc_length_outside_fraction']*100:.2f}% 的弧长采样位于原 GCS 区域覆盖之外。随后用 RRT 路径生成新 IRIS 覆盖并交给 GCS，直接检验覆盖限制。</p><section><h2>固定场景与独立检查</h2><p>同一真实七关节模型、相同起终点、整机入室语义和碰撞模型。五个固定查询 × 三个随机种子；预先指定查询 0、种子 100 做接触执行。OMPL 使用全关节界，不限制在旧 IRIS 通道中。</p><p>粗分辨率轮 {counts['coarse_rrt_valid']}/15 通过；细分辨率轮 {counts['fine_rrt_valid']}/15 通过；缓存 GCS 五条轨迹经同一细检查 {counts['gcs_valid']}/5 通过。粗轮两处指尖穿透约 29–41 微米，失败保留。细轮规划步长 0.00125 rad，所有方法最终按最大关节步长 0.00125 rad 重新审计；仍非连续安全证明。</p></section><section><h2>五个查询</h2><p>路径长度采用关节欧氏弧长。RRT 时间包含模型设置、规划和实际后处理时间；GCS 时间包含图构造和求解，不含共享区域生成。RRT 简化器的 1 秒请求是软终止，单次操作会超时，因此全部报告真实耗时。</p><table><tr><th>查询</th><th>GCS 长度</th><th>RRT 三种子长度范围</th><th>GCS 秒</th><th>RRT 秒范围</th><th>RRT 细审计</th></tr>{''.join(trs)}</table></section><section><h2>完整抓取与混合方法</h2><p>以下均包含相同导航、闭爪和抬升阶段，仅替换接近轨迹。执行使用自由杯体接触动力学，未用 attachment 替代执行。</p><table><tr><th>方法</th><th>接近长度</th><th>执行验收</th><th>非预期接触</th><th>成本口径</th><th>动画</th></tr>{''.join(physical)}</table><iframe id="viewer" src="physical_execution/execution_ompl_rrtconnect/scene.html" title="真实抓取动画"></iframe></section><section><h2>区域修复与失败记录</h2><p>混合方法的引导域 padding 依次为 0.001、0.0001、0.00001 rad。前两轮严格审计分别发现 4 个和 1 个违规样本，均判失败；最终轮通过。初版虽通过旧参考检查及物理回放，仍被严格路径审计否决。已加入执行前严格拦截，并用原失败路径验证它不会进入物理执行。</p><p>最终保留 125 个七维 IRIS 区域，未用手写路径替代 GCS。通道已收得很窄，不能把这说成自由探索。上表只列最终成功轮的新增覆盖成本；此前两次失败另有成本，不是零成本或从零总耗时。</p><p><a href="../rrt_guided_iris_20261019/strict_path_audit.json">初版失败</a> · <a href="../rrt_guided_iris_refined_20261019/strict_path_audit.json">第一次收紧失败</a> · <a href="../rrt_guided_iris_refined2_20261019/strict_path_audit.json">最终严格审计</a> · <a href="rejection_guard_regression/execution_comparison.json">执行前拒绝回归</a></p></section><section><h2>结论范围</h2><p>这些结果支持“当前引导生成的区域覆盖限制了 GCS 的路径质量”。它们不证明 GCS 本身无用，也不证明 RRT 在所有任务上更好。混合方法中的路线发现来自 RRT，不能把它计为 GCS 独立发现路线的成功。五个查询来自同一场景，不是独立场景的成功率试验。</p><p>此前目标构型仍由外层候选搜索选择；本轮未把这一选择改成 GCS 联合优化，也没有声称实现自动发现缺失语义。</p></section><p><a href="summary.json">全部汇总</a> · <a href="protocol.json">固定方案</a> · <a href="common_fine_audit.json">统一细审计</a> · <a href="../ompl_room_comparison_20261019/common_fine_audit.json">粗轮反例</a> · <a href="coverage_diagnosis.json">覆盖诊断</a> · <a href="source_manifest.json">代码快照</a> · <a href="https://ompl.kavrakilab.org/">官方 OMPL</a></p><script>document.querySelectorAll('button[data-url]').forEach(b=>b.onclick=()=>document.getElementById('viewer').src=b.dataset.url);</script></html>'''
    (root/'index.html').write_text(text)
    print(counts);print('lengths',g['path_length_rad'],r['path_length_rad'],h['path_length_rad']);print('physical',gp['accepted'],rp['accepted'],hp['accepted'])

if __name__=='__main__':main()
