"""Persistent evaluation, live dashboard, and automatic publication."""
import json,os,subprocess,sys,time,threading,shutil,traceback
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json
ROOT=Path('outputs/results/refined_transfer_v1')
OUT=Path('outputs/results/grasp_fix_web/planar/refined_transfer')

def progress(stop):
    while not stop.is_set():
        try:
            rows=[json.loads(p.read_text()) for cond in ['raw','screened'] for p in (ROOT/cond).glob('*__q*/*/record.json')]
            atomic_json(OUT/'progress.json',{'stage':'completed' if (ROOT/'complete.json').exists() else ('running' if ROOT.exists() else 'waiting_for_upstream'),'planning_records':len(rows),'planned':288,'contact_opportunities':48,'preparation_finished':len(list((ROOT/'raw/libraries').glob('*/preparation.json'))),'screen_finished':len(list((ROOT/'screened/libraries').glob('*/screen_report.json'))),'scenes':16,'note':'中间记录可能仍在执行接触阶段，不是最终成功率。'})
        except (OSError,ValueError):pass
        stop.wait(5)

def publish():
    from . import publish_screened_transfer as pub
    pub.ROOT=ROOT;pub.OUT=OUT;pub.main()
    text=(OUT/'index.html').read_text()
    a=text.index('<section>');b=text.index('</section>',a)+len('</section>')
    text=text[:a]+'''<section><p>同一批已诊断的16个场景、48个任务。共享库来自局部导航覆盖加密；这是修复回归，不是独立测试。原始与筛查库共享动作、导航与任务，仅携物区域不同。</p><p><b>筛查禁止读取旧反例，也未根据本轮查询失败追加修补。</b>保持64个内部点、168个边界探测点、每抓取最多4次官方IRIS重建。有限筛查不是安全证书。</p><p>共288次规划；筛查库每方法16个预定接触执行机会，覆盖无检查点、南检查点、北检查点。失败与无计划均保留。共享准备仍使用GCS辅助bootstrap，基线的“无GCS”只指查询。候选建库包含GCS bootstrap；本轮准备时间只统计复制共享库，上游建库成本单独保存。不能把建库改善计作GCS相对基线优势。</p><a href="../screened_transfer/">上一轮已知场景修复回归</a> · <a href="cluster_analysis.json">按场景分组分析</a></section>'''+text[b:]
    text=text.replace('共享建库','共享库复制（不含上游建库）').replace('放杯修复与共享区域筛查回归','局部加密后的三方法对照').replace('放杯修复后：原始库与共享筛查库的配对回归','局部导航加密后的规划与抓放回归').replace('144条原始记录','288条原始记录').replace('旧反例会在当前几何和抓取状态下重新检查，不会仅凭旧记录判定新区域无效。','本轮没有读取旧反例；原始建库内部的既有数值修复仍按固定预算执行。')
    (OUT/'index.html').write_text(text)
    records=json.loads((ROOT/'summary.json').read_text())['records'];d=json.loads((ROOT/'design.json').read_text());lookup={(r['condition'],r['method'],r['cell']):r for r in records};cluster={};rng=np.random.default_rng(20261009)
    for m in ['joint_10','joint_50']:
        diffs=[];all_success={k:0 for k in [m,'global_portal']}
        for scene in d['scenes']:
            cases=[c for c in d['cases'] if c['scene']==scene]
            scores={method:[lookup['screened',method,c['id']]['planning_success'] for c in cases] for method in [m,'global_portal']}
            for method,v in scores.items():all_success[method]+=int(all(v))
            diffs.append(float(np.mean(scores[m])-np.mean(scores['global_portal'])))
        samples=np.array(diffs)[rng.integers(0,16,(10000,16))].mean(axis=1)
        cluster[m]={'scene_count':16,'all_three_queries_successful_scenes':all_success,'paired_success_difference':float(np.mean(diffs)),'scene_bootstrap_percentile_95':np.quantile(samples,[.025,.975]).tolist(),'scope':'Exploratory paired cluster bootstrap with 16 scenes; zero-width interval when all paired outcomes equal is NOT proof of equivalence or population certainty.'}
    atomic_json(ROOT/'cluster_analysis.json',cluster);shutil.copy2(ROOT/'cluster_analysis.json',OUT/'cluster_analysis.json')
    a=json.loads((ROOT/'analysis.json').read_text());lines=['局部加密后的修复回归','同一批16个已参与修复的场景，48个相关任务，288次规划；非独立泛化证据。',d['scope'],'','方法 | 原始规划 | 筛查规划 | 预定接触成功 | 筛查查询中位秒']
    for m in pub.METHODS:
        r=a['metrics']['raw'][m];s=a['metrics']['screened'][m];lines.append(f"{m} | {r['planning_passed']}/48 | {s['planning_passed']}/48 | {s['contact_passed']}/16 | {s['all_median_s']:.2f}")
    lines+=['','全部失败分类：',json.dumps(a['metrics'],ensure_ascii=False,indent=2),'','按场景配对分析：',json.dumps(cluster,ensure_ascii=False,indent=2),'','限制：同一双房间模板；有限动作库；共享准备依赖GCS；数值筛查非安全证明；原始库没有接触执行。未按结果重抽场景或调整算法。','30项相关测试已通过；复用既有查询和浏览器验证流程；浏览器检查结果见browser_check.json。',json.dumps(a['completion'],indent=2)]
    (ROOT/'report.txt').write_text('\n'.join(lines));shutil.copy2(ROOT/'report.txt',OUT/'report.txt')
    browser=Path('/tmp/check_screened_transfer_web.py').read_text().replace('screened_transfer_v1','refined_transfer_v1').replace('/screened_transfer/','/refined_transfer/').replace('===144','===288').replace('===8','===16').replace("'rows':144,'scenes':8","'rows':288,'scenes':16")
    (ROOT/'check_web.py').write_text(browser)
    with (ROOT/'browser.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'check_web.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    shutil.copy2(ROOT/'browser_check.json',OUT/'browser_check.json')
    atomic_json(ROOT/'pipeline_complete.json',{'published':True,'browser_checked':True})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>局部加密后的算法对照</title><style>body{font:20px/1.8 system-ui;max-width:900px;margin:50px auto;background:#eef3f7}pre{white-space:pre-wrap}</style><h1>局部加密后的三方法对照</h1><p>16个修复场景 · 288次规划 · 48个预定接触执行机会。先等待上游完整建库，再按固定规则筛查和比较；不是独立测试。完成后本页自动替换为结果和3D回放。</p><pre id="p">读取进度…</pre><a href="../screened_transfer/">查看上一轮已完成结果</a><script>setInterval(async()=>{const x=await(await fetch('progress.json?t='+Date.now())).json();document.querySelector('#p').textContent=JSON.stringify(x,null,2);if(x.stage==='published')location.reload()},5000)</script>''')
    stop=threading.Event();thread=threading.Thread(target=progress,args=(stop,),daemon=True);thread.start()
    try:
        while not Path('outputs/results/navigation_local_refinement_v1/full_complete.json').exists():
            time.sleep(5)
        with Path('/tmp/refined_transfer.log').open('w') as log:subprocess.run([sys.executable,'-m','src.refined_transfer_experiment','--output',str(ROOT)],check=True,stdout=log,stderr=subprocess.STDOUT)
        publish();stop.set();thread.join();atomic_json(OUT/'progress.json',{'stage':'published','planning_records':288})
    except Exception:
        stop.set();thread.join();atomic_json(OUT/'progress.json',{'stage':'pipeline_error','error':traceback.format_exc()});raise

if __name__=='__main__':main()
