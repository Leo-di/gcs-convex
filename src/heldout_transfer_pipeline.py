"""Persistent evaluation, live dashboard, and automatic publication."""
import json,os,subprocess,sys,time,threading,shutil,traceback
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json
ROOT=Path('outputs/results/heldout_transfer_v1')
OUT=Path('outputs/results/grasp_fix_web/planar/heldout_transfer')

def progress(stop):
    while not stop.is_set():
        try:
            rows=[json.loads(p.read_text()) for cond in ['raw','screened'] for p in (ROOT/cond).glob('*__q*/*/record.json')]
            atomic_json(OUT/'progress.json',{'stage':'completed' if (ROOT/'complete.json').exists() else 'running','planning_records':len(rows),'planned':288,'contact_opportunities':48,'preparation_finished':len(list((ROOT/'raw/libraries').glob('*/preparation.json'))),'screen_finished':len(list((ROOT/'screened/libraries').glob('*/screen_report.json'))),'scenes':16,'note':'中间记录可能仍在执行接触阶段，不是最终成功率。'})
        except (OSError,ValueError):pass
        stop.wait(5)

def publish():
    from . import publish_screened_transfer as pub
    pub.ROOT=ROOT;pub.OUT=OUT;pub.main()
    text=(OUT/'index.html').read_text()
    a=text.index('<section>');b=text.index('</section>',a)+len('</section>')
    text=text[:a]+'''<section><p>16个预先固定的新随机组合场景、48个相关任务；门宽、推车、杯子、放置台同时变化，场景随机种子20261008。原始与筛查库共享动作、导航与任务，仅携物区域不同。</p><p><b>筛查禁止读取旧反例，也未根据本轮查询失败追加修补。</b>保持64个内部点、168个边界探测点、每抓取最多4次官方IRIS重建。有限筛查不是安全证书。</p><p>共288次规划；筛查库每方法16个预定接触执行机会，覆盖无检查点、南检查点、北检查点。失败与无计划均保留。共享准备仍使用GCS辅助bootstrap，基线的“无GCS”只指查询。属于同一双房间模板内迁移，不代表任意环境。</p><a href="../screened_transfer/">上一轮已知场景修复回归</a> · <a href="cluster_analysis.json">按场景分组分析</a></section>'''+text[b:]
    text=text.replace('放杯修复与共享区域筛查回归','新场景固定流程评测').replace('放杯修复后：原始库与共享筛查库的配对回归','新场景：无旧反例的共享区域筛查').replace('144条原始记录','288条原始记录').replace('旧反例会在当前几何和抓取状态下重新检查，不会仅凭旧记录判定新区域无效。','本轮没有读取旧反例；原始建库内部的既有数值修复仍按固定预算执行。')
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
    a=json.loads((ROOT/'analysis.json').read_text());lines=['固定流程的新场景评测','16个未参与修复的随机组合场景，48个相关任务，288次规划。',d['scope'],'','方法 | 原始规划 | 筛查规划 | 预定接触成功 | 筛查查询中位秒']
    for m in pub.METHODS:
        r=a['metrics']['raw'][m];s=a['metrics']['screened'][m];lines.append(f"{m} | {r['planning_passed']}/48 | {s['planning_passed']}/48 | {s['contact_passed']}/16 | {s['all_median_s']:.2f}")
    lines+=['','全部失败分类：',json.dumps(a['metrics'],ensure_ascii=False,indent=2),'','按场景配对分析：',json.dumps(cluster,ensure_ascii=False,indent=2),'','限制：同一双房间模板；有限动作库；共享准备依赖GCS；数值筛查非安全证明；原始库没有接触执行。未按结果重抽场景或调整算法。','27项相关测试在启动前通过；浏览器检查结果见browser_check.json。',json.dumps(a['completion'],indent=2)]
    (ROOT/'report.txt').write_text('\n'.join(lines));shutil.copy2(ROOT/'report.txt',OUT/'report.txt')
    browser=Path('/tmp/check_screened_transfer_web.py').read_text().replace('screened_transfer_v1','heldout_transfer_v1').replace('/screened_transfer/','/heldout_transfer/').replace('===144','===288').replace('===8','===16').replace("'rows':144,'scenes':8","'rows':288,'scenes':16")
    (ROOT/'check_web.py').write_text(browser)
    with (ROOT/'browser.log').open('w') as log:subprocess.run([sys.executable,str(ROOT/'check_web.py')],check=True,stdout=log,stderr=subprocess.STDOUT)
    shutil.copy2(ROOT/'browser_check.json',OUT/'browser_check.json')
    atomic_json(ROOT/'pipeline_complete.json',{'published':True,'browser_checked':True})

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>新场景评测进行中</title><style>body{font:20px/1.8 system-ui;max-width:900px;margin:50px auto;background:#eef3f7}pre{white-space:pre-wrap}</style><h1>新场景固定流程评测</h1><p>16个新场景 · 288次规划 · 48个预定接触执行机会。禁止旧反例；不按结果换场景。完成后本页自动替换为结果和3D回放。</p><pre id="p">读取进度…</pre><a href="../screened_transfer/">查看上一轮已完成结果</a><script>setInterval(async()=>{const x=await(await fetch('progress.json?t='+Date.now())).json();document.querySelector('#p').textContent=JSON.stringify(x,null,2);if(x.stage==='published')location.reload()},5000)</script>''')
    stop=threading.Event();thread=threading.Thread(target=progress,args=(stop,),daemon=True);thread.start()
    try:
        with Path('/tmp/heldout_transfer.log').open('w') as log:subprocess.run([sys.executable,'-m','src.heldout_transfer_experiment','--output',str(ROOT)],check=True,stdout=log,stderr=subprocess.STDOUT)
        publish();stop.set();thread.join();atomic_json(OUT/'progress.json',{'stage':'published','planning_records':288})
    except Exception:
        stop.set();thread.join();atomic_json(OUT/'progress.json',{'stage':'pipeline_error','error':traceback.format_exc()});raise

if __name__=='__main__':main()
