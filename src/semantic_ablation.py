"""Frozen, matched three-policy experiment. Failed runs are retained."""
import argparse
import concurrent.futures
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

from .scene_spec import load_scene_config
from .semantic_refinement import generate_candidates
from .semantic_policy import POLICIES


def write_live_review(output):
    """Read saved worker summaries; never infer a pass from a live animation."""
    links=''.join(f'<p><a href="{p}/">{p}: 候选、失败记录与完成后的动画</a></p>' for p in POLICIES)
    (Path(output)/'index.html').write_text('''<!doctype html><meta charset="utf-8">
<title>同场景语义对照 · 运行状态</title><style>body{font:17px system-ui;margin:32px;color:#203449}pre{white-space:pre-wrap}</style>
<h1>同场景语义对照</h1><p>自动读取实际结果，每 10 秒更新。运行中不代表规划或物理执行通过。</p>
<p>共同场景、候选顺序、碰撞与物理模型、规划器和预算；只改变语义约束作用范围。</p>
<pre id="status">读取状态…</pre>'''+links+'''
<p><a href="protocol.json">冻结协议</a></p>
<script>
async function update(){
const rows=await Promise.all(['geometry_only','goal_only','full_motion'].map(async method=>{
try{const response=await fetch(method+'/summary.json',{cache:'no-store'});if(!response.ok)throw Error(response.status);
const s=await response.json(),r=s.selected_result||{};
const active=await Promise.all((s.candidates||[]).filter(x=>x.status==='planning').map(async x=>{
try{const p=await (await fetch(method+'/'+x.id+'/progress.json',{cache:'no-store'})).json();return {candidate:x.id,stage:p.stage};}
catch(e){return {candidate:x.id,stage:'等待阶段记录'}}}));
return {method,status:s.status,attempts:s.attempt_count,selected:s.selected_candidate,
method_success:r.satisfied??null,semantic_task_success:r.semantic_task_satisfied??null,
failures:(s.candidates||[]).filter(x=>x.failure_stage).reduce((a,x)=>{a[x.failure_stage]=(a[x.failure_stage]||0)+1;return a},{}),active};
}catch(e){return {method,status:'等待结果文件',detail:String(e)}}}));
document.getElementById('status').textContent=JSON.stringify(rows,null,2);}
update();setInterval(update,10000);
</script>''',encoding='utf-8')


def run(config_path, output, dynamics=True):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    cfg=load_scene_config(config_path)
    candidates=generate_candidates(cfg)
    protocol={'common_config':cfg,'candidates':candidates,'policies':POLICIES,'dynamics':dynamics,
              'scope':'One fixed scene, shared candidate ordering and per-policy budgets; no success-rate claim',
              'common_config_sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    write_live_review(output)
    def worker(policy):
        c=copy.deepcopy(cfg);c['semantic_policy']=policy
        path=output/f'{policy}.json';path.write_text(json.dumps(c,indent=2))
        command=[sys.executable,'-m','src.semantic_refinement','--config',str(path),'--output',str(output/policy)]
        if not dynamics:command+=['--no-dynamics','--no-visualize']
        with (output/f'{policy}.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',LCM_DEFAULT_URL='memq://'),check=False)
        p=output/policy/'summary.json'
        s=json.loads(p.read_text()) if p.exists() else {}
        r=s.get('selected_result',{})
        return {'method':policy,'exit_code':result.returncode,'status':s.get('status','worker_failed'),
                'attempt_count':s.get('attempt_count'), 'candidate':s.get('selected_candidate'),
                'dock':r.get('dock_q'),'method_success':r.get('satisfied',False),
                'semantic_task_success':r.get('semantic_task_satisfied',False),
                'physics':r.get('physics'), 'elapsed_s':s.get('elapsed_s')}
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(worker,p):p for p in POLICIES}
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result());(output/'comparison.json').write_text(json.dumps(rows,indent=2))
            print(rows[-1],flush=True)
    rows.sort(key=lambda r:POLICIES.index(r['method']))
    (output/'comparison.json').write_text(json.dumps(rows,indent=2))
    cells=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k)))+'</td>' for k in
                  ['method','attempt_count','dock','method_success','semantic_task_success','status'])+'</tr>' for r in rows)
    links=''.join(f'<p><a href="{p}/">{p}: 动画与完整候选失败记录</a></p>' for p in POLICIES)
    (output/'index.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>Semantic ablation</title>
<style>body{{font:17px system-ui;margin:32px;color:#203449}}td,th{{padding:12px;border:1px solid #ccd}}table{{border-collapse:collapse}}</style>
<h1>同场景语义对照</h1><p>共同场景、候选顺序、碰撞与物理模型、规划器和预算；只改变语义约束的作用范围。</p>
<p>方法成功不等于满足教授的完整任务。失败与超时均保留，不自动判为任务不可行。</p>
<table><tr><th>方法</th><th>完整尝试</th><th>停靠配置</th><th>方法成功</th><th>统一任务语义成功</th><th>状态</th></tr>{cells}</table>{links}
<p><a href="protocol.json">冻结协议及全部候选</a> · <a href="comparison.json">结果</a></p>
<p>单场景诊断，不是多场景成功率结论；goal-only 与 full-motion 若同样成功，也如实报告。</p>''',encoding='utf-8')
    return rows

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/scene_semantic_choice_v2.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--no-dynamics',action='store_true')
    args=parser.parse_args();run(args.config,args.output,not args.no_dynamics)
