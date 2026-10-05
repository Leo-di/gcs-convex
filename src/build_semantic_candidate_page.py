"""Display finite grasp-candidate choices and their actual execution evidence."""
import json
from pathlib import Path
import shutil


def main():
    root=Path('outputs/results/semantic_candidates');out=Path('outputs/results/grasp_fix_web/planar/candidates');out.mkdir(parents=True,exist_ok=True)
    results=json.loads((root/'results.json').read_text());library=json.loads((root/'library/report.json').read_text())
    config=json.loads((root/'library/config.json').read_text())
    data={'results':results,'library':library,'config':config,'traces':{},'disks':json.loads(Path('outputs/results/semantic_joint_gcs/shared_model.json').read_text())['empty_cover']}
    for case,r in results.items():
        if r['status']!='solved':continue
        rows=json.loads((root/case/'delivery_trace.json').read_text())
        data['traces'][case]=[{'xy':x['base_q'][:2],'attached':x['attached']} for x in rows]
        shutil.copy2(root/case/'scene.html',out/(case+'.html'))
        f=root/(case+'_dynamics')/'result.json'
        if f.exists():
            raw=json.loads(f.read_text());r['physics']={k:v for k,v in raw.items() if k!='trace'}
            shutil.copy2(f,out/(case+'_dynamics.json'));shutil.copy2(f.parent/'scene.html',out/(case+'_dynamics.html'))
    (out/'results.json').write_text(json.dumps(data,ensure_ascii=False))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GCS 选择在哪里抓取</title><style>body{margin:0;background:#f1f5f9;color:#24364c;font:16px/1.7 system-ui}main{max-width:1180px;padding:24px;margin:auto}.card{padding:20px;margin:16px 0;background:white;border-radius:12px}h1{font-size:28px}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce4ec}select,button{font:inherit;padding:9px}iframe{width:100%;height:650px;border:0}svg{width:100%;height:430px}a{color:#087a85}</style><main>
<h1>语义 GCS：选择“在哪里抓”以及前后路线</h1><div class="card"><strong>从固定动作点，推进到有限候选动作集合。</strong><p>同一场景中生成 5 个抓取基座候选，4 个通过几何、抓取和参考轨迹检查。GCS 同时选择空手路线、一个合法抓取候选及其携物路线；每个候选保留自己的抓取变换，分支间不能混接。</p><p>空手仍用原场景圆盘，提升到固定姿态的 10D 配置集合；携物用每个候选对应的 10D IRIS 凸区域。取放是已检查的固定动作原语。当前是有限候选联合选择，不是连续抓取位姿优化，也不再是“第一个通过就使用”。</p></div>
<div class="card"><h2>候选检查</h2><table><thead><tr><th>候选</th><th>基座 (x,y,yaw)</th><th>检查结果</th><th>准备耗时</th></tr></thead><tbody id="library"></tbody></table><p>拒绝表示本次 IK / 原语 / 检查条件下不接受，不证明该位置物理不可行。候选准备包含初步完整规划和区域生成，成本必须计入；表中后续求解复用此库。</p></div>
<div class="card"><h2>改变动作可用性，重新求解</h2><table><thead><tr><th>条件</th><th>选中抓取位置</th><th>完整几何 / 语义</th><th>本次接触执行</th><th>基座总路长</th></tr></thead><tbody id="results"></tbody></table><p><strong>取整质量限制：</strong>本次禁用首次选中的候选后，返回目标值从 9.07175 降至 8.85712。后者本来也属于原可行域，说明第一次返回解并非全局最优；不能把所选候选称为“最佳抓取点”。</p><p>各组共享同一候选库及场景，仅改变允许的抓取动作。目标是加权配置空间路径长度，包含固定动作原语代价；不是只选距离最近的抓取点。禁用全部候选只说明当前图无接受路径。</p></div>
<div class="card"><label>查看 <select id="case"></select></label><p id="detail"></p><label><input id="disks" type="checkbox" checked> 显示空手圆盘</label><svg id="map" viewBox="-4 -3 7 6"></svg><p>原场景基座轨迹投影：蓝色为空手，橙色为携物。绿色标记所选抓取基座位置；这些候选仅相隔 5 cm，详情见上表。完整机器人见下方 3D。</p></div>
<div class="card" id="replay"><button id="reference">规划参考回放</button> <button id="physics">本次接触仿真</button><p id="label"></p><iframe id="viewer" title="候选抓取任务回放"></iframe></div>
<div class="card"><strong>研究边界</strong><p>该实验验证动作语义、候选选择与连续路径进入同一 GCS；不是算法优越性证明。候选位于有限固定网格，抓取姿态由 IK 预生成；IRIS 区域可能含未被采样发现的碰撞，所有输出路径另做密集审计，不能称严格认证区域。动力学每条新路线只运行一次，理想平面底盘无轮地模型。</p><a href="results.json">完整数据</a> · <a href="../semantics/">上一步：任务顺序对照</a></div></main>
<script>const D=__DATA__,names={all_candidates:'所有通过检查的候选可用',selected_site_unavailable:'禁用首次选中的抓取位置',none_available:'所有抓取动作不可用'},sel=document.getElementById('case');function row(id,values){const tr=document.createElement('tr');for(const v of values){const td=document.createElement('td');td.textContent=v;tr.append(td)}document.getElementById(id).append(tr)}
for(const a of D.library.attempts)row('library',[a.name,a.base_q.join(', '),a.status==='accepted'?'通过':(a.reason==='pick geometric guard failed'?'抓取几何条件不满足':a.reason),a.preparation_seconds.toFixed(2)+' s']);
for(const[k,r]of Object.entries(D.results)){const opt=document.createElement('option');opt.value=k;opt.textContent=names[k];sel.append(opt);const c=D.library.attempts.find(a=>a.name===r.selected_candidate);row('results',[names[k],c?c.name+' ('+c.base_q.join(', ')+')':'无接受路径',r.audit?(r.audit.accepted?'通过':'拒绝'):'未执行',r.physics?(r.physics.satisfied?'通过':'失败'):'未运行',r.audit?r.audit.base_path_length_m.toFixed(3)+' m':'—'])}
function map(k){const svg=document.getElementById('map');svg.replaceChildren();function el(tag,a){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);svg.append(e);return e}if(document.getElementById('disks').checked)for(const d of D.disks)el('circle',{cx:d.center[0],cy:-d.center[1],r:d.radius,fill:'#dceffa','fill-opacity':.23,stroke:'#9ecbd7','stroke-width':.012});const objects=[...D.config.world.walls,...Object.values(D.config.objects).filter(x=>x&&x.center&&x.size),...(D.config.objects.distractors||[])];for(const o of objects){if(o.size&&o.center)el('rect',{x:o.center[0]-o.size[0]/2,y:-o.center[1]-o.size[1]/2,width:o.size[0],height:o.size[1],fill:'#748395'})}const points=D.traces[k]||[];for(let i=1;i<points.length;i++)el('line',{x1:points[i-1].xy[0],y1:-points[i-1].xy[1],x2:points[i].xy[0],y2:-points[i].xy[1],stroke:points[i].attached?'#eb7e22':'#2862db','stroke-width':.04});const c=D.library.attempts.find(a=>a.name===D.results[k].selected_candidate);if(c){el('circle',{cx:c.base_q[0],cy:-c.base_q[1],r:.07,fill:'#159b64'});el('text',{x:c.base_q[0]+.13,y:-c.base_q[1]+.1,'font-size':.16,fill:'#087750'}).textContent=c.name}}
function show(physics){const k=sel.value;document.getElementById('viewer').src=k+(physics?'_dynamics':'')+'.html';document.getElementById('label').textContent=physics?'本次接触执行：自由杯子，力驱动，无 attachment':'本次规划参考：运动学 attachment，不能代替物理执行'}
function update(){const k=sel.value,r=D.results[k];document.getElementById('detail').textContent=r.status==='solved'?`候选 ${r.selected_candidate}；同一次 GCS 求解 ${r.solve_seconds.toFixed(2)} 秒，${r.vertices} 顶点 / ${r.edges} 边；目标值 ${r.objective.toFixed(4)}（加权配置路径长度）。`:'动作库为空，任务图没有到接受状态的路径，不输出完成动画。';document.getElementById('replay').hidden=r.status!=='solved';document.getElementById('physics').hidden=!r.physics;map(k);if(r.status==='solved')show(!!r.physics);else document.getElementById('viewer').src='about:blank'}sel.onchange=update;document.getElementById('disks').onchange=()=>map(sel.value);document.getElementById('reference').onclick=()=>show(false);document.getElementById('physics').onclick=()=>show(true);update();</script></html>'''
    (out/'index.html').write_text(page.replace('__DATA__',json.dumps(data,ensure_ascii=False)))
    print(out/'index.html')


if __name__=='__main__':main()
