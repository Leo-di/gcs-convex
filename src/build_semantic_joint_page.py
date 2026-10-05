"""Publish the shared-scene semantic GCS experiment without hiding negatives."""
import json
from pathlib import Path
import shutil
from .semantic_gcs_experiment import contact_checkpoint_monitor


def main():
    source=Path('outputs/results/semantic_joint_gcs')
    output=Path('outputs/results/grasp_fix_web/planar/semantics');output.mkdir(parents=True,exist_ok=True)
    results=json.loads((source/'results.json').read_text())
    model=json.loads((source/'shared_model.json').read_text())
    config=json.loads((source/'delivery/scene_config_snapshot.json').read_text())
    data={'results':results,'model':model,'config':config,'traces':{}}
    for case,result in results.items():
        if result['status']=='solved':
            rows=json.loads((source/case/'delivery_trace.json').read_text())
            data['traces'][case]=[{'xy':r['base_q'][:2],'attached':r['attached']} for r in rows]
            shutil.copy2(source/case/'scene.html',output/(case+'.html'))
            physics=source/(case+'_dynamics')/'result.json'
            if physics.exists():
                raw=json.loads(physics.read_text())
                result['physics']={k:v for k,v in raw.items() if k!='trace'}
                result['physics']['checkpoint_before_contact']=contact_checkpoint_monitor(raw)
                shutil.copy2(physics,output/(case+'_dynamics.json'))
                shutil.copy2(physics.parent/'scene.html',output/(case+'_dynamics.html'))
    (output/'results.json').write_text(json.dumps(data,ensure_ascii=False))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>语义真正进入 GCS</title>
<style>body{margin:0;background:#f1f5f9;color:#20334a;font:16px/1.7 system-ui}main{max-width:1180px;margin:auto;padding:24px}.card{padding:20px;background:white;margin:16px 0;border-radius:12px}h1{font-size:28px}table{width:100%;border-collapse:collapse}td,th{padding:10px;text-align:left;border-bottom:1px solid #dbe4ed}select,button{font:inherit;padding:8px}svg{width:100%;height:440px}iframe{width:100%;height:650px;border:0}a{color:#087a85}.notice{border-left:5px solid #087a85}</style>
<main><h1>同一 3D 场景：改变语义，改变 GCS 合法解</h1><div class="card notice"><strong>本次改变的是任务约束，不是障碍物或圆盘。</strong><p>空手区域 × 任务进度 → 抓取动作 → 携物姿态区域 → 放置动作 → 接受状态。所有路线在同一次 GCS 求解中选择，没有先用 Dijkstra 决定区域顺序。</p><p>“先检查再取杯”会复制一层完全相同的空手区域；只有经过检查点才能进入允许抓取的层。抓取边连接固定动作点，并展开为已有 IK 接近、闭合和抬杯动作；放置边同理。连续抓取姿态与动作位置尚未联合优化。</p></div>
<div class="card"><table><thead><tr><th>指令 / 条件</th><th>GCS 选出的动作</th><th>完整送杯检查</th><th>取杯前到检查点</th><th>基座路长</th><th>本次接触执行</th></tr></thead><tbody id="table"></tbody></table><p>纯几何对照只要求到达终点；它没有被要求完成送杯，因此任务失败是预期结果，不是性能落败。缺少抓取动作的负例只说明当前动作库和图不接受任务，不证明物理世界无解。</p></div>
<div class="card"><label>查看实验 <select id="case"></select></label><p id="notice"></p><p id="automaton"></p><svg id="map" viewBox="-4 -3 7 6" aria-label="原三维场景基座路径投影"></svg><p>原场景俯视投影：浅蓝圆盘为空手固定姿态可行区域；蓝线为空手，橙线为携物，红色十字为新增检查点。圆盘中心是完整机器人的基座位置，本体和杯子请看下面 3D。携物使用另四层姿态条件区域，未全部叠画。</p></div>
<div class="card" id="replay"><p><button id="kinematic">规划参考回放</button> <button id="physics">本次接触动力学回放</button></p><p id="replaylabel"></p><iframe id="viewer" title="当前语义任务的三维回放"></iframe></div>
<div class="card"><strong>证据与限制</strong><p>所有组使用同一几何模型哈希、圆盘库、姿态库、起点和终点。独立监视器根据原始位姿、attachment 变化和几何残差检查取放顺序；检查点根据取杯前的基座坐标检查，不信任动作名称。</p><p>参考回放仍使用运动学 attachment；仅标注“本次接触动力学”的回放才使用自由杯子和力驱动执行。有限姿态库及固定动作原语来自原系统，语义由人工结构化定义，未实现自然语言解析；密集采样不是连续安全证明。这是功能性对照，尚非研究方法优越性的证据。</p><a href="results.json">完整求解与审计数据</a> · <a href="../comparison/">上一轮规划器对照</a></div></main>
<script>const D=__DATA__,names={geometry_only:'只到终点',delivery:'取杯后放入箱子',inspect_then_delivery:'先到检查点，再取杯放入箱子',pick_unavailable:'取杯送达，但抓取动作不可用'},actions={inspect:'检查点',pick:'抓取并抬杯',place:'放置并撤回'},sel=document.getElementById('case');
for(const[k,r]of Object.entries(D.results)){const o=document.createElement('option');o.value=k;o.textContent=names[k];sel.append(o);const tr=document.createElement('tr');for(const s of [names[k],r.status==='solved'?r.actions.map(a=>actions[a]).join(' → ')||'无取放动作':'图拒绝',r.audit?(r.audit.delivery.satisfied?'通过':'未完成'):'未执行',r.audit?(r.audit.checkpoint_before_pick.satisfied?'通过':'未到访'):'未执行',r.audit?r.audit.base_path_length_m.toFixed(3)+' m':'—',r.physics?(r.physics.satisfied?'通过':'失败'):'未运行']){const td=document.createElement('td');td.textContent=s;tr.append(td)}document.getElementById('table').append(tr)}
function draw(k){const svg=document.getElementById('map');svg.replaceChildren();function el(tag,a){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const[k,v]of Object.entries(a))e.setAttribute(k,v);svg.append(e);return e}
for(const d of D.model.empty_cover)el('circle',{cx:d.center[0],cy:-d.center[1],r:d.radius,fill:'#dceffa','fill-opacity':.23,stroke:'#9ecbd7','stroke-width':.012});
const objects=[...D.config.world.walls,...Object.values(D.config.objects).filter(x=>x&&x.center&&x.size),...(D.config.objects.distractors||[])];for(const o of objects){if(!o.center||!o.size)continue;el('rect',{x:o.center[0]-o.size[0]/2,y:-o.center[1]-o.size[1]/2,width:o.size[0],height:o.size[1],fill:'#748395','fill-opacity':.8})}
const rows=D.traces[k]||[];for(let i=1;i<rows.length;i++){const a=rows[i-1],b=rows[i];el('line',{x1:a.xy[0],y1:-a.xy[1],x2:b.xy[0],y2:-b.xy[1],stroke:b.attached?'#eb7e22':'#2862db','stroke-width':.045})}
const[x,y]=D.model.checkpoint;el('path',{d:`M ${x-.1} ${-y} H ${x+.1} M ${x} ${-y-.1} V ${-y+.1}`,stroke:'#cc2936','stroke-width':.035});
for(const [label,xy]of [['检查点',D.model.checkpoint],['取物',D.model.carry.start],['放置',D.model.carry.goal]]){const t=el('text',{x:xy[0]+.08,y:-xy[1]-.12,'font-size':.14,fill:'#22334a'});t.textContent=label}}
function replay(physics){const k=sel.value,r=D.results[k];document.getElementById('viewer').src=k+(physics?'_dynamics':'')+'.html';document.getElementById('replaylabel').textContent=physics?'本次自由杯子接触仿真：'+(r.physics.satisfied?'送杯通过':'送杯未通过')+(k==='inspect_then_delivery'?'；实际检查点到访早于双侧抓取接触：'+(r.physics.checkpoint_before_contact.satisfied?'通过':'失败'):''):'本次规划参考：运动学 attachment；不代表接触执行'}
function update(){const k=sel.value,r=D.results[k],solved=r.status==='solved';document.getElementById('notice').textContent=solved?`同一次 GCS 求解，${r.vertices} 个顶点、${r.edges} 条边；求解 ${r.solve_seconds.toFixed(2)} 秒（单次，复用区域，非性能排名）。`:'抓取边不存在，接受状态不可达；未调用求解器，也不显示完成轨迹。';document.getElementById('automaton').textContent=k==='inspect_then_delivery'?'未检查空手 → 检查点 → 已检查空手 → 抓取 → 携物 → 放置 → 接受':k==='geometry_only'?'空手到终点即可接受（没有送杯语义要求）':'空手 → 合法抓取 → 携物 → 合法放置 → 接受';document.getElementById('replay').hidden=!solved;document.getElementById('physics').hidden=!r.physics;draw(k);if(solved)replay(!!r.physics);else document.getElementById('viewer').src='about:blank'}
sel.value='inspect_then_delivery';sel.onchange=update;document.getElementById('kinematic').onclick=()=>replay(false);document.getElementById('physics').onclick=()=>replay(true);update();</script></html>'''
    (output/'index.html').write_text(page.replace('__DATA__',json.dumps(data,ensure_ascii=False)))
    print(output/'index.html')


if __name__=='__main__':main()
