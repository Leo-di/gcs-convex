"""Frozen paired timed-carry mechanism experiment; no full-task claim."""
import argparse
import concurrent.futures
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/results/timed_carry_v1'
WEB=ROOT/'outputs/results/grasp_fix_web/planar/timed_carry'
LIB=ROOT/'outputs/results/refined_transfer_v1/screened/libraries'
METHODS=['gcs10','gcs50','portal50']

def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def worker(case,method):
    import numpy as np
    from pydrake.geometry.optimization import HPolyhedron
    from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES
    from .carry_mode_gcs import CarryOracle
    from .timed_carry_bezier import graph_solve,portal_solve,audit,DEADLINES
    begin=time.perf_counter();d=json.loads(Path(case['input']).read_text());c=d['candidates'][case['candidate']]
    anchor=[r for r in c['rows'] if r['phase'] in PICK_PHASES][-1]
    target=[r for r in d['goals'][case['candidate']][case['goal']]['rows'] if r['phase'] in PLACE_PHASES][0]
    start=q(anchor);goal=q(target);regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in c['cover']['regions']]
    if method.startswith('gcs'):info,segments=graph_solve(regions,start,goal,rounded=int(method[3:]))
    else:info,segments=portal_solve(regions,start,goal)
    result={'case':case,'method':method,'solver':info,'status':'solver_failed','deadlines':{str(int(t)):False for t in DEADLINES}}
    if segments:
        oracle=CarryOracle(d['config'],anchor)
        checked=audit(segments,regions,start,goal,distance=lambda x:oracle.query(x[:2],x[2:])[1])
        frames=checked.pop('frames');result['audit']=checked
        valid=checked['kinematics_passed'] and checked['collision_samples_passed']
        result['status']='valid' if valid else 'audit_failed'
        result['deadlines']={str(int(t)):bool(valid and info['duration']<=t+1e-5) for t in DEADLINES}
        write(OUT/'trajectories'/f"{case['id']}_{method}.json",{'segments':segments,'frames':frames})
    result['wall_seconds']=time.perf_counter()-begin
    write(OUT/'results'/f"{case['id']}_{method}.json",result)

def publish(protocol):
    results=[json.loads(p.read_text()) for p in sorted((OUT/'results').glob('*.json'))]
    summary={m:{'completed':sum(r['method']==m for r in results),
        'valid':sum(r['method']==m and r['status']=='valid' for r in results),
        'deadlines':{str(t):sum(r['method']==m and r.get('deadlines',{}).get(str(t),False) for r in results) for t in [10,20,40,80]}}
        for m in METHODS}
    write(OUT/'summary.json',{'expected':len(protocol['cases'])*3,'completed':len(results),'methods':summary})
    WEB.mkdir(parents=True,exist_ok=True)
    for name in ['protocol.json','summary.json']:shutil.copy2(OUT/name,WEB/name)
    rows=''.join('<tr><td>'+html.escape(r['case']['id'])+'</td><td>'+r['method']+'</td><td>'+r['status']+'</td><td>'+str(round(r['solver']['duration'],3) if r.get('solver',{}).get('duration') is not None else '—')+'</td><td>'+str(round(r.get('wall_seconds',0),1))+'</td></tr>' for r in results)
    table=''.join('<tr><td>'+m+'</td><td>'+str(v['completed'])+'/25</td>'+''.join('<td>'+str(v['deadlines'][str(t)])+'</td>' for t in [10,20,40,80])+'</tr>' for m,v in summary.items())
    page='''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>限时搬运 GCS 对照</title><style>body{font:17px sans-serif;max-width:1100px;margin:35px auto;color:#19324a}table{border-collapse:collapse;width:100%;margin:24px 0}td,th{padding:9px;border:1px solid #bbc}p{line-height:1.7}</style><h1>限时搬运：GCS 与多路线 Portal 对照</h1><p>真实 10D 配置空间、携杯几何、三次 Bézier 空间/时间曲线、连续速度。25 个抓取条件，来自 11 个有可用动作的场景；另 5 个场景缺少动作库，不计入模块分母。不是完整抓放任务成功率，也不是独立测试集。</p><p>双方相同区域、端点、速度、80 秒规划范围；GCS 官方松弛与 10/50 路径舍入，基线为时间加权 Portal + Yen 路线搜索 + 相同连续优化。每方法每条件 300 秒预算。四个预定时间窗相关，不视作独立重复。</p><p>成功需运动学与采样碰撞审计通过。没有加速度/力矩限制或连续碰撞安全证明。</p><a href="protocol.json">冻结实验协议</a> · <a href="summary.json">结果数据</a><h2>累计成功数量（全部完成后才比较比例）</h2><table><tr><th>方法</th><th>完成</th><th>≤10 s</th><th>≤20 s</th><th>≤40 s</th><th>≤80 s</th></tr>'''+table+'</table><h2>逐例结果</h2><table><tr><th>条件</th><th>方法</th><th>审计状态</th><th>轨迹时长/s</th><th>运行耗时/s</th></tr>'+rows+'</table>'
    tmp=WEB/'index.tmp';tmp.write_text(page);tmp.replace(WEB/'index.html')

def run_one(case,method):
    from .transfer_scene_experiment import run_process
    begin=time.perf_counter();path=OUT/'results'/f"{case['id']}_{method}.json"
    code,expired=run_process([sys.executable,'-m','src.run_timed_carry_experiment','--worker',case['id'],'--method',method],OUT/'logs'/f"{case['id']}_{method}.log",300)
    if expired or code!=0 or not path.exists():
        write(path,{'case':case,'method':method,'status':'timeout' if expired else 'error','exit_code':code,'wall_seconds':time.perf_counter()-begin,'deadlines':{str(t):False for t in [10,20,40,80]}})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker');parser.add_argument('--method');args=parser.parse_args()
    if args.worker:
        protocol=json.loads((OUT/'protocol.json').read_text());worker(next(c for c in protocol['cases'] if c['id']==args.worker),args.method);return
    if (OUT/'protocol.json').exists():raise RuntimeError('Frozen run already exists; inspect it instead of overwriting.')
    cases=[];empty=[];inputs={}
    for p in sorted(LIB.glob('*/shared.json')):
        d=json.loads(p.read_text());inputs[str(p)]=digest(p)
        if not d['candidates']:empty.append(p.parent.name)
        for name in sorted(d['candidates']):
            goals=sorted(d['goals'].get(name,{}))
            if not goals:raise RuntimeError('Candidate lacks goal: '+p.parent.name+'/'+name)
            cases.append({'id':p.parent.name+'_'+name,'scene':p.parent.name,'candidate':name,'goal':goals[0],'input':str(p)})
    sources={str(p):digest(p) for p in sorted((ROOT/'src').glob('*.py'))}
    protocol={'created_unix':time.time(),'cases':cases,'empty_scenes':empty,'methods':METHODS,'deadlines_seconds':[10,20,40,80],
      'velocity_caps':[.3,.3,.4]+[.4]*7,'horizon':80,'wall_budget_seconds':300,'parallel_slots':4,
      'selection':'All available screened grasp candidates; lexicographically first available goal. No filtering by new solver outcome.',
      'portal':{'density':5,'yen_unique_region_route_cap':50,'raw_path_cap':2000,'internal_seconds':280},
      'model':'Cubic r(s), cubic strictly monotone h(s); q(t)=r(h^-1(t)); C1 r and h; rest endpoints; duration objective. Shared velocity constraints. No acceleration/torque constraints.',
      'success':'Solver success AND control-point/C1/velocity audit AND 101 collision samples per segment AND duration within deadline.',
      'scope':'Conditional carrying module on previously used scenes, not independent full-task success or contact simulation. Four deadlines are correlated. Region preprocessing is shared and excluded from query timing.',
      'reference':'https://arxiv.org/abs/2205.04422','inputs':inputs,'sources':sources}
    for sub in ['results','logs','source_snapshot']: (OUT/sub).mkdir(parents=True,exist_ok=True)
    for source in sources:shutil.copy2(source,OUT/'source_snapshot'/Path(source).name)
    write(OUT/'protocol.json',protocol);publish(protocol)
    jobs=[(c,METHODS[(j+i)%3]) for i,c in enumerate(cases) for j in range(3)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run_one,c,m) for c,m in jobs]
        for f in concurrent.futures.as_completed(futures):f.result();publish(protocol)
    changed=[p for p,h in {**inputs,**sources}.items() if digest(Path(p))!=h]
    write(OUT/'complete.json',{'seconds':time.time()-protocol['created_unix'],'changed_inputs_or_sources':changed});publish(protocol)

if __name__=='__main__':main()
