"""Prospectively frozen new-scene time-pressure comparison; all outcomes kept."""
import argparse,copy,html,json,math,os,shutil,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .benchmark_semantic_baselines import atomic_json,file_hash
from .scene_spec import load_candidate_library_config
from .transfer_scene_experiment import run_process

ROOT=Path('outputs/results/fresh_timed_carry_v1')
WEB=Path('outputs/results/grasp_fix_web/planar/fresh_timed_carry')
METHODS=['gcs50','portal50']
WINDOWS={'relative_105':1.05,'relative_115':1.15,'relative_130':1.30,'loose_80':None}
REFINEMENT=Path('outputs/results/navigation_refresh_diagnostic_v1/full_refinement.py')

def lower_bound(start,goal,limits):
    """Necessary travel time from integrating each coordinate's speed limit."""
    limits=np.asarray(limits);assert np.all(limits>0)
    return float(np.max(np.abs(np.asarray(goal)-start)/limits))

def design():
    base=load_candidate_library_config();base['pick_task'].update(collision_checked_pregrasp=True,cartesian_approach=True)
    base['iris_np2_cover']['counterexample_refinement']=True
    rng=np.random.default_rng(20261012);scenes={};changes={};cases=[]
    for i in range(16):
        name=f'fresh_{i:02d}';c=copy.deepcopy(base)
        width=float(rng.uniform(1.,1.2));cy=float(rng.uniform(-.85,.95));dx=float(rng.uniform(-.07,.07));cup=rng.uniform([1.53,.64],[1.67,.70]).tolist()
        for w in c['world']['walls']:
            if w['name'] in ['shared_wall_upper','shared_wall_lower']:
                sign=1 if w['name'].endswith('upper') else -1;w['size'][1]=3-width/2;w['center'][1]=sign*(3+width/2)/2
            elif w['name']=='doorway_header':w['size'][1]=width
        c['objects']['mobile_cart']['center']=[-1.25,cy,.41];c['objects']['cup']['center'][:2]=cup
        for obj in ['dropoff_table','dropoff_bin']:c['objects'][obj]['center'][0]+=dx
        c['task_zones']['dropoff']['center'][0]+=dx;c['delivery_task']['place_cup_center'][0]+=dx
        for k in ['iris_np2_cover','iris_np2_doorway']:
            c[k]['random_seed']=320000+i*1000;c[k]['holdout_random_seed']=420000+i*1000
        scenes[name]=c;changes[name]={'door_width':width,'cart_y':cy,'cup_xy':cup,'dropoff_dx':dx}
        cases.append({'id':name,'scene':name,'start':[-3.,0.,0.]})
    return {'scenes':scenes,'changes':changes,'cases':cases,'pickup_x':[1.1,1.15,1.2,1.25],
            'goal_x':[-2.5,-2.35,-2.2],'goal_y':1.25,'goal_heights':[.14,.22],'seed':20261012}

def prepare(name):
    from outputs.results.navigation_refresh_diagnostic_v1 import full_refinement as refine
    from . import geometry_outbound_gcs as nav
    from .prepare_transfer_scenes import prepare as build
    refine.ROOT=ROOT/'raw';refine.name=name;refine.events=[];nav.make_cover=refine.refined_cover
    try:build(ROOT/'raw',name)
    finally:nav.make_cover=refine.ORIGINAL

def query(name,window,method):
    from pydrake.geometry.optimization import HPolyhedron
    from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES
    from .timed_carry_bezier import graph_solve,portal_solve,audit,LIMITS
    from .carry_mode_gcs import CarryOracle
    begin=time.perf_counter();selection=json.loads((ROOT/'selection.json').read_text())[name]
    record={'scene':name,'window':window,'method':method,'selection':selection,'success':False}
    if selection.get('failure'):record.update(status=selection['failure'])
    else:
        d=json.loads((ROOT/'screened/libraries'/name/'shared.json').read_text());candidate=d['candidates'][selection['candidate']]
        anchor=[r for r in candidate['rows'] if r['phase'] in PICK_PHASES][-1]
        goalrow=[r for r in d['goals'][selection['candidate']][selection['goal']]['rows'] if r['phase'] in PLACE_PHASES][0]
        start=q(anchor);goal=q(goalrow);regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in candidate['cover']['regions']]
        horizon=80. if WINDOWS[window] is None else WINDOWS[window]*lower_bound(start,goal,LIMITS)
        record.update(horizon=horizon,lower_bound=lower_bound(start,goal,LIMITS))
        if method=='gcs50':info,segments=graph_solve(regions,start,goal,horizon=horizon,rounded=50)
        else:info,segments=portal_solve(regions,start,goal,horizon=horizon)
        record.update(solver=info,status='solver_failed')
        if segments:
            oracle=CarryOracle(d['config'],anchor);checked=audit(segments,regions,start,goal,distance=lambda x:oracle.query(x[:2],x[2:])[1]);frames=checked.pop('frames')
            passed=checked['kinematics_passed'] and checked['collision_samples_passed'] and info['duration']<=horizon+1e-5
            record.update(audit=checked,success=bool(passed),status='valid' if passed else 'audit_failed')
            atomic_json(ROOT/'trajectories'/f'{name}_{window}_{method}.json',{'segments':segments,'frames':frames})
    record['worker_wall_seconds']=time.perf_counter()-begin
    atomic_json(ROOT/'results'/f'{name}_{window}_{method}.json',record)

def publish(stage):
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    prep=len(list((ROOT/'raw/libraries').glob('*/preparation.json')));screen=len(list((ROOT/'screened/libraries').glob('*/preparation.json')))
    summary={'stage':stage,'prepared_scenes':prep,'screened_scenes':screen,'completed_queries':len(rows),'expected_queries':128,'metrics':{}}
    table=''
    for w in WINDOWS:
        summary['metrics'][w]={}
        for m in METHODS:
            selected=[r for r in rows if r['window']==w and r['method']==m];v={'completed':len(selected),'success':sum(r['success'] for r in selected)};summary['metrics'][w][m]=v
            table+=f'<tr><td>{w}</td><td>{m}</td><td>{v["completed"]}/16</td><td>{v["success"]}</td></tr>'
    atomic_json(ROOT/'summary.json',summary);atomic_json(WEB/'summary.json',summary)
    detail=''.join('<tr>'+''.join('<td>'+html.escape(str(r.get(k,'—')))+'</td>' for k in ['scene','window','method','status','horizon'])+'</tr>' for r in rows)
    page='''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>新场景：GCS 时间约束检验</title><style>body{font:17px/1.7 sans-serif;max-width:1100px;margin:35px auto;color:#19324a}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #bbc;padding:8px}pre{white-space:pre-wrap}</style><h1>新场景：GCS 时间约束检验</h1><p>16 个新随机组合场景，随机种子 20261012；同一双房间模板，不代表任意环境。每场景固定选择首个有放置动作的抓取候选及其第一个目标，不按连通性或求解表现筛选。无动作库、无覆盖、超时和审计失败全部保留。</p><p>时间下界 L = max |目标配置 − 起始配置| / 各维速度上限。主比较为 1.15L；1.05L、1.30L 是预定敏感性分析，80 秒为宽松对照。每个窗口独立重新求解。这是归一化运动时间压力测试，不是假定真实用户要求这些截止时间。</p><p>双方相同 10D 区域、端点、三次空间/时间曲线、连续速度及检查。GCS 使用官方 50 路径舍入；基线使用 Portal + Yen 多路线及相同连续优化，最多 50 条区域路线/2000 条原始路线。各 300 秒 worker 预算。共享准备也使用 GCS/IRIS，因此基线只在查询阶段不用 GCS。</p><p>本轮是建库加搬运模块检验，没有完整抓放接触执行；无加速度/力矩约束。区域及轨迹碰撞检查为数值验证，不是连续安全证明。每个窗口 16 个成对场景，四窗口不能合并成 64 个独立场景。</p><a href="protocol.json">冻结协议</a> · <a href="summary.json">进度数据</a> · <a href="../timed_carry/">上一轮结果</a>'''
    page+=f'<p>阶段：{stage}；建库 {prep}/16；筛查 {screen}/16；查询 {len(rows)}/128。</p><table><tr><th>窗口</th><th>方法</th><th>完成</th><th>成功数</th></tr>'+table+'</table>'
    if (ROOT/'analysis.json').exists():
        a=json.loads((ROOT/'analysis.json').read_text());page+='<h2>最终成对分析</h2><p><a href="report.txt">最终报告</a> · <a href="analysis.json">成对统计</a></p><pre>'+html.escape(json.dumps(a['paired'],ensure_ascii=False,indent=2))+'</pre>'
    page+='<table><tr><th>场景</th><th>窗口</th><th>方法</th><th>状态</th><th>实际时间上限/s</th></tr>'+detail+'</table>'
    tmp=WEB/'index.tmp';tmp.write_text(page);tmp.replace(WEB/'index.html')

def finish(protocol):
    rows=[json.loads(p.read_text()) for p in (ROOT/'results').glob('*.json')];by={(r['scene'],r['window'],r['method']):r for r in rows};paired={}
    for window in WINDOWS:
        pairs=[(by[s,window,'gcs50']['success'],by[s,window,'portal50']['success']) for s in protocol['scenes']]
        wins=sum(a and not b for a,b in pairs);losses=sum(b and not a for a,b in pairs);discordant=wins+losses
        p=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
        paired[window]={'gcs_success':sum(a for a,b in pairs),'portal_success':sum(b for a,b in pairs),'denominator':16,'gcs_only':wins,'portal_only':losses,'ties':16-discordant,'exact_mcnemar_two_sided_p':p}
    changed=[p for p,h in protocol['hashes'].items() if file_hash(Path(p))!=h]
    frozen=json.loads((ROOT/'query_inputs.json').read_text());changed += [p for p,h in frozen.items() if file_hash(Path(p))!=h]
    analysis={'paired':paired,'primary':'relative_115','changed_sources_or_inputs':changed,'scope':protocol['scope'],'statistical_note':'Primary window prespecified. Sensitivity tests are not independent confirmatory claims; 16 scenes provide limited power. Ties do not establish equivalence.'}
    atomic_json(ROOT/'analysis.json',analysis);atomic_json(WEB/'analysis.json',analysis)
    report='新场景限时搬运对照\n'+json.dumps(analysis,ensure_ascii=False,indent=2)
    (ROOT/'report.txt').write_text(report);(WEB/'report.txt').write_text(report)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare');parser.add_argument('--screen');parser.add_argument('--query');parser.add_argument('--window');parser.add_argument('--method');args=parser.parse_args()
    if args.prepare:return prepare(args.prepare)
    if args.screen:
        from .screen_carry_regions import screen_scene
        return screen_scene(ROOT,args.screen,use_known_witnesses=False)
    if args.query:return query(args.query,args.window,args.method)
    if ROOT.exists():raise RuntimeError('Refusing to overwrite frozen experiment')
    spec=design();ROOT.mkdir();WEB.mkdir(parents=True,exist_ok=True)
    for folder in ['results','trajectories','logs']:(ROOT/folder).mkdir()
    for c in ['raw','screened']:(ROOT/c).mkdir();atomic_json(ROOT/c/'design.json',spec)
    atomic_json(ROOT/'design.json',spec)
    paths=list(Path('src').glob('*.py'))+[BASE,REFINEMENT,ROOT/'design.json',ROOT/'raw/design.json',ROOT/'screened/design.json']
    protocol={'seed':spec['seed'],'scenes':list(spec['scenes']),'methods':METHODS,'windows':WINDOWS,'primary':'relative_115','worker_timeout_seconds':300,'preparation_timeout_seconds':900,'workers':4,
              'selection':'Lexicographically first candidate having any goal, then first goal. No connectivity or query-outcome filtering. One case per scene; preparation failures retained in denominator.',
              'scope':'Prospective new random combinations of the same template. Shared preparation uses GCS/IRIS. Query-only comparison, conditioned on finite actions; all 16 scenes retained. Kinematic carrying module, not full contact execution. No acceleration/torque constraints or continuous collision certificate.',
              'velocity_caps':[.3,.3,.4]+[.4]*7,'portal':{'density':5,'unique_route_cap':50,'raw_path_cap':2000,'internal_seconds':280},
              'hashes':{str(p):file_hash(p) for p in paths}}
    atomic_json(ROOT/'protocol.json',protocol);atomic_json(WEB/'protocol.json',protocol)
    (ROOT/'source_snapshot').mkdir()
    for p in paths:
        if p.suffix=='.py':shutil.copy2(p,ROOT/'source_snapshot'/p.name)
    shutil.copy2(BASE,ROOT/'source_snapshot/base_config.json')
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[(i*len(cpus))//4] for i in range(4)];names=protocol['scenes'];begin=time.perf_counter();publish('preparation')
    def prep_slot(i):
        for name in names[i::4]:
            raw=ROOT/'raw/libraries'/name;raw.mkdir(parents=True,exist_ok=True)
            code,expired=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.fresh_timed_carry','--prepare',name],raw/'prepare.log',900)
            if code or expired:
                atomic_json(raw/'shared.json',{'config':spec['scenes'][name],'candidates':{},'goals':{},'disks':[]})
                atomic_json(raw/'preparation.json',{'candidate_count':0,'goal_count':0,'connectivity':[],'failure':'preparation_timeout' if expired else 'preparation_error'})
            dest=ROOT/'screened/libraries'/name;dest.mkdir(parents=True,exist_ok=True)
            code,expired=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.fresh_timed_carry','--screen',name],dest/'screen.log',900)
            if code or expired:
                atomic_json(dest/'shared.json',{'config':spec['scenes'][name],'candidates':{},'goals':{},'disks':[]})
                atomic_json(dest/'preparation.json',{'candidate_count':0,'goal_count':0,'failure':'screen_timeout' if expired else 'screen_error'})
            print('PREPARED',name,flush=True)
    with ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(prep_slot,i) for i in range(4)]
        while not all(f.done() for f in futures):publish('preparation_and_screening');time.sleep(5)
        for f in futures:f.result()
    prep_seconds=time.perf_counter()-begin;selection={};input_hashes={}
    for name in names:
        path=ROOT/'screened/libraries'/name/'shared.json';d=json.loads(path.read_text());input_hashes[str(path)]=file_hash(path)
        candidates=[c for c in sorted(d['candidates']) if d['goals'].get(c)]
        selection[name]={'candidate':candidates[0],'goal':sorted(d['goals'][candidates[0]])[0]} if candidates else {'failure':'shared_preparation_no_candidate'}
    atomic_json(ROOT/'selection.json',selection);input_hashes[str(ROOT/'selection.json')]=file_hash(ROOT/'selection.json');atomic_json(ROOT/'query_inputs.json',input_hashes)
    query_begin=time.perf_counter()
    jobs=[(name,w,m) for i,name in enumerate(names) for w in WINDOWS for m in METHODS[i%2:]+METHODS[:i%2]]
    def query_slot(i):
        for name,w,m in jobs[i::4]:
            log=ROOT/'logs'/f'{name}_{w}_{m}.log';log.parent.mkdir(exist_ok=True);record=ROOT/'results'/f'{name}_{w}_{m}.json'
            code,expired=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.fresh_timed_carry','--query',name,'--window',w,'--method',m],log,300)
            if code or expired or not record.exists():atomic_json(record,{'scene':name,'window':w,'method':m,'status':'timeout' if expired else 'program_error','success':False,'return_code':code})
    with ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(query_slot,i) for i in range(4)]
        while not all(f.done() for f in futures):publish('queries');time.sleep(5)
        for f in futures:f.result()
    finish(protocol);atomic_json(ROOT/'complete.json',{'preparation_seconds':prep_seconds,'query_seconds':time.perf_counter()-query_begin,'total_seconds':time.perf_counter()-begin});publish('complete')

if __name__=='__main__':main()
