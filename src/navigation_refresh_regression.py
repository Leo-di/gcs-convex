"""Full-library regression for current-geometry bootstrap navigation."""
import argparse,json,os,shutil,sys,time,threading,html
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
from .prepare_transfer_scenes import prepare
from .transfer_scene_experiment import run_process
ROOT=Path('outputs/results/se2_navigation_refresh_v3')
OLD=Path('outputs/results/heldout_transfer_v1')
WEB=Path('outputs/results/grasp_fix_web/planar/se2_navigation_refresh')

def publish(done=False):
    rows=[]
    for name in json.loads((ROOT/'design.json').read_text())['scenes']:
        old=json.loads((OLD/'raw/libraries'/name/'preparation.json').read_text());p=ROOT/'libraries'/name/'preparation.json';new=json.loads(p.read_text()) if p.exists() else None
        rows.append({'scene':name,'old_candidates':old['candidate_count'],'old_goals':old['goal_count'],'new_candidates':new['candidate_count'] if new else None,'new_goals':new['goal_count'] if new else None,'failure':new.get('failure') if new else None})
    atomic_json(WEB/'progress.json',{'complete':done,'scenes':rows,'completed':sum(x['new_candidates'] is not None for x in rows),'total':16})
    if done:
        atomic_json(ROOT/'comparison.json',{'scenes':rows,'scope':'Same diagnosed scenes, repair regression only; not held out and not GCS-vs-baseline evidence. Full bootstrap plus IRIS and goal construction; no query/contact executions.'})
        table=''.join('<tr>'+''.join('<td>'+html.escape(str(x[k]))+'</td>' for k in ['scene','old_candidates','new_candidates','old_goals','new_goals','failure'])+'</tr>' for x in rows)
        (WEB/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>导航覆盖修复回归</title><style>body{font:18px/1.8 system-ui;margin:40px}td,th{padding:8px;border-bottom:1px solid #ddd}</style><h1>当前几何导航覆盖：完整建库回归</h1><p>相同16个已诊断场景、64个预定抓取候选。仅检验共享建库；没有查询算法对照或实际抓放执行，不代表GCS成功率优势。新导航采用固定姿态、固定yaw、一次Bezier及官方Drake GCS，保持旧语义通道；旧导航允许yaw变化且使用三次Bezier。失败保留，断连不证明任务无解。</p><table><tr><th>场景</th><th>旧候选</th><th>新候选</th><th>旧放置组合</th><th>新放置组合</th><th>运行异常</th></tr>'+table+'</table><p><a href="comparison.json">比较数据</a> · <a href="details.json">全部候选和失败</a> · <a href="../heldout_transfer/">原始冻结实验</a></p>')
        shutil.copy2(ROOT/'comparison.json',WEB/'comparison.json');atomic_json(WEB/'details.json',{n:json.loads((ROOT/'libraries'/n/'preparation.json').read_text()) for n in json.loads((ROOT/'design.json').read_text())['scenes']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--worker');a=p.parse_args()
    if a.worker:return prepare(ROOT,a.worker)
    ROOT.mkdir();WEB.mkdir(parents=True,exist_ok=True);spec=json.loads((OLD/'design.json').read_text());spec.pop('source_hashes',None);spec['scope']='Posthoc navigation-cover repair regression on the SAME16 scenes. 64 grasp attempts. Three-layer fix: all-lane outbound permission, SE(2) yaw-sliced current-geometry disks with tucked navigation arm and symmetric 0.10m corridor refinement (cart x-range, both detour bands), first-order GCS, 20.1mm margin. 3600 seconds per full scene preparation. No test-query feedback. Compare full bootstrap acceptance, not just navigation. No actual contact claims.';atomic_json(ROOT/'design.json',spec)
    hashes={str(f):file_hash(f) for f in Path('src').glob('*.py')};(ROOT/'source_snapshot').mkdir()
    for f in hashes:shutil.copy2(f,ROOT/'source_snapshot'/Path(f).name)
    atomic_json(ROOT/'protocol.json',{'hashes':hashes,'tests_passed':30,'scene_timeout_s':900,'scope':spec['scope']})
    (WEB/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>导航修复回归运行中</title><style>body{font:18px/1.8 system-ui;margin:40px}pre{white-space:pre-wrap}</style><h1>导航覆盖修复：16场景完整建库回归</h1><p>64个固定抓取候选；保留原语义通道，使用当前Drake几何生成区域与官方GCS。此轮仅建库，不宣称完整抓放或GCS优势。</p><pre id="p">运行中…</pre><a href="../heldout_transfer/">上一轮已完成实验</a><script>setInterval(async()=>{let d=await(await fetch('progress.json?t='+Date.now())).json();document.querySelector('#p').textContent=JSON.stringify(d,null,2);if(d.complete)location.reload()},5000)</script>''')
    stop=threading.Event()
    def monitor():
        while not stop.is_set():publish();stop.wait(5)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();names=list(spec['scenes']);cpus=sorted(os.sched_getaffinity(0));slots=[cpus[i] for i in [4,12,20,28]];start=time.perf_counter()
    def worker(i):
        for name in names[i::4]:
            dest=ROOT/'libraries'/name;dest.mkdir(parents=True,exist_ok=True);t=time.perf_counter();print('START',name,flush=True)
            code,expired=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.navigation_refresh_regression','--worker',name],dest/'prepare.log',3600)
            if code or expired:
                progress=dest/'preparation_progress.json';d=json.loads(progress.read_text()) if progress.exists() else {'attempts':[],'goal_attempts':[]};d.update(candidate_count=0,goal_count=0,failure='timeout' if expired else 'program_error',return_code=code,seconds=time.perf_counter()-t);atomic_json(dest/'preparation.json',d)
            print('DONE',name,code,flush=True)
    try:
        with ThreadPoolExecutor(4) as pool:list(pool.map(worker,range(4)))
    finally:stop.set();thread.join()
    changed=[f for f,h in hashes.items() if file_hash(f)!=h];atomic_json(ROOT/'complete.json',{'seconds':time.perf_counter()-start,'changed_sources':changed,'scenes':16});assert not changed;publish(True);print('COMPLETE',flush=True)

if __name__=='__main__':main()
