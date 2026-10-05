"""Frozen physical-scene experiment; retains every proposed candidate and scene."""
import argparse, copy, json, os, subprocess, sys, time, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .benchmark_semantic_baselines import atomic_json, file_hash, METHODS, run

ROOT=Path('outputs/results/semantic_multiscene_v1')

def prepare(root,name):
    from .build_scene import build_scene
    from .animate_delivery_demo import plan_delivery_candidate
    from .audit_replay import audit_replay
    from .semantic_gcs_experiment import guards
    from .iris_region_generator import cover_to_dict
    from .scene_slice_gcs import SliceOracle
    frozen=json.loads((root/'design.json').read_text());config=frozen['scenes'][name]
    directory=root/'libraries'/name;directory.mkdir(parents=True,exist_ok=True)
    atomic_json(directory/'config.json',config);began=time.perf_counter();report={'attempts':[]}
    for proposal in frozen['candidates']:
        attempt=copy.deepcopy(proposal);tick=time.perf_counter()
        try:
            scene=build_scene(config=copy.deepcopy(config),visualize=False)
            data=plan_delivery_candidate(scene,np.array(proposal['base_q']),attempt)
            dense=audit_replay(data['trace'],config,10)
            if not dense['satisfied']:raise RuntimeError('Dense reference audit failed')
            gate=guards(data['trace'],config)
            dest=directory/proposal['name'];dest.mkdir(exist_ok=True)
            for file,value in [('delivery_trace.json',data['trace']),('cover.json',cover_to_dict(data['carrying_iris_cover'])),('guards.json',gate),('dense.json',dense),('scene_config_snapshot.json',config)]:atomic_json(dest/file,value)
            attempt.update(status='accepted',regions=len(data['carrying_iris_cover'].regions))
        except RuntimeError as e:
            attempt.update(status='rejected',reason=str(e),details=getattr(e,'details',{}))
        except Exception as e:attempt.update(status='error',reason=repr(e))
        attempt['preparation_seconds']=time.perf_counter()-tick;report['attempts'].append(attempt)
        atomic_json(directory/'report.json',report);print(name,attempt['name'],attempt['status'],flush=True)
    scene=build_scene(config=copy.deepcopy(config),visualize=False)
    oracle=SliceOracle(scene,scene.arm_home_positions,scene.gripper_open_positions)
    lower=np.array(config['robot']['lower_limits'][:2]);upper=np.array(config['robot']['upper_limits'][:2])
    seeds=[[-3.,0.]]+[c['base_q'][:2] for c in frozen['candidates']]
    seeds += [[float(x),float(y)] for x in np.arange(lower[0]+.2,upper[0],.4) for y in np.arange(lower[1]+.2,upper[1],.4)]
    seeds += [[float(x),y] for x in np.arange(-.8,.81,.2) for y in [-.2,0.,.2]]
    disks=[];seen=set()
    for xy in seeds:
        key=tuple(np.round(xy,7))
        if key in seen:continue
        seen.add(key);distance,self_distance,closest=oracle.query(xy)
        if self_distance<0:raise RuntimeError('Fixed posture self collision')
        radius=float(min(.65,distance-.0201,np.min(np.array(xy)-lower),np.min(upper-np.array(xy))))
        if radius>=.035:disks.append({'center':xy,'radius':radius})
    atomic_json(directory/'model.json',{'empty_cover':disks,'margin':.0201,'oracle_calls':oracle.calls})
    report.update(preparation_seconds=time.perf_counter()-began,disks=len(disks),accepted=sum(a['status']=='accepted' for a in report['attempts']))
    atomic_json(directory/'report.json',report)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare');p.add_argument('--repaired',action='store_true');p.add_argument('--root',type=Path,default=ROOT);args=p.parse_args();root=args.root
    if args.prepare:prepare(root,args.prepare);return
    if root.exists():raise RuntimeError('Output exists; refusing to mix runs')
    root.mkdir(parents=True)
    base=json.loads(Path('outputs/results/semantic_candidates/library/config.json').read_text())
    if args.repaired:
        base['pick_task']['collision_checked_pregrasp']=True
        base['iris_np2_cover']['counterexample_refinement']=True
    scenes={n:copy.deepcopy(base) for n in ['nominal','cup_left','cup_right','cart_near_route']}
    scenes['cup_left']['objects']['cup']['center'][0]-=.10
    scenes['cup_right']['objects']['cup']['center'][0]+=.10
    scenes['cart_near_route']['objects']['mobile_cart']['center']=[-1.35,-.90,.41]
    candidates=[{'name':f'x{round(x*100)}_y{round(y*100)}','base_q':[x,y,0.]} for x in [1.05,1.15,1.25] for y in [-.25,-.10,.05]]
    design={'scenes':scenes,'candidates':candidates,'seeds':[0,7,19],'start':[-3.,0.,0.],
      'scope':'Four predefined physical scenes, nine ordered proposals per scene; no outcome-based scene filtering. Original task and 3D collision geometry. Rebuild every library and empty cover.',
      'cost_scope':'Fresh eager shared-library preparation measured separately. Adding prep to warm time describes this implementation, not an optimized lazy baseline.',
      'contact_cohort':'Every scene, seed 0, all three methods; retain missing plans as skipped failures.',
      'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    design['repaired']=args.repaired
    design['refinement_scope']='Common eager library preparation uses counterexample cuts; timed B1/B2/joint all use the resulting frozen regions, same audit, no method-specific repair.'
    snapshots=root/'source_snapshot';snapshots.mkdir()
    for source in Path('src').glob('*.py'):shutil.copyfile(source,snapshots/source.name)
    design['source_sha256']={str(p):file_hash(p) for p in sorted(Path('src').glob('*.py'))}
    atomic_json(root/'design.json',design)
    begin=time.perf_counter()
    def job(pair):
        i,name=pair;directory=root/'libraries'/name;directory.mkdir(parents=True)
        env={**os.environ,'LCM_DEFAULT_URL':'memq://','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with (directory/'prepare.log').open('w') as log:
            result=subprocess.run(['taskset','-c',str([4,12,20,28][i]),sys.executable,'-m','src.benchmark_semantic_multiscene','--root',str(root),'--prepare',name],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        if result.returncode:raise RuntimeError('Preparation process failed: '+name)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(job,enumerate(scenes)))
    atomic_json(root/'preparation.json',{'wall_seconds':time.perf_counter()-begin,'scenes':{n:json.loads((root/'libraries'/n/'report.json').read_text()) for n in scenes}})
    cells=[]
    for i,name in enumerate(scenes):
        library=root/'libraries'/name;report=json.loads((library/'report.json').read_text());allowed=[a['name'] for a in report['attempts'] if a['status']=='accepted']
        for j,seed in enumerate(design['seeds']):
            shift=(i+j)%3
            cells.append({'id':f'{name}__seed{seed}','physical_scene':name,'start':design['start'],'policy':'all_prepared','allowed':allowed,'seed':seed,'methods':METHODS[shift:]+METHODS[:shift],'library':str(library),'model':str(library/'model.json')})
    paths=sorted(Path('src').glob('*.py'))+sorted((root/'libraries').rglob('*.json'))+[root/'design.json']
    spec={'cells':cells,'methods':METHODS,'workers':4,'cpus':[4,12,20,28],'budget_seconds_per_method':300,'rounding_paths':10,'library':cells[0]['library'],'model':cells[0]['model'],'input_sha256':{str(p):file_hash(p) for p in paths}}
    atomic_json(root/'protocol.json',spec);run(root,spec)
    subprocess.run([sys.executable,'-m','src.benchmark_semantic_contacts','--root',str(root)],check=True)

if __name__=='__main__':main()
