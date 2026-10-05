"""Matched planning wall budget on the entire previously used 16-scene cohort.
Development diagnostic, never an independent confirmation of superiority.
Run with PYTHONPATH=outputs/dependencies/toppra python -m src.regularized_carry_budget.
"""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path('outputs/results/regularized_budget_dev_v1')
SOURCE=Path('outputs/results/fresh_timed_carry_v1')
BUDGET=30.
EPSILON=.001


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def load_scene(name):
    import numpy as np
    from pydrake.geometry.optimization import HPolyhedron
    from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES
    selection=json.loads((SOURCE/'selection.json').read_text())[name]
    d=json.loads((SOURCE/'screened/libraries'/name/'shared.json').read_text())
    candidate=d['candidates'][selection['candidate']]
    anchor=[r for r in candidate['rows'] if r['phase'] in PICK_PHASES][-1]
    start=q(anchor)
    goal=q([r for r in d['goals'][selection['candidate']][selection['goal']]['rows'] if r['phase'] in PLACE_PHASES][0])
    regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in candidate['cover']['regions']]
    return d,anchor,start,goal,regions


def plan(name,method,cpu):
    os.sched_setaffinity(0,{int(cpu)})
    from .timed_carry_bezier import graph_solve,region_yen_solve,trajectory_objective
    _,_,start,goal,regions=load_scene(name)
    base=ROOT/f'{name}_{method}'
    begin=time.monotonic()
    write(base.with_suffix('.ready.json'),{'monotonic_start':begin})
    def checkpoint(segments,cost,attempts=None):
        elapsed=time.monotonic()-begin
        if elapsed<=BUDGET:
            write(base.with_suffix('.incumbent.json'),{'segments':segments,'objective':cost,'seconds':elapsed,'attempts':attempts})
    if method=='gcs50':
        info,segments=graph_solve(regions,start,goal,rounded=50,smoothness=EPSILON)
        if segments:checkpoint(segments,trajectory_objective(segments,len(start),EPSILON))
    else:
        info,segments=region_yen_solve(regions,start,goal,smoothness=EPSILON,
            budget_seconds=BUDGET,on_incumbent=checkpoint)
    write(base.with_suffix('.solver.json'),info)


def validate(name,method):
    import numpy as np
    from .timed_carry_bezier import audit,LIMITS
    from .carry_mode_gcs import CarryOracle
    from .carry_toppra import retime
    base=ROOT/f'{name}_{method}'
    record=json.loads(base.with_suffix('.process.json').read_text())
    file=base.with_suffix('.incumbent.json')
    record.update({'scene':name,'method':method,'success':False})
    if file.exists():
        incumbent=json.loads(file.read_text());d,anchor,start,goal,regions=load_scene(name)
        oracle=CarryOracle(d['config'],anchor)
        check=audit(incumbent['segments'],regions,start,goal,distance=lambda x:oracle.query(x[:2],x[2:])[1]);check.pop('frames')
        timing=retime(incumbent['segments'])
        record.update({'incumbent':{k:v for k,v in incumbent.items() if k!='segments'},'audit':check,'toppra':timing})
        record['success']=bool(check['kinematics_passed'] and check['collision_samples_passed'] and timing['solved'] and timing['checked_velocity_ratio']<=1 and timing['checked_acceleration_ratio']<=1)
        acceleration=np.array([.5,.5,.8]+[1.]*7);distance=abs(goal-start)
        lower=np.where(distance<=LIMITS**2/acceleration,2*np.sqrt(distance/acceleration),distance/LIMITS+LIMITS/acceleration)
        record['unobstructed_rest_to_rest_lower_bound']=float(max(lower))
        record['execution_ratio_to_lower_bound']=timing['scaled_duration']/max(lower) if timing['solved'] else None
    write(base.with_suffix('.result.json'),record)


def run_job(name,method,cpu):
    base=ROOT/f'{name}_{method}'
    with base.with_suffix('.log').open('w') as log:
        p=subprocess.Popen([sys.executable,'-m',__name__ if __name__!='__main__' else 'src.regularized_carry_budget','plan',name,method,str(cpu)],stdout=log,stderr=log,start_new_session=True)
        started=time.monotonic();deadline=None;expired=False
        while p.poll() is None:
            ready=base.with_suffix('.ready.json')
            if deadline is None and ready.exists():deadline=json.loads(ready.read_text())['monotonic_start']+BUDGET
            if time.monotonic()>(deadline if deadline is not None else started+60):
                os.killpg(p.pid,signal.SIGKILL);expired=True;break
            time.sleep(.01)
        code=p.wait()
        write(base.with_suffix('.process.json'),{'planning_budget':BUDGET,'timeout':expired,'returncode':code,'cpu':cpu})
        try:
            result=subprocess.run([sys.executable,'-m','src.regularized_carry_budget','validate',name,method],stdout=log,stderr=log,timeout=120)
            if result.returncode:raise RuntimeError(f'validation exit {result.returncode}')
        except Exception as e:
            write(base.with_suffix('.result.json'),{'scene':name,'method':method,'success':False,'error':repr(e)})
    record=json.loads(base.with_suffix('.result.json').read_text())
    print(name,method,'success',record['success'],'timeout',record.get('timeout'),flush=True)


def main():
    if ROOT.exists():raise RuntimeError('Refuse overwrite of experiment')
    ROOT.mkdir(parents=True)
    selected=json.loads((SOURCE/'selection.json').read_text())
    files=[Path(__file__),Path('src/timed_carry_bezier.py'),Path('src/carry_toppra.py'),SOURCE/'selection.json']
    files += [SOURCE/'screened/libraries'/n/'shared.json' for n,s in selected.items() if 'failure' not in s]
    protocol={'scope':'Development on ALL 16 previously used scenes; not held-out confirmation. Shared preparation failure remains in denominator. Carrying-module success only, no new contact execution.',
        'budget_seconds':BUDGET,'epsilon':EPSILON,'horizon':80,'gcs_rounding':50,
        'baseline':'Yen region routes with identical convex restrictions, no route count cap; best objective checkpoint preserved at timeout.',
        'timing':'30s wall clock from immediately before graph construction, excluding import/library loading and postprocessing. Every saved incumbent timestamp must be <=30s. Supervisor polls10ms. Same pinned CPU per paired scene. Other machine jobs may cause noise.',
        'primary':'Valid collision-sampled and TOPPRA velocity/acceleration-sampled trajectory found within planning budget. No execution deadline for primary.',
        'secondary':'Paired smooth objective and post-TOPPRA duration; exploratory deadline ratios 1.15,1.30,1.50 times unobstructed rest-to-rest lower bound.',
        'scenes':selected,'hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    write(ROOT/'protocol.json',protocol)
    snap=ROOT/'source_snapshot';snap.mkdir()
    for f in files:
        if f.suffix=='.py':(snap/f.name).write_bytes(f.read_bytes())
    cpus=sorted(os.sched_getaffinity(0))[:2]
    def scene_job(item):
        i,(name,selection)=item;cpu=cpus[i%len(cpus)]
        if 'failure' in selection:
            for method in ['gcs50','region_yen']:write(ROOT/f'{name}_{method}.result.json',{'scene':name,'method':method,'success':False,'reason':selection['failure']})
        else:
            methods=['gcs50','region_yen'] if i%2==0 else ['region_yen','gcs50']
            for method in methods:run_job(name,method,cpu)
    # One sequential queue per CPU; no accidental co-location from thread scheduling.
    items=list(enumerate(sorted(selected.items())))
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures=[pool.submit(lambda queue:[scene_job(x) for x in queue],items[k::len(cpus)]) for k in range(len(cpus))]
        for future in futures:future.result()
    write(ROOT/'complete.json',{'records':len(list(ROOT.glob('*.result.json'))),'complete':True})


if __name__=='__main__':
    if len(sys.argv)==1:main()
    elif sys.argv[1]=='plan':plan(*sys.argv[2:])
    elif sys.argv[1]=='validate':validate(*sys.argv[2:])
