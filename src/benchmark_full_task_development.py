"""Ten paired development tasks, common audits and contact execution for all plans."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np
from .benchmark_semantic_baselines import atomic_json, file_hash
from .semantic_candidate_gcs import build_and_solve, expand, q, PICK_PHASES, PLACE_PHASES
from .online_carry_iris import attach_carry_endpoints, sets
from .online_iris_attachment import connection_status

METHODS=['sequential_gcs','product_gcs','product_online_iris']


def plan_worker(args):
    root=args.output;design=json.loads((root/'protocol.json').read_text())
    for path,h in design['hashes'].items():
        if file_hash(path)!=h:raise RuntimeError('Frozen input changed: '+path)
    bundle=json.loads((root/'inputs'/f'{args.cell}.json').read_text())
    cell=bundle['cell'];config=bundle['config'];candidates=bundle['candidates'];disks=bundle['disks']
    out=root/args.cell/args.method;out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    record={'cell':args.cell,'method':args.method,'planning_success':False,'end_to_end_success':False,
            'attempts':[],'online':{},'status':'planning','primitive_rejections':bundle['primitive_rejections'],
            'preparation_seconds_shared':bundle['preparation_seconds']}
    def save():
        record['planning_wall_seconds']=time.perf_counter()-began
        atomic_json(out/'record.json',record)
    save()
    allowed=[name for name in cell['allowed'] if name in candidates]
    if not allowed:
        record.update(status='failed',failure='action_primitive_unavailable');save();return
    if args.method=='product_online_iris':
        for j,name in enumerate(allowed):
            candidate=candidates[name];pick=[r for r in candidate['rows'] if r['phase'] in PICK_PHASES]
            place=[r for r in candidate['rows'] if r['phase'] in PLACE_PHASES]
            new,info=attach_carry_endpoints(config,candidate['cover']['regions'],pick[-1],q(pick[-1]),q(place[0]),
                seed_base=91000+100*design['cell_ids'].index(args.cell)+2*j)
            record['online'][name]=info;candidate['cover']['regions']=new
            save()
    groups=[allowed] if args.method!='sequential_gcs' else [[n] for n in allowed]
    sites=config.get('inspection_sites')
    site_groups=([sites] if args.method!='sequential_gcs' else [[s] for s in sites]) if sites is not None else [None]
    for subset in groups:
        for site_subset in site_groups:
            c=copy.deepcopy(config)
            if site_subset is not None:c['inspection_sites']=site_subset
            connectivity={}
            for name in subset:
                rows=candidates[name]['rows']
                start=q([r for r in rows if r['phase'] in PICK_PHASES][-1]);goal=q([r for r in rows if r['phase'] in PLACE_PHASES][0])
                connectivity[name]=connection_status(sets(candidates[name]['cover']['regions']),start,goal)
            entry={'allowed':subset,'sites':site_subset,'connectivity':connectivity}
            attempt_dir=out/f'attempt_{len(record["attempts"]):02d}'
            result,path=build_and_solve(disks,candidates,subset,c,rounding_seed=0,max_rounded_paths=10)
            attempt_dir.mkdir(exist_ok=True);atomic_json(attempt_dir/'gcs.json',result)
            entry.update(solver_status=result['status'],build_seconds=result['build_seconds'],solve_seconds=result['solve_seconds'])
            if path is None:
                if not any(s['endpoints_covered'] for s in connectivity.values()):failure='endpoint_not_covered'
                elif not any(s['region_graph_connected'] for s in connectivity.values()):failure='region_graph_disconnected'
                else:failure='gcs_solver_or_task_graph_failed'
                entry['failure']=failure
            else:
                audit=expand(path,candidates,disks,c,attempt_dir,visualize=False)
                entry.update(audit=audit,selected_candidate=result['selected_candidate'],inspection_site=result.get('inspection_site'))
                if not audit['accepted']:
                    entry['failure']='trajectory_geometry_or_semantic_audit_failed'
                else:
                    used=[]
                    for v in path:
                        if v['kind']=='carry' and candidates[v['candidate']]['cover']['regions'][v['region']].get('online'):
                            used.append({'candidate':v['candidate'],'region':v['region']})
                    record.update(planning_success=True,status='planned',reference=str(attempt_dir),
                                  selected_candidate=result['selected_candidate'],inspection_site=result.get('inspection_site'),
                                  used_online_regions=used)
            record['attempts'].append(entry);save()
            if record['planning_success']:return
    record.update(status='failed',failure=record['attempts'][-1]['failure']);save()


def contact_worker(args):
    from .execute_delivery_dynamics import execute
    from .semantic_gcs_experiment import contact_checkpoint_monitor
    out=args.output/args.cell/args.method;r=json.loads((out/'record.json').read_text())
    ref=Path(r['reference']);config=json.loads((ref/'scene_config_snapshot.json').read_text())
    rows=json.loads((ref/'delivery_trace.json').read_text());config['execution_dynamics']={'hydroelastic_support':True}
    began=time.perf_counter();report,_=execute(config,rows,visualize=False)
    actual_inspection=True
    if 'inspection_sites' in config:
        checks=[contact_checkpoint_monitor(report,s['xy']) for s in config['inspection_sites']]
        actual_inspection=any(s['satisfied'] for s in checks);report['actual_inspection_checks']=checks
    target=np.asarray(config['delivery_task']['place_cup_center'])
    # Target is a placement instruction, not a scene geometry change. Check
    # actual horizontal placement; vertical settling is already checked by execute.
    final=[t for t in report['trace'] if t['time_s']>=report['duration_s']-1]
    error=max(float(np.linalg.norm(np.asarray(t['cup_pose'])[:2,3]-target[:2])) for t in final)
    report['actual_target_xy_error_m']=error;report['target_tolerance_m']=.025
    report['complete_task_satisfied']=bool(report['satisfied'] and actual_inspection and error<=.025)
    atomic_json(out/'physics.json',report)
    r.update(end_to_end_success=report['complete_task_satisfied'],contact_wall_seconds=time.perf_counter()-began,
             contact_satisfied=report['satisfied'],actual_inspection_satisfied=actual_inspection,
             actual_target_xy_error_m=error,status='completed')
    if not r['end_to_end_success']:r['failure']='contact_execution_or_actual_task_failed'
    atomic_json(out/'record.json',r)


def run_process(root,cell,method,cpu,stage,budget):
    out=root/cell/method;out.mkdir(parents=True,exist_ok=True)
    env={**os.environ,'LCM_DEFAULT_URL':'memq://','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    cmd=['taskset','-c',str(cpu),sys.executable,'-m','src.benchmark_full_task_development',
         '--output',str(root),'--cell',cell,'--method',method,'--worker',stage]
    with (out/(stage+'.log')).open('w') as log:
        p=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        timeout=False
        try:p.wait(timeout=budget)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    f=out/'record.json';r=json.loads(f.read_text()) if f.exists() else {'cell':cell,'method':method,'planning_success':False,'end_to_end_success':False}
    if timeout or p.returncode:
        r.update(status='timeout' if timeout else 'error',failure=stage+'_timeout' if timeout else stage+'_error',end_to_end_success=False)
        if stage=='plan':r['planning_success']=False
    r[stage+'_return_code']=p.returncode;atomic_json(f,r)
    return r


def summarize(root):
    records=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/record.json'))]
    design=json.loads((root/'protocol.json').read_text())
    summary={'development_only':not design.get('frozen_random_evaluation',False),'completed_records':len(records),
             'expected_records':len(design['cell_ids'])*len(METHODS),'methods':{},'records':records}
    for method in METHODS:
        rows=[r for r in records if r['method']==method];failures={}
        for r in rows:
            if not r['end_to_end_success']:failures[r.get('failure','unfinished')]=failures.get(r.get('failure','unfinished'),0)+1
        summary['methods'][method]={'tasks':len(rows),'planning_successes':sum(r['planning_success'] for r in rows),
            'end_to_end_successes':sum(r['end_to_end_success'] for r in rows),'failures':failures,
            'planning_wall_median_s':float(np.median([r['planning_wall_seconds'] for r in rows if 'planning_wall_seconds' in r])) if rows else None,
            'online_calls':sum(len(a['attempts']) for r in rows for a in r.get('online',{}).values()),
            'online_region_users':sum(bool(r.get('used_online_regions')) for r in rows)}
    atomic_json(root/'summary.json',summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worker',choices=['plan','contact']);p.add_argument('--cell');p.add_argument('--method',choices=METHODS)
    p.add_argument('--prepared',action='store_true',help='Use preregistered prepared inputs; refuse to overwrite protocol/results')
    args=p.parse_args()
    if args.worker:
        try:(plan_worker if args.worker=='plan' else contact_worker)(args)
        except Exception:
            traceback.print_exc();raise
        return
    root=args.output
    if args.prepared:
        if (root/'protocol.json').exists() or not (root/'design.json').exists():
            raise ValueError('Need prepared inputs without an existing run protocol')
        cells=json.loads((root/'development_cases.json').read_text())
        design_input=json.loads((root/'design.json').read_text())
        if cells!=design_input['evaluation_cases']:raise ValueError('Evaluation cases differ from frozen design')
    else:
        if root.exists():raise ValueError('Use a new output directory')
        root.mkdir(parents=True)
        from .prepare_full_task_development import prepare
        cells=prepare(root)
    cpus=sorted(os.sched_getaffinity(0));cpus=[cpus[i] for i in [4,12,20,28]] if len(cpus)>28 else cpus[:4]
    source=list(Path('src').glob('*.py'));inputs=list((root/'inputs').glob('*.json'))
    design={'cell_ids':[c['id'] for c in cells],'methods':METHODS,'cpus':cpus,'workers':len(cpus),
        'planning_budget_seconds':300,'contact_budget_seconds':300,'rounding_seed':0,'max_rounded_paths':10,
        'hashes':{str(f):file_hash(f) for f in source+inputs},
        'scope':'10 development tasks, not a held-out evaluation; 6 policy/checkpoint cases and 4 real target changes of 4cm; no endpoint-region deletion.',
        'baseline':'Sequential tries legal candidates ascending, then legal inspection sites in fixed order; each fixed branch is solved by the SAME official GCS transcription. Fixed primitive endpoints make its motion-stage objectives separable. It stops at first fully audited plan; no retry after physical failure.',
        'product':'Joint candidate/site choice in semantic GCS; online variant adds query-local endpoint regions for every legal available branch before the same solve.',
        'common_inputs':'Same filtered offline cache and checked primitives; all known colliding cached regions removed for ALL methods. Geometry and each branch grasp transform unchanged. New targets change place IK only.',
        'primary_metric':'Planning audit AND actual contact execution AND actual pre-grasp checkpoint when required AND final target horizontal error <= 2.5cm.',
        'timing':'Planning measured in worker including library handling/online growth/solve/audit; shared target-primitive preparation reported separately; historical offline cover cost not remeasured. Startup included only in timeout budget. Four pinned single-core workers, method order rotated by cell.'}
    if args.prepared:
        design.update(frozen_random_evaluation=True,
            scope='40 preregistered random placement-base/target tasks in one unchanged scene; four balanced policies. Same-method paired evaluation; all preparation failures retained.',
            evaluation_design=design_input,
            common_inputs='Same filtered offline covers and fixed grasp transforms; targets and placement bases change, place primitives re-generated identically for all methods; no transit seed paths.')
    atomic_json(root/'protocol.json',design)
    import shutil
    snapshot=root/'source_snapshot';snapshot.mkdir()
    for f in source:shutil.copy2(f,snapshot/f.name)
    began=time.perf_counter()
    def slot(i):
        for ci in range(i,len(cells),len(cpus)):
            cell=cells[ci]['id'];order=METHODS[ci%3:]+METHODS[:ci%3]
            for method in order:
                print('START',cell,method,flush=True)
                r=run_process(root,cell,method,cpus[i],'plan',300)
                if r['planning_success']:r=run_process(root,cell,method,cpus[i],'contact',300)
                print('DONE',cell,method,'plan',r['planning_success'],'task',r['end_to_end_success'],r.get('failure',''),flush=True)
    with ThreadPoolExecutor(len(cpus)) as pool:list(pool.map(slot,range(len(cpus))))
    summary=summarize(root)
    changed=[f for f,h in design['hashes'].items() if file_hash(f)!=h]
    atomic_json(root/'complete.json',{'elapsed_seconds':time.perf_counter()-began,'runs':len(summary['records']),'changed_inputs':changed})
    print(json.dumps(summary['methods'],indent=2),flush=True)


if __name__=='__main__':main()
