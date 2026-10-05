"""Execute the preregistered seed-0 cohort after planning has finished."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


def atomic(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/results/semantic_baselines_long'))
    args=parser.parse_args();root=args.root
    if not (root/'complete.json').exists():raise RuntimeError('Finish timed planning before starting physics')
    spec=json.loads((root/'protocol.json').read_text());jobs=[]
    for cell in spec['cells']:
        if cell['seed']==0 and (cell['allowed'] or cell.get('physical_scene')):
            for method in spec['methods']:jobs.append((cell,method))
    lock=threading.Lock();results=[];active={};begin=time.perf_counter()
    def save():atomic(root/'contacts.json',{'elapsed_seconds':time.perf_counter()-begin,'total':len(jobs),'completed':len(results),'active':active,'runs':results})
    def slot(slot):
        for index in range(slot,len(jobs),2):
            cell,method=jobs[index];directory=root/cell['id']/method
            record=json.loads((directory/'record.json').read_text());best=record.get('best')
            result={'cell':cell['id'],'method':method,'status':'skipped_no_valid_plan'}
            if best:
                destination=directory/'dynamics';destination.mkdir(exist_ok=True)
                key=cell['id']+'/'+method;began=time.perf_counter()
                with lock:active[key]=began;save()
                env={**os.environ,'LCM_DEFAULT_URL':'memq://','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
                command=['taskset','-c',str(spec['cpus'][slot]),sys.executable,'-m','src.execute_delivery_dynamics',
                         best['reference'],'--output-dir',str(destination),'--html']
                with (destination/'run.log').open('w') as log:
                    process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                    try:process.wait(timeout=180)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait();result['status']='timeout'
                file=destination/'result.json'
                if file.exists():
                    raw=json.loads(file.read_text())
                    result.update(status='passed' if raw['satisfied'] else 'failed',
                                  result={k:v for k,v in raw.items() if k not in ('trace','unexpected_contacts')},
                                  output=str(destination),candidate=best['candidate'])
                elif result['status']!='timeout':result['status']='error'
                result.update(wall_seconds=time.perf_counter()-began,return_code=process.returncode)
                with lock:active.pop(key)
            with lock:
                results.append(result);save()
                print(json.dumps({'completed':len(results),'total':len(jobs),'cell':cell['id'],'method':method,'status':result['status']}),flush=True)
    save()
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(slot,range(2)))


if __name__=='__main__':main()
