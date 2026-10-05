"""Run the frozen execution manifest without dropping rejected plans."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys

from .arm_branch_gcs import ROOT


def main():
    manifest=json.loads((ROOT/'execution_protocol.json').read_text())
    cpus=Queue()
    for cpu in [4,12,20,28]:cpus.put(cpu)
    env={**os.environ,'LCM_DEFAULT_URL':'memq://','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    def run(item):
        if not item['run']:return dict(**item,execution_status='no_plan')
        cpu=cpus.get()
        try:
            directory=ROOT/item['case']/item['method']
            with (directory/'execution.log').open('w') as log:
                completed=subprocess.run(['taskset','-c',str(cpu),sys.executable,'-m','src.execute_arm_branch_gcs',
                    '--case',item['case'],'--method',item['method']],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            result=directory/'execution/result.json'
            if completed.returncode or not result.exists():
                return dict(**item,execution_status='worker_failed',returncode=completed.returncode)
            return dict(**item,execution_status='completed',accepted=json.loads(result.read_text())['accepted'])
        except subprocess.TimeoutExpired:
            return dict(**item,execution_status='timeout')
        finally:cpus.put(cpu)
    status=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,item) for item in manifest['attempts']]
        for future in as_completed(futures):
            item=future.result();status.append(item)
            (ROOT/'execution_status.json').write_text(json.dumps(status,indent=2))
            print(json.dumps(item),flush=True)


if __name__=='__main__':main()
