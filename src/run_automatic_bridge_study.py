"""Fresh matched runs, including failed candidates; no outcome-based replacement."""
import argparse,json,os,signal,subprocess,sys,time,hashlib,shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    original=Path('outputs/results/semantic_grounding_choices_20261019/protocol.json');protocol=json.loads(original.read_text());methods=['fixed_gcs','repair_gcs','repair_ompl'];protocol.update(methods=methods,seed=20261019,scope='Matched rerun on eight previously studied correlated candidates. Development experiment, not held-out generalization. Repair attempts only endpoint coverage/graph gaps, not IK failures. Outer candidate selection, not joint GCS.',repair={'margin_m':.0201,'ompl_seconds':20,'motion_check_step_m':.0025,'simplification_seconds':.5,'max_repairs':1},candidate_timeout_s=600)
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));sources={}
    for name in ['src/automatic_motion_bridge.py','src/run_automatic_bridge_study.py','src/semantic_grounding_candidates.py','src/geometry_outbound_gcs.py','configs/scene_semantic_choice_v6.yaml']:
        src=Path(name);dst=root/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);sources[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(sources,indent=2))
    for method in methods:
        folder=root/method;folder.mkdir();(folder/'protocol.json').write_text(json.dumps(dict(protocol,bridge_method=method),indent=2))
    def run(method,index):
        sub=root/method;candidate=protocol['candidates'][index];folder=sub/candidate['id'];folder.mkdir();began=time.perf_counter()
        with (folder/'run.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.semantic_grounding_candidates','--output',str(sub),'--worker',str(index)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid,signal.SIGTERM);proc.wait();code='timeout'
        f=folder/'candidate_result.json';row=json.loads(f.read_text()) if f.exists() else {'id':candidate['id'],'base_q':candidate['base_q']}
        row.update(method=method,process_exit=code,total_wall_seconds=time.perf_counter()-began)
        row['verified_task_success']=bool(code==0 and all(row.get(k,{}).get('accepted') for k in ['bridge','operation','physical']))
        f.write_text(json.dumps(row,indent=2));return row
    started=time.perf_counter();rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,m,i) for i in range(len(protocol['candidates'])) for m in methods]
        for future in as_completed(futures):
            row=future.result();rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2));print(row['method'],row['id'],row['verified_task_success'],row.get('failure_reason'),flush=True)
    (root/'run_summary.json').write_text(json.dumps({'wall_seconds':time.perf_counter()-started,'runs':len(rows)},indent=2))

if __name__=='__main__':main()
