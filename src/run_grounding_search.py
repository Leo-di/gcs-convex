"""Live sequential search; descending candidate order is a development stress test."""
import argparse,json,os,signal,subprocess,sys,hashlib,shutil
from pathlib import Path
from .grounding_search_controller import search_groundings


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(Path('outputs/results/semantic_grounding_choices_20261019/protocol.json').read_text());protocol.update(bridge_method='repair_gcs',seed=20261019,total_search_budget_s=600,scope='Live sequential controller development demonstration, descending original distance order. Previously studied candidates; not an unbiased success-rate experiment.')
    protocol['candidates']=list(reversed(protocol['candidates']));(root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    files=['src/grounding_search_controller.py','src/run_grounding_search.py','src/automatic_motion_bridge.py','src/semantic_grounding_candidates.py'];manifest={}
    for name in files:
        p=Path(name);dst=root/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    def evaluate(candidate,remaining):
        index=protocol['candidates'].index(candidate);folder=root/candidate['id'];folder.mkdir()
        with (folder/'run.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.semantic_grounding_candidates','--output',str(root),'--worker',str(index)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait();code='timeout'
        path=folder/'candidate_result.json';row=json.loads(path.read_text()) if path.exists() else {}
        row['process_exit']=code;path.write_text(json.dumps(row,indent=2));print(candidate['id'],code,row.get('stage'),row.get('failure_reason'),flush=True);return row
    print(search_groundings(protocol['candidates'],evaluate,600,root/'search_result.json'),flush=True)

if __name__=='__main__':main()
