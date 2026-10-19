"""Frozen two-target full-pipeline regression of rotation-safe navigation."""

def main():
    import copy,hashlib,json,subprocess,sys
    from concurrent.futures import ThreadPoolExecutor,as_completed
    from pathlib import Path
    root=Path('outputs/results/rotation_safe_full_20261019');root.mkdir(exist_ok=False);(root/'configs').mkdir()
    old=Path('outputs/results/se2_rotating_goal_corrected_20261019');prior=json.loads((old/'protocol.json').read_text());selected=[c for c in prior['candidates'] if c['id'] in ['se2_020','se2_021']]
    protocol=copy.deepcopy(prior);protocol.update(bridge_method='rotation_safe_gcs',total_search_seconds=360,per_candidate_seconds=360,tasks=[{'id':c['id']} for c in selected],candidates=selected,scope='Two previously operation-valid candidates, frozen before new bridge outcomes; targeted regression, not random success-rate estimate.')
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest={}
    for p in Path('src').glob('*.py'):
     dest=root/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());manifest[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    # Each candidate is an independent fixed target, not a sequential selection.
    for c in selected:
     (root/'configs'/f"{c['id']}.json").write_bytes((old/'configs/screen_narrow.json').read_bytes())
     folder=root/c['id']/'operation_first';folder.mkdir(parents=True)
     local=dict(protocol,candidates=[c],config=str(root/'configs'/f"{c['id']}.json"),endpoint_first=True,operation_first=True)
     (folder/'protocol.json').write_text(json.dumps(local,indent=2))
    def run(c):
     import os,signal,time
     folder=root/c['id']/'operation_first';t=time.monotonic()
     with (root/f"{c['id']}.log").open('w') as log:
      p=subprocess.Popen([sys.executable,'-m','src.semantic_grounding_candidates','--output',str(folder),'--worker','0'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      try:code=p.wait(timeout=360)
      except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait();code='timeout'
     path=folder/c['id']/'candidate_result.json';d=json.loads(path.read_text()) if path.exists() else {};d['process_exit']=code;path.write_text(json.dumps(d,indent=2));ok=code==0 and all(d.get(k,{}).get('accepted') for k in ['bridge','operation','physical'])
     row={'task':c['id'],'method':'operation_first','success':ok,'selected':c['id'] if ok else None,'total_wall_seconds':time.monotonic()-t,'attempts':[{'id':c['id'],'complete':ok}],'search_exit':code};(folder/'search_result.json').write_text(json.dumps(row,indent=2));print(row,flush=True);return row
    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
     for f in as_completed([pool.submit(run,c) for c in selected]):rows.append(f.result());(root/'comparison.json').write_text(json.dumps(rows,indent=2))

if __name__ == "__main__":
    main()
