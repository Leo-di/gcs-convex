"""Frozen new cup placements; matched live searches with endpoint-check ordering."""
import argparse,copy,hashlib,itertools,json,os,signal,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed


def validate_bridge_yaws(config,candidates,method):
    start=float(config['robot']['start'][2])
    if method not in ('se2_gcs','rotation_safe_gcs'):
        if any(abs(float(c['base_q'][2])-start)>1e-10 for c in candidates):
            raise ValueError('Fixed-yaw bridge cannot connect rotated goal candidates')
        return
    allowed=config['green_room']['navigation_regions']
    intervals=[r['yaw_bounds'] for r in config['gcs_base_planning']['regions'] if r['name'] in allowed]
    if not all(any(lo<=yaw<=hi for lo,hi in intervals) for yaw in [start,*[float(c['base_q'][2]) for c in candidates]]):
        raise ValueError('SE(2) candidate yaw outside navigation domain; fix protocol before planning')


def search(root,task,method):
    from .semantic_refinement import screen_candidates
    from .scene_spec import load_scene_config
    from .grounding_search_controller import search_groundings
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());folder=root/task/method;folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();config=load_scene_config(root/'configs'/f'{task}.json')
    screened=screen_candidates(config,protocol['candidates']);(folder/'candidate_screen.json').write_text(json.dumps(screened,indent=2));candidates=[r for r in screened if r['status']=='eligible']
    validate_bridge_yaws(config,candidates,protocol.get('bridge_method','repair_gcs'))
    local=dict(protocol,config=str(root/'configs'/f'{task}.json'),candidates=candidates,endpoint_first=method in ('endpoint_first','operation_first'),operation_first=method=='operation_first',bridge_method=protocol.get('bridge_method','repair_gcs'))
    (folder/'protocol.json').write_text(json.dumps(local,indent=2))
    def evaluate(candidate,remaining):
        out=folder/candidate['id'];out.mkdir();index=candidates.index(candidate);tick=time.monotonic()
        with (out/'run.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.semantic_grounding_candidates','--output',str(folder),'--worker',str(index)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=min(remaining,protocol['per_candidate_seconds']))
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGTERM);proc.wait();code='timeout'
        file=out/'candidate_result.json';row=json.loads(file.read_text()) if file.exists() else {};row.update(process_exit=code,task=task,selection_method=method,candidate_wall_seconds=time.monotonic()-tick)
        file.write_text(json.dumps(row,indent=2));print(task,method,candidate['id'],code,row.get('failure_stage'),row.get('physical',{}).get('accepted',False),flush=True);return row
    result=search_groundings(candidates,evaluate,max(0,protocol['total_search_seconds']-(time.monotonic()-started)),folder/'search_result.json')
    result.update(task=task,method=method,total_wall_seconds=time.monotonic()-started,screened_candidates=len(screened),eligible_candidates=len(candidates));(folder/'search_result.json').write_text(json.dumps(result,indent=2));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',nargs=2);a=p.parse_args()
    if a.worker:return search(a.output,*a.worker)
    from .scene_spec import load_scene_config
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False);(root/'configs').mkdir()
    original=load_scene_config('configs/scene_semantic_choice_v6.yaml');tasks=[]
    for name,dx,dy in [('cup_a',.02,-.04),('cup_b',.06,0.),('cup_c',.10,.04)]:
        config=copy.deepcopy(original);config['objects']['cup']['center'][0]+=dx;config['objects']['cup']['center'][1]+=dy
        (root/'configs'/f'{name}.json').write_text(json.dumps(config,indent=2));tasks.append({'id':name,'cup_center':config['objects']['cup']['center'],'offset':[dx,dy]})
    candidates=[]
    for x,y,(j,angle) in itertools.product([.45,.65,.85,1.05],[-.15,0.,.15],enumerate([0.,-.2,.2])):
        candidates.append({'base_q':[x,y,0.],'grasp_rpy':[0.,0.,angle],'orientation_index':j,'ranking_score':((x+3)**2+y*y)**.5})
    candidates.sort(key=lambda c:(round(c['ranking_score'],10),c['orientation_index'],c['base_q']))
    for i,c in enumerate(candidates):c['id']=f'joint_{i:03d}'
    protocol={'tasks':tasks,'methods':['goal_only','endpoint_first'],'candidates':candidates,'seed':20261019,'total_search_seconds':300,'per_candidate_seconds':180,'scope':'Three newly frozen local cup-position perturbations, not diverse scenes or a statistical generalization study. No position changes after outcomes. Same candidate ordering, IK solver, GCS/IRIS and physical audits. Only endpoint-check order differs.','candidate_order':'Navigation distance, orientation index (0,-0.2,+0.2 yaw), base coordinates. Identical for both methods.','joint_filter_scope':'IK and whole-robot room checks at pregrasp, grasp and lift only. Passing does not certify a continuous operation path.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));manifest={}
    for f in Path('src').glob('*.py'):
        dest=root/'source_snapshot'/f;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes());manifest[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2));rows=[];began=time.monotonic()
    def run(task,method):
        with (root/f'{task}_{method}.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.joint_grounding_study','--output',str(root),'--worker',task,method],stdout=log,stderr=subprocess.STDOUT)
            code=proc.wait()
        path=root/task/method/'search_result.json';row=json.loads(path.read_text()) if path.exists() else {'task':task,'method':method,'success':False,'reason':'search worker failed'};row['search_exit']=code;return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,t['id'],m) for t in tasks for m in protocol['methods']]
        for future in as_completed(futures):
            row=future.result();rows.append(row);(root/'comparison.json').write_text(json.dumps(rows,indent=2));print(row['task'],row['method'],row['success'],row.get('selected'),flush=True)
    (root/'run_summary.json').write_text(json.dumps({'runs':len(rows),'parallel_wall_seconds':time.monotonic()-began},indent=2))

if __name__=='__main__':main()
