"""Repair regression: fixed Cartesian lowering, raw vs common screened cover."""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
UPSTREAM=Path('outputs/results/navigation_local_refinement_v1')

def design():
    spec=json.loads((UPSTREAM/'design.json').read_text())
    spec['scope']='Repair regression on the SAME16 previously diagnosed scenes. Shared candidate libraries come from fixed local .30/.15m navigation-cover refinement. Not heldout generalization. Raw/screened compare common numerical region screening, no old witnesses. Same official GCS10/GCS50 and global portal+SOCP query methods; 48 tasks each, 288 plans, screened16 scheduled contact opportunities per method. Preparation timing in this run is COPY ONLY; upstream construction cost is saved separately.'
    return spec

def prepare(root,name):
    source=UPSTREAM/'libraries'/name;dest=root/'libraries'/name;dest.mkdir(parents=True,exist_ok=True)
    prep=json.loads((source/'preparation.json').read_text())
    if prep.get('failure'):
        config=json.loads((root/'design.json').read_text())['scenes'][name]
        prep.update(candidate_count=0,goal_count=0,connectivity=[],regions=0)
        atomic_json(dest/'shared.json',{'config':config,'candidates':{},'goals':{},'disks':[]})
        atomic_json(dest/'preparation.json',prep)
    else:
        for f in ['shared.json','preparation.json']:shutil.copy2(source/f,dest/f)

from .screen_carry_regions import screen_scene,SPEC
from .transfer_scene_experiment import execute,run_process,METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare');p.add_argument('--screen');a=p.parse_args();root=a.output
    if a.prepare:return prepare(root/'raw',a.prepare)
    if a.screen:return screen_scene(root,a.screen,use_known_witnesses=False)
    finished=json.loads((UPSTREAM/'full_complete.json').read_text())
    if finished['changed_inputs']:raise RuntimeError('Upstream frozen inputs changed')
    root.mkdir();atomic_json(root/'upstream_cost.json',{'full_extension':finished,'pilot':json.loads((UPSTREAM/'pilot_complete.json').read_text()),'note':'This run reuses libraries; copy time is not cold-start preparation time. Pilot and extension ran with other diagnostics, so times are descriptive.'})
    shutil.copytree(UPSTREAM/'extension_source_snapshot',root/'upstream_source_snapshot')
    spec=design();spec['screen_spec']=SPEC;
    sources=list(Path('src').glob('*.py'));hashes={str(f):file_hash(f) for f in sources};spec['source_hashes']=hashes;atomic_json(root/'design.json',spec);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    for condition in ['raw','screened']:(root/condition).mkdir();atomic_json(root/condition/'design.json',spec)
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[i] for i in [4,12,20,28]];names=list(spec['scenes']);begin=time.perf_counter()
    def prep_slot(i):
        for si in range(i,len(names),4):
            name=names[si];directory=root/'raw/libraries'/name;directory.mkdir(parents=True);print('START_PREP',name,flush=True)
            code,timeout=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.refined_transfer_experiment','--output',str(root),'--prepare',name],directory/'prepare.log',900)
            if code or timeout:
                atomic_json(directory/'preparation.json',{'failure':'preparation_timeout' if timeout else 'preparation_program_error','return_code':code,'attempts':[],'goal_attempts':[],'connectivity':[],'candidate_count':0,'goal_count':0,'regions':0})
                atomic_json(directory/'shared.json',{'config':spec['scenes'][name],'candidates':{},'goals':{},'disks':[]})
            print('DONE_PREP',name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(prep_slot,range(4)))
    prep_seconds=time.perf_counter()-begin;screen_begin=time.perf_counter()
    def screen_slot(i):
        for si in range(i,len(names),4):
            name=names[si];directory=root/'screened/libraries'/name;directory.mkdir(parents=True);print('START_SCREEN',name,flush=True)
            code,timeout=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.refined_transfer_experiment','--output',str(root),'--screen',name],directory/'screen.log',900)
            if code or timeout:
                raw=root/'raw/libraries'/name;shared=json.loads((raw/'shared.json').read_text());pre=json.loads((raw/'preparation.json').read_text())
                for candidate in shared['candidates'].values():candidate['cover']['regions']=[]
                pre.update(failure='screen_timeout' if timeout else 'screen_program_error',regions=0,before_screen_connectivity=pre['connectivity'])
                pre['connectivity']=[dict(x,region_graph_connected=False) for x in pre['connectivity']]
                atomic_json(directory/'shared.json',shared);atomic_json(directory/'preparation.json',pre)
                atomic_json(directory/'screen_report.json',{'failure':pre['failure'],'spec':SPEC,'use_known_witnesses':False,'candidates':{},'seconds':900 if timeout else None,'before_connected':sum(x['region_graph_connected'] for x in pre['before_screen_connectivity']),'after_connected':0})
            print('DONE_SCREEN',name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(screen_slot,range(4)))
    screen_seconds=time.perf_counter()-screen_begin;changed=[f for f,h in hashes.items() if file_hash(f)!=h]
    if changed:raise RuntimeError('Sources changed during preparation: '+str(changed))
    resource_paths=[root/'design.json']
    for condition in ['raw','screened']:resource_paths+=list((root/condition/'libraries').glob('*/shared.json'))+list((root/condition/'libraries').glob('*/preparation.json'))
    hashes.update({str(f):file_hash(f) for f in resource_paths});atomic_json(root/'preparation_complete.json',{'preparation_seconds':prep_seconds,'screen_seconds':screen_seconds,'changed_inputs':changed})
    for condition in ['raw','screened']:
        cases=copy.deepcopy(spec['cases'])
        if condition=='raw':
            for c in cases:c['contact']=False
        protocol={'cases':cases,'methods':METHODS,'hashes':hashes,'scope':spec['scope'],'condition':condition,'budget_per_stage_s':300,'preparation_wall_seconds':prep_seconds,'screen_wall_seconds':screen_seconds}
        atomic_json(root/condition/'protocol.json',protocol)
    query_begin=time.perf_counter()
    def slot(i):
        for j in range(i,len(spec['cases']),4):
            c=spec['cases'][j];conditions=['raw','screened'] if j%2==0 else ['screened','raw']
            for condition in conditions:
                for m in METHODS[j%3:]+METHODS[:j%3]:
                    print('START',condition,c['id'],m,flush=True);r=execute(root/condition,c['id'],m,slots[i],'plan')
                    contact=condition=='screened' and c['contact']
                    if contact and r['planning_success']:r=execute(root/condition,c['id'],m,slots[i],'contact')
                    r.update(contact_selected=contact,scene=c['scene'],condition=condition);atomic_json(root/condition/c['id']/m/'record.json',r)
                    print('DONE',condition,c['id'],m,r['planning_success'],r['end_to_end_success'],r.get('failure',''),flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(slot,range(4)))
    allrows=[]
    for condition in ['raw','screened']:
        rows=[json.loads(f.read_text()) for f in (root/condition).glob('*__q*/*/record.json')];atomic_json(root/condition/'summary.json',{'records':rows});allrows+=rows
    atomic_json(root/'summary.json',{'records':allrows});atomic_json(root/'complete.json',{'runs':len(allrows),'preparation_seconds':prep_seconds,'screen_seconds':screen_seconds,'query_contact_seconds':time.perf_counter()-query_begin,'total_seconds':time.perf_counter()-begin,'changed_inputs':[f for f,h in hashes.items() if file_hash(f)!=h]});print('COMPLETE',flush=True)


if __name__=='__main__':main()
