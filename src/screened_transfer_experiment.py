"""Repair regression: fixed Cartesian lowering, raw vs common screened cover."""
import argparse,copy,json,os,shutil,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
from .prepare_transfer_scenes import design,prepare
from .screen_carry_regions import screen_scene,SPEC,OLD
from .transfer_scene_experiment import execute,run_process,METHODS


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--prepare');p.add_argument('--screen');a=p.parse_args();root=a.output
    if a.prepare:return prepare(root/'raw',a.prepare)
    if a.screen:return screen_scene(root,a.screen)
    root.mkdir();spec=design();spec['screen_spec']=SPEC;spec['scope']='Repair regression on the SAME eight physical conditions and 24 tasks from report50, not held-out evidence. Fresh libraries use corrected Cartesian lowering. Raw and screened conditions share actions; screening precedes all evaluated queries and uses only old witnesses plus fixed numerical probes. Raw condition is planning-only; screened condition has the same eight preselected contact tasks per method.'
    sources=list(Path('src').glob('*.py'));hashes={str(f):file_hash(f) for f in sources};spec['source_hashes']=hashes;atomic_json(root/'design.json',spec);(root/'source_snapshot').mkdir()
    for f in sources:shutil.copy2(f,root/'source_snapshot'/f.name)
    for condition in ['raw','screened']:(root/condition).mkdir();atomic_json(root/condition/'design.json',spec)
    cpus=sorted(os.sched_getaffinity(0));slots=[cpus[i] for i in [4,12,20,28]];names=list(spec['scenes']);begin=time.perf_counter()
    def prep_slot(i):
        for si in range(i,len(names),4):
            name=names[si];directory=root/'raw/libraries'/name;directory.mkdir(parents=True);print('START_PREP',name,flush=True)
            code,timeout=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.screened_transfer_experiment','--output',str(root),'--prepare',name],directory/'prepare.log',900)
            if code or timeout:raise RuntimeError('Shared preparation process failed: '+name+'; raw log retained')
            print('DONE_PREP',name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(prep_slot,range(4)))
    prep_seconds=time.perf_counter()-begin;screen_begin=time.perf_counter()
    def screen_slot(i):
        for si in range(i,len(names),4):
            name=names[si];directory=root/'screened/libraries'/name;directory.mkdir(parents=True);print('START_SCREEN',name,flush=True)
            code,timeout=run_process(['taskset','-c',str(slots[i]),sys.executable,'-m','src.screened_transfer_experiment','--output',str(root),'--screen',name],directory/'screen.log',900)
            if code or timeout:raise RuntimeError('Common screening process failed: '+name+'; log retained')
            print('DONE_SCREEN',name,flush=True)
    with ThreadPoolExecutor(4) as pool:list(pool.map(screen_slot,range(4)))
    screen_seconds=time.perf_counter()-screen_begin;changed=[f for f,h in hashes.items() if file_hash(f)!=h]
    if changed:raise RuntimeError('Sources changed during preparation: '+str(changed))
    resource_paths=[root/'design.json',OLD/'failure_witnesses.json']
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
