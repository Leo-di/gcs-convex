"""Paired post-hoc removal of one witnessed colliding region, no new regions."""
import json,shutil,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from .benchmark_semantic_baselines import atomic_json,file_hash
from .transfer_scene_experiment import execute,METHODS
from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES
from .online_carry_iris import sets
from .online_iris_attachment import connection_status
ROOT=Path('outputs/results/transfer_region_counterexample_v1')
BASE=Path('outputs/results/transfer_scene_evaluation_v1')


def main():
    ROOT.mkdir();directory=ROOT/'libraries/cart_south';directory.mkdir(parents=True);shared=json.loads((BASE/'libraries/cart_south/shared.json').read_text());prep=json.loads((BASE/'libraries/cart_south/preparation.json').read_text());witness=json.loads((BASE/'collision_diagnostic.json').read_text())
    assert witness['candidate']=='x115' and witness['inside_regions']==[3]
    removed=shared['candidates']['x115']['cover']['regions'].pop(3);atomic_json(ROOT/'removed_region.json',removed);shutil.copy2(BASE/'collision_diagnostic.json',ROOT/'collision_diagnostic.json');prep['counterexample_removed']={'candidate':'x115','original_region_index':3};prep['connectivity']=[]
    for name,c in shared['candidates'].items():
        start=q([r for r in c['rows'] if r['phase'] in PICK_PHASES][-1]);rs=sets(c['cover']['regions'])
        for gid,goal in shared['goals'][name].items():
            end=q([r for r in goal['rows'] if r['phase'] in PLACE_PHASES][0]);prep['connectivity'].append({'candidate':name,'goal':gid,**connection_status(rs,start,end)})
    atomic_json(directory/'shared.json',shared);atomic_json(directory/'preparation.json',prep);old=json.loads((BASE/'protocol.json').read_text());cases=[{**c,'contact':False} for c in old['cases'] if c['scene']=='cart_south']
    resources=list(Path('src').glob('*.py'))+[directory/'shared.json',directory/'preparation.json',ROOT/'removed_region.json',ROOT/'collision_diagnostic.json'];hashes={str(f):file_hash(f) for f in resources}
    atomic_json(ROOT/'protocol.json',{'cases':cases,'methods':METHODS,'hashes':hashes,'scope':'Post-hoc paired counterfactual on the three observed cart_south tasks: delete exactly one witnessed unsafe region from the shared x115 cover for ALL methods. No new actions/regions, no further repair, no contact execution. Does not replace frozen transfer-scene results or certify other regions.'})
    cpus=sorted(os.sched_getaffinity(0));begin=time.perf_counter()
    def job(i):
        c=cases[i]
        for m in METHODS:
            r=execute(ROOT,c['id'],m,cpus[[4,12,20][i]],'plan');print(c['id'],m,r['planning_success'],r.get('failure',''),flush=True)
    with ThreadPoolExecutor(3) as pool:list(pool.map(job,range(3)))
    records=[json.loads(f.read_text()) for f in ROOT.glob('*__q*/*/record.json')];atomic_json(ROOT/'summary.json',{'records':records});atomic_json(ROOT/'complete.json',{'runs':len(records),'seconds':time.perf_counter()-begin,'changed_inputs':[f for f,h in hashes.items() if file_hash(f)!=h]})


if __name__=='__main__':main()
