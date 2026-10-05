"""Post-hoc task_11 diagnosis; never replace the paired benchmark results."""
import copy,importlib.util,json,time
from pathlib import Path
import numpy as np
from .benchmark_semantic_baselines import atomic_json,file_hash
from .coupled_goal_experiment import options,materialize,path_cost
from .semantic_candidate_gcs import expand,WEIGHTS
ROOT=Path('outputs/results/goal_rounding_diagnostic_v1')
SOURCE=Path('outputs/results/coupled_goal_evaluation_v1/source_snapshot/semantic_candidate_gcs.py')


def main():
    ROOT.mkdir();shared_path=Path('outputs/results/coupled_goal_evaluation_v1/shared.json');shared=json.loads(shared_path.read_text());cases=json.loads(Path('outputs/results/coupled_goal_evaluation_v1/protocol.json').read_text())['cases'];cell=next(c for c in cases if c['id']=='task_11')
    config=copy.deepcopy(shared['config']);config['inspection_sites']=cell['sites'];goals=options(shared,cell)
    source=SOURCE.read_text();needle='    selected=g.GetSolutionPath(source,target,result)'
    assert source.count(needle)==1
    source=source.replace(needle,"    if max_rounded_paths<=0:return {**info,'status':'relaxation_only','objective':float(result.get_optimal_cost())},None\n"+needle)
    model=ROOT/'model.py';model.write_text(source)
    spec=importlib.util.spec_from_file_location('src._rounding_diagnostic_model',model);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    settings=[{'paths':k,'seed':s} for k in [10,50,100] for s in [0,1,2]]
    atomic_json(ROOT/'protocol.json',{'task':cell,'settings':settings,'scope':'Post-hoc diagnosis of an observed bad case; not held-out evidence, no replacement of original results. Same mathematical graph; only relaxation-only return added to frozen model. Each run audited, no contact rerun. Numerical solver lower bound, not an exact arithmetic certificate.','hashes':{str(p):file_hash(p) for p in [SOURCE,model,shared_path,Path(__file__)]}})
    t=time.perf_counter();lower,_=module.build_and_solve(shared['disks'],shared['candidates'],cell['allowed'],config,start_base_q=cell['start'],goal_options=goals,max_rounded_paths=0);lower['wall_seconds']=time.perf_counter()-t;atomic_json(ROOT/'relaxation.json',lower);rows=[]
    for setting in settings:
        out=ROOT/f"paths_{setting['paths']}_seed_{setting['seed']}";out.mkdir();t=time.perf_counter()
        result,path=module.build_and_solve(shared['disks'],shared['candidates'],cell['allowed'],config,start_base_q=cell['start'],goal_options=goals,max_rounded_paths=setting['paths'],rounding_seed=setting['seed']);elapsed=time.perf_counter()-t;atomic_json(out/'plan.json',result)
        row={**setting,'seconds':elapsed,'status':result['status'],'accepted':False}
        if path is not None:
            cs,c=materialize(shared,result['selected_candidate'],result['selected_goal'],config);audit=expand(path,cs,shared['disks'],c,out,visualize=False)
            parts={}
            for v in path:
                if v['kind'] in ['empty','carry']:parts[v['kind']]=parts.get(v['kind'],0)+float(np.linalg.norm(WEIGHTS*(np.array(v['x'][10:])-v['x'][:10])))
            row.update(accepted=audit['accepted'],cost=path_cost(path,cs),objective_with_tiebreak=result['objective'],motion_costs=parts,goal=result['selected_goal'],inspection_site=result['inspection_site'])
            if lower['status']=='relaxation_only':row['numerical_relaxation_gap_percent']=100*(result['objective']/lower['objective']-1)
        rows.append(row);atomic_json(ROOT/'summary.json',{'lower_bound':lower,'runs':rows});print(json.dumps(row),flush=True)
    atomic_json(ROOT/'complete.json',{'runs':len(rows),'all_audited':all(r['accepted'] for r in rows)})


if __name__=='__main__':main()
