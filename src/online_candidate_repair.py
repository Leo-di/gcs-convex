"""Identical solve/audit/retry or counterexample repair loop for all methods."""
import copy,json
import numpy as np
from .semantic_candidate_gcs import build_and_solve,expand,q,PICK_PHASES
from .carry_mode_gcs import CarryOracle


def solve_audited(disks,candidates,allowed,config,directory,start,seed,paths,mode='none',rounds=0):
    candidates=copy.deepcopy(candidates);history=[];total_build=0.;total_solve=0.
    for iteration in range(rounds+1):
        dest=directory/f'round{iteration}';dest.mkdir(parents=True,exist_ok=True)
        result,path=build_and_solve(disks,candidates,allowed,config,start_base_q=start,
            rounding_seed=seed+iteration*101,max_rounded_paths=paths)
        total_build+=result['build_seconds'];total_solve+=result['solve_seconds']
        (dest/'solve.json').write_text(json.dumps(result,indent=2))
        audit=expand(path,candidates,disks,config,dest,visualize=False) if path is not None else None
        item={'iteration':iteration,'rounding_seed':seed+iteration*101,'solver_status':result['status'],'accepted':bool(audit and audit['accepted']),'reference':str(dest)}
        history.append(item)
        if item['accepted'] or iteration==rounds:
            break
        if mode=='cuts' and audit and not audit['dense']['satisfied']:
            violation=audit['dense']['first_violation'];rows=json.loads((dest/'delivery_trace.json').read_text())
            edge=violation['edge'];a,b=rows[edge:edge+2]
            if not a['attached'] or not b['attached']:
                item['repair_status']='noncarrying_violation_not_repaired'
            else:
                state=(1-violation['alpha'])*q(a)+violation['alpha']*q(b)
                name=result['selected_candidate'];candidate=candidates[name]
                anchor=[r for r in candidate['rows'] if r['phase'] in PICK_PHASES][-1]
                oracle=CarryOracle(config,anchor)
                def distance(x):return oracle.query(x[:2],x[2:])[1]
                d=distance(state);h=1e-5
                grad=np.array([(distance(state+np.eye(10)[j]*h)-distance(state-np.eye(10)[j]*h))/(2*h) for j in range(10)])
                bound=d-.002-grad@state;changed=[]
                for i,region in enumerate(candidate['cover']['regions']):
                    A=np.array(region['A']);B=np.array(region['b'])
                    if np.max(A@state-B)>1e-6 or -grad@np.array(region['seed'])>bound:continue
                    region['A']=np.vstack([A,-grad]).tolist();region['b']=np.r_[B,bound].tolist()
                    region['number_of_faces']=len(region['b'])
                    region['audit_scope']='Stored holdout audit predates this online cut; not rerun. Final trajectory independently audited.'
                    changed.append(i)
                item.update(repair_status='cut' if changed else 'no_seed_preserving_cut',candidate=name,cut_regions=changed,distance=d)
                (dest/'updated_cover.json').write_text(json.dumps(candidate['cover']))
        (directory/'repair_history.json').write_text(json.dumps(history,indent=2))
    (directory/'repair_history.json').write_text(json.dumps(history,indent=2))
    result['build_seconds']=total_build;result['solve_seconds']=total_solve
    return result,audit,dest,history
