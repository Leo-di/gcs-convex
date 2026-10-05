"""Audit finite-library compatibility and additive placement-cost interaction."""
import json
from pathlib import Path
import numpy as np
from .semantic_candidate_gcs import primitive_cost,PLACE_PHASES
from .benchmark_semantic_baselines import atomic_json


def main():
    root=Path('outputs/results/coupled_goal_library_v1');shared=json.loads((root/'shared.json').read_text());goals=shared['goals'];names=list(goals)
    union=set.union(*[set(g) for g in goals.values()]);common=sorted(set.intersection(*[set(g) for g in goals.values()]))
    cost=np.array([[primitive_cost([r for r in goals[n][g]['rows'] if r['phase'] in PLACE_PHASES]) for g in common] for n in names])
    residual=cost-cost.mean(axis=1,keepdims=True)-cost.mean(axis=0,keepdims=True)+cost.mean()
    result={'grasp_names':names,'goal_ids_union':sorted(union),'common_goal_ids':common,
            'missing_library_pairs':[{'grasp':n,'goal':g} for n in names for g in sorted(union) if g not in goals[n]],
            'placement_primitive_cost_matrix':cost.tolist(),'additive_row_column_residual':residual.tolist(),
            'interaction_rms':float(np.sqrt(np.mean(residual**2))),'interaction_max_abs':float(abs(residual).max()),
            'relative_interaction_rms_percent':float(100*np.sqrt(np.mean(residual**2))/cost.mean()),
            'scope':'Finite primitive catalog only; failure to generate a primitive is not a physical infeasibility proof. Non-additivity does not imply GCS must outperform enumeration.'}
    atomic_json(root/'coupling_diagnostic.json',result)


if __name__=='__main__':main()
