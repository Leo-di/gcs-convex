"""Check saved Bézier coefficients independently of dense trajectory sampling."""
import argparse
import json
from pathlib import Path

import numpy as np
from .joint_cspace_research import write_json


def validate(source, labels=('collision_only','semantic')):
    output={}
    for label in labels:
        plan=json.loads((source/f'{label}_plan.json').read_text())
        regions=[r for r in json.loads((source/f'{label}_regions.json').read_text()) if r['accepted']]
        if not plan.get('success'):
            output[label]=dict(checked=False,reason='No successful GCS trajectory')
            continue
        maximum_speed=0.;maximum_c0=0.;maximum_speed_jump=0.;previous=None;previous_velocity=None
        upper_length=0.;containments=[]
        for segment in plan['segments']:
            controls=np.array(segment['control_points']);degree=controls.shape[1]-1
            containing=[]
            for j,region in enumerate(regions):
                violation=np.array(region['A'])@controls-np.array(region['b'])[:,None]
                if violation.max()<=1e-6:containing.append(j)
            containments.append(containing)
            if previous is not None:
                maximum_c0=max(maximum_c0,float(np.linalg.norm(controls[:,0]-previous)))
            previous=controls[:,-1]
            if degree:
                dt=segment['end_s']-segment['start_s']
                if dt<=0:raise ValueError('Nonconstant segment has nonpositive duration')
                derivative=degree*np.diff(controls,axis=1)/dt
                maximum_speed=max(maximum_speed,float(np.max(np.abs(derivative))))
                upper_length+=float(np.linalg.norm(np.diff(controls,axis=1),axis=0).sum())
                if previous_velocity is not None:
                    maximum_speed_jump=max(maximum_speed_jump,float(np.max(np.abs(derivative[:,0]-previous_velocity))))
                previous_velocity=derivative[:,-1]
        objective_reconstructed=upper_length+.05*plan['duration_s']
        objective_error=abs(objective_reconstructed-plan['objective'])
        output[label]=dict(checked=True,all_control_polygons_contained=all(containments),
                           containing_regions=containments,max_control_point_speed_rad_s=maximum_speed,
                           max_position_jump_rad=maximum_c0,max_time_velocity_jump_rad_s=maximum_speed_jump,
                           path_length_upper_bound_rad=upper_length,objective_reconstruction_error=objective_error,
                           passed=bool(all(containments) and maximum_speed<=.60001 and maximum_c0<1e-6 and objective_error<1e-4),
                           scope='Numerical coefficient check: convex-hull containment and derivative bounds conditional on region validity; no collision certificate')
    write_json(source/'coefficient_audit.json',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--conditions',nargs='+',choices=['collision_only','semantic'],default=['collision_only','semantic'])
    args=parser.parse_args()
    print(json.dumps(validate(args.source,args.conditions),indent=2))
