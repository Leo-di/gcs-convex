"""Bounded state-derived transport posture search with complete-action acceptance.

A coarse disk-cover screen only orders trials. A disconnected screen is not
an infeasibility certificate; deferred trials remain available to the bounded
full planner. Every selection must pass reconfiguration, GCS transport,
restoration, and complete world-state/semantic audits supplied by the caller.
"""
import itertools
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .build_scene import set_robot_configuration
from .grasp_planner import solve_arm_ik
from .payload_translation_bridge import PayloadTranslationOracle

DEFAULT_POLICY = dict(radial_fractions=[1., .85, .7, .55],
    lateral_fractions=[0., -.15, .15], height_offsets_m=[0., .1],
    max_candidates=10, max_full_plans=4, time_budget_s=600., screen_spacing_m=.12)


def validate_search(policy, actions=()):
    if policy is None:return
    if not isinstance(policy,dict) or set(policy)!=set(DEFAULT_POLICY):
        raise ValueError('Explicit bounded transport posture search policy required')
    for key in ('radial_fractions','lateral_fractions','height_offsets_m'):
        a=np.asarray(policy[key],float)
        if a.ndim!=1 or not len(a) or not np.all(np.isfinite(a)) or len(np.unique(a))!=len(a):
            raise ValueError('Distinct finite transport search axes required')
    r=np.asarray(policy['radial_fractions']);l=np.asarray(policy['lateral_fractions']);h=np.asarray(policy['height_offsets_m'])
    if np.any(r<=0) or np.any(r>1) or 1. not in r or 0. not in l or 0. not in h:
        raise ValueError('Search must contain identity and positive inward radial fractions')
    for key in ('max_candidates','max_full_plans'):
        if type(policy[key]) is not int or not 1<=policy[key]<=64:raise ValueError('Bounded integer search budgets required')
    for key in ('time_budget_s','screen_spacing_m'):
        if isinstance(policy[key],bool) or not np.isfinite(policy[key]) or policy[key]<=0:raise ValueError('Positive search timing and spacing required')
    if any('transport_offset_m' in action for action in actions):
        raise ValueError('Automatic transport search forbids per-action transport_offset_m')


def generate_candidates(scene,base,arm,policy):
    validate_search(policy)
    set_robot_configuration(scene,base,arm,scene.gripper_closed_positions)
    X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)
    p=X.multiply(np.asarray(scene.config['gripper']['grasp_point_in_body']))
    radial=p[:2]-np.asarray(base[:2]);length=float(np.linalg.norm(radial))
    if length<1e-8:raise ValueError('Transport search requires a nonzero horizontal grasp standoff')
    lateral=np.array([-radial[1],radial[0]])
    rows=[]
    for fraction,side,height in itertools.product(policy['radial_fractions'],policy['lateral_fractions'],policy['height_offsets_m']):
        offset=np.r_[(fraction-1)*radial+side*lateral,height]
        rows.append(dict(offset_world_m=offset.tolist(),radial_fraction=fraction,lateral_fraction=side,
            height_offset_m=height,movement_cost_m=float(np.linalg.norm(offset)+.25*abs(height)),
            horizontal_standoff_m=float(np.linalg.norm(radial+offset[:2]))))
    order=lambda r:(round(r['movement_cost_m'],10),r['horizontal_standoff_m'],r['offset_world_m'])
    rows.sort(key=order)
    # Reserve every radial level before filling with cheap lateral/height variants.
    # A pure cost truncation can otherwise omit the compact end of the search.
    selected=[r for r in rows if r['lateral_fraction']==0 and r['height_offset_m']==0][:policy['max_candidates']]
    for row in rows:
        if len(selected)>=policy['max_candidates']:break
        if row not in selected:selected.append(row)
    selected.sort(key=order)
    for i,row in enumerate(selected):row['id']=f'posture_{i:03d}'
    return selected


def screen_candidate(scene,base,destination,arm,X_GC,offset,carry,action_id,spacing):
    """Finite conservative coarse disk graph; only a search priority signal."""
    began=time.perf_counter();offset=np.asarray(offset,float)
    if np.linalg.norm(offset)>1e-10:
        set_robot_configuration(scene,base,arm,scene.gripper_closed_positions)
        X=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)
        p=X.multiply(np.asarray(scene.config['gripper']['grasp_point_in_body']))
        arm=solve_arm_ik(scene,base_q=np.asarray(base),target_position=p+offset,target_orientation=X.rotation(),
            initial_arm_q=np.asarray(arm),position_tolerance=.0002,orientation_tolerance=.005,
            minimum_clearance=scene.config['pick_task']['minimum_clearance'],stage='posture_screen')
    margin=float(carry['margin_m']);cap=float(carry['max_radius_m'])
    oracle=PayloadTranslationOracle(scene,arm,scene.gripper_closed_positions,base[2],X_GC,
        action_id=action_id,max_distance=cap+margin)
    lo=np.asarray(carry['lower_xy']);hi=np.asarray(carry['upper_xy'])
    seeds=[np.asarray(base[:2]),np.asarray(destination[:2])]
    seeds.extend(np.array([x,y]) for x in np.arange(lo[0],hi[0]+1e-9,spacing)
                 for y in np.arange(lo[1],hi[1]+1e-9,spacing))
    centers=[];radii=[];indices=[];minimum=[]
    for i,p in enumerate(seeds):
        d,internal,_=oracle.query(p);r=min(cap,d-margin)
        if i<2:minimum.append(dict(clearance_m=d,internal_m=internal))
        if internal>=-1e-7 and r>=carry.get('min_radius_m',.025):
            centers.append(p);radii.append(r);indices.append(i)
    count=len(centers);connected=False;edges=0;gap=None
    if 0 in indices and 1 in indices:
        centers=np.asarray(centers);radii=np.asarray(radii)
        pairs=cKDTree(centers).query_pairs(2*cap,output_type='ndarray')
        if len(pairs):
            distances=np.linalg.norm(centers[pairs[:,0]]-centers[pairs[:,1]],axis=1)
            pairs=pairs[distances<radii[pairs[:,0]]+radii[pairs[:,1]]-1e-8]
        graph=coo_matrix((np.ones(len(pairs)),(pairs[:,0],pairs[:,1])),shape=(count,count))
        _,labels=connected_components(graph,directed=False)
        source_label=labels[indices.index(0)];target_label=labels[indices.index(1)]
        connected=bool(source_label==target_label);edges=len(pairs)*2
        if connected:gap=0.
        else:
            a=np.flatnonzero(labels==source_label);b=np.flatnonzero(labels==target_label);gap=float('inf')
            for chunk in np.array_split(a,max(1,int(np.ceil(len(a)/256)))):
                d=np.linalg.norm(centers[chunk,None,:]-centers[None,b,:],axis=2)-radii[chunk,None]-radii[None,b]
                gap=min(gap,float(d.min()))
    return dict(connected=connected,component_gap_m=gap,regions=count,directed_edges=edges,endpoints=minimum,
        oracle_calls=oracle.calls,wall_seconds=time.perf_counter()-began,
        scope='Coarse fixed-arm/yaw disk cover; component gap only ranks deferred trials; disconnected is not physically infeasible; final continuation IK posture may differ')



class PostureSearchRejected(RuntimeError):
    def __init__(self,report):
        self.stage='transport_posture_search_exhausted';self.details=report
        super().__init__('Bounded transport posture search exhausted; no global infeasibility claim')


def select_posture(candidates,screen,evaluate,policy,artifact):
    """Callbacks isolate scenes; only a fully accepted action may be selected."""
    began=time.monotonic();artifact=Path(artifact);rows=[];deferred=[];full_plans=0
    report=dict(method='state_derived_bounded_transport_posture_search',policy=policy,
        generated_candidates=candidates,trials=rows,selected=None,
        budget_scope='Time budget stops admission of new trials; an in-flight bounded stage may finish later',
        scope='Coarse-connectivity guided finite Cartesian posture family with continuation IK; cost ordering for connected screens, component-gap ordering for deferred screens; first fully accepted candidate; no global posture/path optimum or completeness claim')
    def save():
        report.update(wall_seconds=time.monotonic()-began,full_plans=full_plans)
        artifact.write_text(json.dumps(report,indent=2))
    def attempt(candidate,row):
        nonlocal full_plans
        if full_plans>=policy['max_full_plans'] or time.monotonic()-began>=policy['time_budget_s']:return None
        full_plans+=1;row['full_plan_attempted']=True;save()
        try:
            answer=evaluate(candidate)
            if answer is None:raise RuntimeError('Full evaluator returned no accepted action')
        except Exception as exc:
            row.update(status='full_action_rejected',reason=str(exc),failure_stage=getattr(exc,'stage',type(exc).__name__),details=getattr(exc,'details',None));save();return None
        row['status']='fully_accepted';report['selected']=candidate;save()
        return answer,report
    for candidate in candidates:
        if full_plans>=policy['max_full_plans'] or time.monotonic()-began>=policy['time_budget_s']:break
        row=dict(id=candidate['id'],offset_world_m=candidate['offset_world_m'],full_plan_attempted=False);rows.append(row)
        try:row['screen']=screen(candidate)
        except Exception as exc:
            row.update(status='screen_error_deferred',reason=str(exc));deferred.append((candidate,row));save();continue
        if row['screen']['connected']:
            row['status']='screen_connected';answer=attempt(candidate,row)
            if answer is not None:return answer
        else:row['status']='coarse_disconnected_deferred';deferred.append((candidate,row))
        save()
    deferred.sort(key=lambda item:(item[1].get('screen',{}).get('component_gap_m') if item[1].get('screen',{}).get('component_gap_m') is not None else float('inf'),item[0].get('movement_cost_m',0.)))
    for candidate,row in deferred:
        answer=attempt(candidate,row)
        if answer is not None:return answer
    save();raise PostureSearchRejected(report)
