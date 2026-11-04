"""Matched carrying replanning on one immutable physical region pool.

Early certification and lazy plan/check/repair differ only in when they ask
for human-access region certificates. GCS always chooses an actual overlap
path. A failed bounded search never means physical/task infeasibility.
"""
from __future__ import annotations

import copy
import hashlib
import json
import time

import numpy as np

from .multibranch_payload import GcsSolveFailure,component_indices,terminal_path_indices,endpoint_ids,solve_graph,to_sets
from .payload_translation_bridge import overlap_edges


def pool_digest(records):
    return hashlib.sha256(json.dumps(records,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def compile_graph(pool, certificates, start, goal):
    """Apply known radius bounds, then rebuild exact intersections and IDs."""
    admitted=[]
    for i,physical in enumerate(pool):
        record=copy.deepcopy(physical)
        if i in certificates:
            bounded,_=certificates[i]
            if bounded is None:
                continue
            # Certified regions must remain subsets of their physical regions.
            if (not np.array_equal(record['center'],bounded['center'])
                    or not np.array_equal(record['lower'],bounded['lower'])
                    or not np.array_equal(record['upper'],bounded['upper'])
                    or not 0 < bounded['radius'] <= record['radius']+1e-12):
                raise ValueError('Semantic certificate must only shrink the physical region')
            record=copy.deepcopy(bounded)
        record['physical_pool_id']=i
        admitted.append(record)
    if not admitted:
        raise RuntimeError('Empty admitted bounded cover')
    edges=overlap_edges(admitted)
    sources,targets=endpoint_ids(to_sets(admitted),start,goal)
    component=component_indices(len(admitted),edges,sources,targets)
    keep=terminal_path_indices(len(admitted),edges,sources,targets)
    remap={old:new for new,old in enumerate(keep)}
    return ([admitted[i] for i in keep],
            [(remap[a],remap[b]) for a,b in edges if a in remap and b in remap],
            dict(admitted_regions=len(admitted),connected_regions=len(keep),
                 component_regions=len(component),cyclic_deadend_regions_pruned=len(component)-len(keep),
                 removed_by_certificates=sum(v[0] is None for v in certificates.values()),
                 certified_regions=len(certificates)))


def sample_translation(trajectory, spacing_m=.01):
    if not np.isfinite(spacing_m) or spacing_m<=0:
        raise ValueError('Positive sampling spacing required')
    # solve_graph bounds EACH velocity component by .15 m/s.
    duration=float(trajectory.end_time())
    times=np.linspace(0,duration,max(2,int(np.ceil(duration*np.sqrt(2)*.15/spacing_m))+1))
    xy=np.array([trajectory.value(t).ravel() for t in times])
    if np.max(np.linalg.norm(np.diff(xy,axis=0),axis=1))>spacing_m+1e-6:
        raise RuntimeError('Trajectory exceeds declared sampling spacing')
    return times,xy


def initial_problem_digest(pool,start,goal,solver_settings,relaxation_solver):
    regions,edges,_=compile_graph(pool,{},start,goal)
    return pool_digest(dict(pool=pool,start=np.asarray(start).tolist(),goal=np.asarray(goal).tolist(),
        connected_pool_ids=[r['physical_pool_id'] for r in regions],edges=edges,
        per_axis_speed_m_s=.15,path_length_weight=1.,time_weight=.05,
        solver_settings=solver_settings or {},relaxation_solver=relaxation_solver))


def make_initial_proposal(pool,start,goal,event,*,solver_settings=None,relaxation_solver='clarabel',provenance=None):
    """Archive an actual returned solution. Never trust its previous audit."""
    proposal=dict(schema='shared_gcs_initial_v1',
        problem_sha256=initial_problem_digest(pool,start,goal,solver_settings,relaxation_solver),
        physical_pool_sha256=pool_digest(pool),
        graph=copy.deepcopy(event['graph']),
        selected_physical_pool_ids=copy.deepcopy(event['selected_physical_pool_ids']),
        path_xy=copy.deepcopy(event['path_xy']),time_s=copy.deepcopy(event['time_s']),
        provenance=provenance or {})
    proposal['proposal_sha256']=pool_digest(proposal)
    return proposal


def validate_initial_proposal(proposal,pool,start,goal,regions,edges,*,solver_settings=None,relaxation_solver='clarabel',spacing_m=.01):
    archived=copy.deepcopy(proposal);digest=archived.pop('proposal_sha256',None)
    if digest!=pool_digest(archived):raise ValueError('Shared initial proposal content hash mismatch')
    if proposal['problem_sha256']!=initial_problem_digest(pool,start,goal,solver_settings,relaxation_solver):
        raise ValueError('Shared initial proposal belongs to a different planning problem')
    graph=proposal['graph'];ids=graph['selected_region_path']
    if (not ids or any(type(i) is not int or not 0<=i<len(regions) for i in ids)
        or graph['nodes']!=len(regions) or graph['directed_edges']!=len(edges)
        or [regions[i]['physical_pool_id'] for i in ids]!=proposal['selected_physical_pool_ids']
        or any((a,b) not in edges for a,b in zip(ids,ids[1:]))):
        raise ValueError('Shared initial region path or graph mismatch')
    xy=np.asarray(proposal['path_xy'],float);times=np.asarray(proposal['time_s'],float)
    if (xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or times.shape!=(len(xy),)
        or not np.all(np.isfinite(xy)) or not np.all(np.isfinite(times))
        or times[0]!=0 or np.any(np.diff(times)<=0)):
        raise ValueError('Finite ordered shared trajectory required')
    if np.max(np.abs(xy[[0,-1]]-np.array([start[:2],goal[:2]])))>1e-6:
        raise ValueError('Shared initial trajectory endpoint mismatch')
    if (np.max(np.linalg.norm(np.diff(xy,axis=0),axis=1))>spacing_m+1e-6
        or np.max(np.abs(np.diff(xy,axis=0))/np.diff(times)[:,None])>.15+1e-6):
        raise ValueError('Shared initial trajectory spacing or speed mismatch')
    residual=np.min(np.array([np.maximum.reduce([
        np.linalg.norm(xy-regions[i]['center'],axis=1)-regions[i]['radius'],
        (np.asarray(regions[i]['lower'])-xy).max(axis=1),
        (xy-np.asarray(regions[i]['upper'])).max(axis=1)]) for i in ids]),axis=0)
    if residual.max()>1e-6:raise ValueError('Shared initial path leaves selected region union')
    return times,xy,float(residual.max())


def violation_region_ids(regions,selected_indices,xy,audit,known_ids=()):
    """Map every failed access sample to ALL selected convex sets containing it.

    This uses the original XY coordinates and exact disk/box membership, not
    a 2D drawing, closest center, or the first few displayed counterexamples.
    """
    rows=audit.get('sample_results')
    if rows is None or len(rows)!=len(xy):raise ValueError('Directed repair requires the full pointwise audit')
    bad=[];targets=set();conflicts=set();known=set(known_ids)
    for i,(q,row) in enumerate(zip(xy,rows)):
        if row['sample']!=i or not np.allclose(row['xy'],q,rtol=0,atol=1e-12):
            raise ValueError('Audit sample indices or coordinates do not match the planned path')
        if row['access_satisfied']:continue
        bad.append(i);containing=[]
        for j in selected_indices:
            r=regions[j];residual=max(np.linalg.norm(q-r['center'])-r['radius'],max(np.asarray(r['lower'])-q),max(q-np.asarray(r['upper'])))
            if residual<=1e-6:containing.append(r['physical_pool_id'])
        if not containing:raise ValueError('Failed access sample is outside the selected region union')
        targets.update(containing);conflicts.update(set(containing)&known)
    ranges=[]
    for i in bad:
        if ranges and i==ranges[-1][1]+1:ranges[-1][1]=i
        else:ranges.append([i,i])
    return dict(bad_sample_indices=bad,bad_sample_intervals=ranges,
                implicated_physical_pool_ids=sorted(targets),
                new_physical_pool_ids=sorted(targets-known),
                previously_certified_conflicts=sorted(conflicts))


def replan(pool,start,goal,certify,audit,*,policy,max_solves=3,spacing_m=.01,checkpoint=None,solver_settings=None,relaxation_solver='clarabel',initial_proposal=None):
    """Plan geometry-only, early, or lazy with one frozen pool and budget.

    certify(i, physical_record) returns (bounded_record_or_None, proof).
    audit(xy) returns a full pointwise audit with `satisfied` and violations.
    A geometry-only violation is retained as a result, never executed.
    Lazy repairs every newly selected region after a failed path audit.
    """
    if policy not in {'geometry','early','lazy','directed'} or type(max_solves) is not int or max_solves<1:
        raise ValueError('Explicit policy and positive solve budget required')
    if initial_proposal is not None and policy=='early':raise ValueError('Early semantics cannot reuse an unconstrained trajectory as its solution')
    began=time.perf_counter();digest=pool_digest(pool);certificates={};history=[]
    cert_seconds=0.;graph_seconds=0.;audit_seconds=0.;solve_seconds=0.;inherited_seconds=0.
    status='budget_exhausted';last_xy=[];last_graph=None;last_audit=None
    def emit(event):
        history.append(event)
        if checkpoint:checkpoint(event)
    def certify_ids(ids):
        nonlocal cert_seconds
        tick=time.perf_counter();new=[]
        for i in sorted(set(ids)-certificates.keys()):
            certificates[i]=certify(i,copy.deepcopy(pool[i]));new.append(i)
        cert_seconds+=time.perf_counter()-tick
        return new
    if policy=='early':
        certify_ids(range(len(pool)))
    for iteration in range(max_solves):
        tick=time.perf_counter()
        try:regions,edges,counts=compile_graph(pool,certificates,start,goal)
        except RuntimeError as exc:
            graph_seconds+=time.perf_counter()-tick;status='bounded_cover_disconnected'
            emit(dict(iteration=iteration,status=status,reason=str(exc)));break
        graph_seconds+=time.perf_counter()-tick
        reused=initial_proposal is not None and iteration==0
        emit(dict(iteration=iteration,status='loading_shared_initial' if reused else 'solving',**counts))
        if reused:
            times,xy,_=validate_initial_proposal(initial_proposal,pool,start,goal,regions,edges,
                solver_settings=solver_settings,relaxation_solver=relaxation_solver,spacing_m=spacing_m)
            graph=copy.deepcopy(initial_proposal['graph']);inherited_seconds+=graph['solve_seconds']
        else:
            try:
                trajectory,graph=solve_graph(regions,edges,start,goal,solver_settings=solver_settings,relaxation_solver=relaxation_solver)
            except GcsSolveFailure as exc:
                solve_seconds+=exc.details['solve_seconds'];status='solver_failed'
                emit(dict(iteration=iteration,status=status,diagnostics=exc.details,**counts));break
            times,xy=sample_translation(trajectory,spacing_m)
        solve_seconds+=graph['solve_seconds'];last_graph=graph
        last_xy=xy.tolist()
        if np.max(np.abs(xy[[0,-1]]-np.array([start[:2],goal[:2]])))>1e-6:
            raise RuntimeError('Replanning endpoint mismatch')
        tick=time.perf_counter();checked=audit(xy);audit_seconds+=time.perf_counter()-tick;last_audit=checked
        selected=[regions[i]['physical_pool_id'] for i in graph['selected_region_path']]
        event=dict(iteration=iteration,status='accepted' if checked['satisfied'] else 'rejected',
                   graph=graph,selected_physical_pool_ids=selected,**counts,
                   reused_shared_initial=reused,
                   audit=checked,path_xy=xy.tolist(),time_s=times.tolist(),
                   path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()))
        emit(event)
        if checked['satisfied']:
            status='accepted';break
        if policy=='geometry':
            status='geometric_plan_violates_access';break
        if policy=='early':
            status='certified_path_failed_independent_audit';break
        if iteration+1==max_solves:
            break
        if policy=='directed':
            if checked.get('geometry_violations',0):
                status='geometry_audit_failed';break
            repair=violation_region_ids(regions,graph['selected_region_path'],xy,checked,certificates)
            event['directed_repair']=repair
            if repair['previously_certified_conflicts']:
                status='certified_path_failed_independent_audit';break
            new=certify_ids(repair['new_physical_pool_ids'])
        else:new=certify_ids(selected)
        event['newly_certified_physical_pool_ids']=new
        if not new:
            status='no_new_region_repair';break
    if digest!=pool_digest(pool):
        raise RuntimeError('Physical pool mutated during replanning')
    return dict(policy=policy,status=status,satisfied=status=='accepted',physical_pool_sha256=digest,
                physical_pool_size=len(pool),max_solves=max_solves,
                solver_calls=sum(e['status'] in {'accepted','rejected','solver_failed'} for e in history),
                actual_new_solver_calls=sum(e['status'] in {'accepted','rejected','solver_failed'} and not e.get('reused_shared_initial',False) for e in history),
                shared_initial_sha256=initial_proposal['proposal_sha256'] if initial_proposal else None,
                inherited_initial_solve_seconds=inherited_seconds,new_gcs_solve_seconds=solve_seconds-inherited_seconds,
                solver_settings=solver_settings or {},
                certificate_seconds=cert_seconds,graph_seconds=graph_seconds,
                gcs_solve_seconds=solve_seconds,audit_seconds=audit_seconds,
                total_seconds=time.perf_counter()-began,certificates={str(i):v[1] for i,v in certificates.items()},
                iterations=history,path_xy=last_xy,final_graph=last_graph,final_audit=last_audit,
                scope='Fixed-arm/yaw carrying only; same immutable physical cover and pose. Finite sampled audits. Lazy selected-region repair, bounded solves. No whole-task/global infeasibility or statistical speed claim.')
