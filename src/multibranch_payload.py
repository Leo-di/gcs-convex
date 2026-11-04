"""Bounded spatial cover, route-witness region certificates, official multi-path GCS.

No source-target region chain is selected before optimization. Spatial packing
restricts the finite cover, so no full-cover/global optimality claim is made.
"""
import copy,json,time
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from pydrake.geometry.optimization import Hyperellipsoid,HPolyhedron,Intersection,Point,GraphOfConvexSetsOptions
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver,ScsSolver,SolverOptions
from .geometry_outbound_gcs import make_cover,prune_contained_disks
from .human_passage import HumanPassageEvaluator

def pack_regions(records,factor,required_ids=None):
    """Endpoint-independent radius packing; retain the explicit endpoint seeds."""
    keep=[];centers=[];radii=[];mandatory=set(range(min(2,len(records)))) if required_ids is None else set(required_ids)
    for i in sorted(mandatory)+sorted((i for i in range(len(records)) if i not in mandatory),key=lambda i:(-records[i]['radius'],i)):
        c=np.asarray(records[i]['center'])
        if i not in mandatory and keep and np.any(np.linalg.norm(np.asarray(centers)-c,axis=1)<=factor*np.asarray(radii)):continue
        keep.append(i);centers.append(c);radii.append(records[i]['radius'])
    return keep

def component_indices(n,edges,sources,targets):
    if not sources or not targets:raise RuntimeError('Missing endpoint coverage')
    ij=np.array(edges,dtype=int).reshape(-1,2)
    graph=coo_matrix((np.ones(len(ij)),(ij[:,0],ij[:,1])),shape=(n,n))
    _,labels=connected_components(graph,directed=False)
    common=set(labels[sources])&set(labels[targets])
    if not common:raise RuntimeError('No connection in bounded spatial cover')
    return np.flatnonzero(np.isin(labels,list(common))).tolist()


def terminal_path_indices(n,edges,sources,targets):
    """Keep every vertex on ANY simple terminal path, without choosing a path.

    Exact block-cut-tree pruning on the undirected overlap graph, augmented
    with virtual start/goal vertices. Cyclic cul-de-sacs can survive connected
    component or degree-one pruning, but never occur on a simple s-t path.
    The positive path-length/time objective needs no repeated region cycles.
    """
    component_indices(n,edges,sources,targets)
    source,target=n,n+1;adj=[set() for _ in range(n+2)]
    for a,b in edges:adj[a].add(b);adj[b].add(a)
    for i in sources:adj[source].add(i);adj[i].add(source)
    for i in targets:adj[target].add(i);adj[i].add(target)
    discovery=[-1]*(n+2);low=[0]*(n+2);stack=[];blocks=[];clock=0
    def visit(v,parent):
        nonlocal clock
        discovery[v]=low[v]=clock;clock+=1
        for w in sorted(adj[v]):
            if discovery[w]<0:
                stack.append((v,w));visit(w,v);low[v]=min(low[v],low[w])
                if low[w]>=discovery[v]:
                    block=set()
                    while stack:
                        edge=stack.pop();block.update(edge)
                        if edge==(v,w):break
                    blocks.append(block)
            elif w!=parent and discovery[w]<discovery[v]:
                stack.append((v,w));low[v]=min(low[v],discovery[w])
    visit(source,-1)
    tree=[set() for _ in range(n+2+len(blocks))]
    for i,block in enumerate(blocks):
        node=n+2+i
        for v in block:tree[node].add(v);tree[v].add(node)
    parent={source:None};queue=[source]
    for v in queue:
        if v==target:break
        for w in sorted(tree[v]):
            if w not in parent:parent[w]=v;queue.append(w)
    if target not in parent:raise RuntimeError('No terminal path in block-cut tree')
    keep=set();v=target
    while v is not None:
        if v>=n+2:keep.update(blocks[v-n-2])
        elif v<n:keep.add(v)
        v=parent[v]
    return sorted(i for i in keep if i<n)

def to_sets(records):
    sets=[]
    for r in records:
        center=np.asarray(r['center']);radius=float(r['radius'])
        lo=np.asarray(r['lower']);hi=np.asarray(r['upper'])
        disk=Hyperellipsoid.MakeHypersphere(radius,center)
        # Exact containment, with NO positive tolerance: dropping a genuinely
        # clipping face would enlarge the feasible set. Most interior disks
        # do not need four extra perspective inequalities per GCS incident edge.
        if np.all(center-radius>=lo) and np.all(center+radius<=hi):
            sets.append(disk)
        else:
            sets.append(Intersection(disk,HPolyhedron.MakeBox(lo,hi)))
    return sets

def endpoint_ids(regions,start,goal):
    return ([i for i,r in enumerate(regions) if r.PointInSet(start[:2],1e-9)],
            [i for i,r in enumerate(regions) if r.PointInSet(goal[:2],1e-9)])

def certify_region(record,human,margin,min_radius):
    """Both grid witnesses persist throughout a fixed-arm/yaw XY disk.

Moving robot/payload AABBs translate exactly by delta_xy. Capsule-to-AABB
distance is 1-Lipschitz in that translation; all other bodies stay fixed.
The existing whole-cell guard is already included in the witness slack.
"""
    r=human.current();children=r.get('connections',{human.spec['id']:r});slack=float(r.get('motion_slack_m',-np.inf))
    radius=min(float(record['radius']),slack-margin)
    certificate=dict(satisfied=bool(r['satisfied'] and radius>=min_radius),geometry_radius_m=record['radius'],witness_slack_m=slack,
        allowed_radius_m=max(0.,radius),required_margin_m=margin,lipschitz_factor=1.,
        connections={k:dict(satisfied=v['satisfied'],slack_m=v.get('motion_slack_m'),path_xy=v.get('path_xy',[])) for k,v in children.items()},
        scope='Fixed arm/yaw and grasp; robot and held payload translate; other objects fixed; same spatial capsule/AABB and whole-cell guard model; numerical distances')
    return dict(record,radius=radius,semantic_certificate=certificate) if certificate['satisfied'] else None,certificate

class GcsSolveFailure(RuntimeError):
    """Numerical solver failure with diagnostics, never an infeasibility proof."""
    def __init__(self,details):
        super().__init__('Multi-branch GCS failed; no single-chain fallback')
        self.details=details


def solve_graph(records,edges,start,goal,*,solver_settings=None,relaxation_solver='clarabel'):
    regions=to_sets(records);sources,targets=endpoint_ids(regions,start,goal);planner=GcsTrajectoryOptimization(2)
    free=planner.AddRegions(regions,edges,order=1,h_min=.01,h_max=20,name='branch')
    src=planner.AddRegions([Point(np.asarray(start[:2]))],order=0,name='start');dst=planner.AddRegions([Point(np.asarray(goal[:2]))],order=0,name='goal')
    planner.AddEdges(src,free,edges_between_regions=[(0,i) for i in sources]);planner.AddEdges(free,dst,edges_between_regions=[(i,0) for i in targets])
    planner.AddPathLengthCost(1.);planner.AddTimeCost(.05);planner.AddVelocityBounds(np.full(2,-.15),np.full(2,.15))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10;options.rounding_seed=0;options.solver=options.restriction_solver=ClarabelSolver()
    if solver_settings:
        settings=SolverOptions()
        for key,value in solver_settings.items():settings.SetOption(ClarabelSolver.id(),key,value)
        options.solver_options=settings;options.restriction_solver_options=settings
    if relaxation_solver=='scs':
        # Keep the SAME full GCS graph and costs. SCS solves its relaxation;
        # Clarabel still solves the rounded restrictions at the stated limits.
        options.solver=ScsSolver();options.preprocessing_solver=ClarabelSolver();scs=SolverOptions()
        for key,value in dict(eps_abs=1e-5,eps_rel=1e-5,max_iters=20000,time_limit_secs=45.).items():scs.SetOption(ScsSolver.id(),key,value)
        options.solver_options=scs
    elif relaxation_solver!='clarabel':raise ValueError('Unknown GCS relaxation solver')
    tick=time.perf_counter();trajectory,result=planner.SolvePath(src,dst,options);seconds=time.perf_counter()-tick
    solver_details=result.get_solver_details()
    diagnostics=dict(solver=result.get_solver_id().name(),solution_result=str(result.get_solution_result()),
        status=str(getattr(solver_details,'status','unknown')),iterations=int(getattr(solver_details,'iterations',0)),
        solver_seconds=float(getattr(solver_details,'solve_time',seconds)),settings=solver_settings or {},
        relaxation_solver=relaxation_solver,restriction_solver='clarabel')
    if not result.is_success():raise GcsSolveFailure(dict(nodes=len(records),directed_edges=len(edges),solve_seconds=seconds,**diagnostics))
    selected=planner.graph_of_convex_sets().GetSolutionPath(src.Vertices()[0],dst.Vertices()[0],result)
    vertex_ids={v.id():i for i,v in enumerate(free.Vertices())};path=[vertex_ids[e.v().id()] for e in selected if e.v().id() in vertex_ids]
    graph=dict(nodes=len(records),directed_edges=len(edges),endpoint_edges=len(sources)+len(targets),selected_region_path=path,
        selected_after_solve=True,objective=float(result.get_optimal_cost()),solve_seconds=seconds,
        solver_diagnostics=diagnostics,
        branch_vertices=sum(sum(a==i for a,b in edges)>2 for i in range(len(records))),
        undirected_cycle_rank=len(set(tuple(sorted(e)) for e in edges))-len(records)+1)
    graph['alternative_source_target_path']=False
    for a,b in zip(path,path[1:]):
        without=[(i,j) for i,j in edges if {i,j}!={a,b}]
        try:component_indices(len(records),without,sources,targets);graph['alternative_source_target_path']=True;break
        except RuntimeError:pass
    return trajectory,graph

def plan(scene,start,goal,arm,gripper,X_GC,spec,artifact,*,action_id):
    from .payload_translation_bridge import PayloadTranslationOracle,PayloadBridgeRejected,overlap_edges
    began=time.perf_counter();artifact=Path(artifact);start=np.asarray(start,float);goal=np.asarray(goal,float)
    if start.shape!=(3,) or goal.shape!=(3,) or abs(start[2]-goal[2])>1e-10:raise ValueError('Fixed-yaw finite base triples required')
    enforce=bool(scene.config.get('human_passage',{}).get('enforce'));original=scene.config.get('human_passage',{}).get('enforce')
    # Same geometry cover for early and posthoc. Only the passage predicate is
    # deferred here; all other existing geometric/spatial predicates remain.
    try:
        if original is not None:scene.config['human_passage']['enforce']=False
        oracle=PayloadTranslationOracle(scene,arm,gripper,start[2],X_GC,action_id=action_id,max_distance=float(spec['max_radius_m'])+float(spec['margin_m']))
    finally:
        if original is not None:scene.config['human_passage']['enforce']=original
    lo=np.asarray(spec['lower_xy']);hi=np.asarray(spec['upper_xy']);margin=float(spec['margin_m']);minimum=spec.get('min_radius_m',.025)
    if np.any(start[:2]<lo) or np.any(start[:2]>hi) or np.any(goal[:2]<lo) or np.any(goal[:2]>hi):raise ValueError('Endpoint outside translation domain')
    for q in [start,goal]:
        d,internal,_=oracle.query(q[:2])
        if d<margin-1e-7 or internal < -1e-7:raise RuntimeError('Geometric endpoint invalid')
    regions,records=make_cover(oracle.query,[dict(name='held_translation_envelope',lower=lo,upper=hi)],start[:2],goal[:2],lo,hi,
        spacing=spec['spacing_m'],margin=margin,max_radius=spec['max_radius_m'],min_radius=minimum,
        refinement_spacing=spec.get('refinement_spacing_m'),refinement_radius_threshold=spec.get('refinement_radius_threshold_m'))
    regions,records=prune_contained_disks(regions,records);geometry_records=records;events=[];human=HumanPassageEvaluator(scene,payload_X_GC=X_GC,force=True) if enforce else None
    endpoint_sources,endpoint_targets=endpoint_ids(regions,start,goal);mandatory=endpoint_sources+endpoint_targets
    selected_records=None;selected_edges=[];physical=[];certificates=[]
    for factor in [1.3,1.2,1.0]:
        ids=pack_regions(geometry_records,factor,mandatory);physical=[dict(geometry_records[i],geometry_region_id=i) for i in ids];physical_edges=overlap_edges(physical)
        ps,pt=endpoint_ids(to_sets(physical),start,goal);event=dict(packing_factor=factor,physical_nodes=len(physical),physical_edges=len(physical_edges));events.append(event)
        try:component_indices(len(physical),physical_edges,ps,pt)
        except RuntimeError:event['status']='geometry_disconnected';continue
        # Physical connectivity determines common packing before rules apply.
        kept=[];certificates=[]
        for r in physical:
            if enforce:
                oracle.query(r['center']);bounded,cert=certify_region(r,human,margin,minimum);certificates.append(dict(geometry_region_id=r['geometry_region_id'],center=r['center'],**cert))
                if bounded is not None:kept.append(bounded)
            else:kept.append(r)
        edges=overlap_edges(kept);sources,targets=endpoint_ids(to_sets(kept),start,goal)
        event.update(semantic_nodes=len(kept),semantic_edges=len(edges),removed_nodes=len(physical)-len(kept))
        try:connected=component_indices(len(kept),edges,sources,targets)
        except RuntimeError:event['status']='semantic_disconnected';continue
        remap={old:new for new,old in enumerate(connected)};selected_records=[kept[i] for i in connected];selected_edges=[(remap[i],remap[j]) for i,j in edges if i in remap and j in remap];event['status']='connected'
        break
    evidence=dict(start=start.tolist(),goal=goal.tolist(),geometry_regions=geometry_records,packing_events=events,enforce_passage=enforce,
        physical_candidates=physical,certificates=certificates,regions=selected_records or [],edges=selected_edges if selected_records else [],
        graph_policy='Radius-based spatial packing, then endpoint-connected components; no selected path or shortest-path chain before GCS')
    artifact.with_name('multibranch_cover.json').write_text(json.dumps(evidence,indent=2))
    if selected_records is None:raise PayloadBridgeRejected('multibranch_cover_disconnected',dict(events=events,scope='Finite bounded cover only, not physical infeasibility'))
    if len(selected_records)>900 or len(selected_edges)>6500:raise PayloadBridgeRejected('multibranch_graph_budget',dict(nodes=len(selected_records),edges=len(selected_edges)))
    trajectory,graph=solve_graph(selected_records,selected_edges,start,goal);duration=float(trajectory.end_time());tt=np.linspace(0,duration,max(101,int(duration*20)+1));xy=np.array([trajectory.value(t).ravel() for t in tt])
    if np.max(np.abs(xy[[0,-1]]-np.array([start[:2],goal[:2]])))>1e-6:raise RuntimeError('Multi-branch GCS endpoint mismatch')
    xy[0],xy[-1]=start[:2],goal[:2];checks=[]
    audit_human=HumanPassageEvaluator(scene,payload_X_GC=X_GC,force=True)
    for a,b in zip(xy,xy[1:]):
        for u in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(b-a)/.0025))+1)):
            q=(1-u)*a+u*b;d,internal,_=oracle.query(q);h=audit_human.current() if enforce else None
            if d<margin-1e-7 or internal < -1e-7 or (enforce and not h['satisfied']):raise RuntimeError('Multi-branch independent trajectory audit failed')
            checks.append(d)
    path=graph['selected_region_path'];report=dict(satisfied=True,method='official_Drake_GCS_multibranch_fixed_arm_yaw_payload_translation',graph=graph,
        cover_region_count=len(geometry_records),selected_region_chain=path,regions=[selected_records[i] for i in path],cover_attempts=events,
        semantic_regions_enforced=enforce,minimum_clearance_m=min(checks),margin_m=margin,audit_samples=len(checks),oracle_calls=oracle.calls,
        base_path_length_m=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),total_seconds=time.perf_counter()-began,
        scope='GCS selects path AFTER solving the bounded multi-branch graph. Spatial packing restricts coverage; fixed arm/yaw only; no global task optimality claim. Per-region passage witnesses apply only to carrying translation.')
    artifact.write_text(json.dumps(report,indent=2));return np.c_[xy,np.full(len(xy),start[2])],report
