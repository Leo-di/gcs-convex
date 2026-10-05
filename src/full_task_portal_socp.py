"""Same-cover solver ablation: overlap roadmap + Dijkstra + fixed-route SOCP.

This project implementation is a mechanism control, not a reproduction of PRM.
It uses no GraphOfConvexSets solver. All allowed action/site branches are compared.
"""
import heapq
import time
import numpy as np
from pydrake.geometry.optimization import HPolyhedron,Hyperellipsoid
from pydrake.solvers import MathematicalProgram,ClarabelSolver
from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES,WEIGHTS,primitive_cost


def project(regions,target,weights):
    p=MathematicalProgram();x=p.NewContinuousVariables(len(target))
    for r in regions:r.AddPointInSetConstraints(p,x)
    p.AddQuadraticErrorCost(np.diag(weights**2),target,x)
    sol=ClarabelSolver().Solve(p)
    if not sol.is_success():return None
    point=sol.GetSolution(x)
    return point if all(r.PointInSet(point,1e-6) for r in regions) else None


def make_roadmap(regions,centers,weights,density=5,pair_allowed=None):
    began=time.perf_counter();points=[];members=[[] for _ in regions];rng=np.random.default_rng(91023)
    for i,a in enumerate(regions):
        for j in range(i):
            b=regions[j]
            if pair_allowed is not None and not pair_allowed(i,j):continue
            if not a.IntersectsWith(b):continue
            mid=(centers[i]+centers[j])/2
            targets=[mid]+[mid+rng.normal(size=len(mid))*.3/weights for _ in range(density-1)]
            accepted=[]
            for t in targets:
                point=project([a,b],t,weights)
                if point is None or any(np.linalg.norm(weights*(point-v))<1e-7 for v in accepted):continue
                accepted.append(point);index=len(points);points.append(point.tolist());members[i].append(index);members[j].append(index)
    return {'points':points,'members':members,'build_seconds':time.perf_counter()-began,'density':density}


def route(regions,roadmap,start,goal,weights):
    points=[np.array(x) for x in roadmap['points']]+[np.array(start),np.array(goal)];source=len(points)-2;target=source+1
    members=[list(m) for m in roadmap['members']]
    for i,r in enumerate(regions):
        if r.PointInSet(start,1e-7):members[i].append(source)
        if r.PointInSet(goal,1e-7):members[i].append(target)
    adjacency=[{} for _ in points]
    for region,m in enumerate(members):
        for ii,u in enumerate(m):
            for v in m[ii+1:]:
                cost=float(np.linalg.norm(weights*(points[u]-points[v])))
                if v not in adjacency[u]:adjacency[u][v]=(cost,region);adjacency[v][u]=(cost,region)
    distance={source:0.};previous={};queue=[(0.,source)]
    while queue:
        cost,u=heapq.heappop(queue)
        if cost>distance[u]+1e-12:continue
        if u==target:break
        for v,(length,region) in adjacency[u].items():
            new=cost+length
            if new<distance.get(v,float('inf'))-1e-12:
                distance[v]=new;previous[v]=(u,region);heapq.heappush(queue,(new,v))
    if target not in previous:return None
    sequence=[];v=target
    while v!=source:
        u,region=previous[v];sequence.append(region);v=u
    sequence=list(reversed(sequence));sequence=[r for i,r in enumerate(sequence) if i==0 or sequence[i-1]!=r]
    p=MathematicalProgram();n=len(weights);x=p.NewContinuousVariables(n*(len(sequence)+1)).reshape(-1,n)
    p.AddBoundingBoxConstraint(start,start,x[0]);p.AddBoundingBoxConstraint(goal,goal,x[-1])
    for k,region in enumerate(sequence):
        regions[region].AddPointInSetConstraints(p,x[k]);regions[region].AddPointInSetConstraints(p,x[k+1])
        p.AddL2NormCost(np.c_[-np.diag(weights),np.diag(weights)],np.zeros(n),np.r_[x[k],x[k+1]])
    sol=ClarabelSolver().Solve(p)
    if not sol.is_success():return None
    values=sol.GetSolution(x)
    if not all(regions[r].PointInSet(v,1e-5) for i,r in enumerate(sequence) for v in values[i:i+2]):return None
    return {'regions':sequence,'points':values.tolist(),'cost':float(sol.get_optimal_cost()),'discrete_cost':distance[target]}


def prepare(candidates,disks,density=5):
    regions=[Hyperellipsoid.MakeHypersphere(d['radius'],np.array(d['center'])) for d in disks]
    adjacent=lambda i,j:np.linalg.norm(np.array(disks[i]['center'])-disks[j]['center'])<disks[i]['radius']+disks[j]['radius']-.003
    data={'empty':make_roadmap(regions,[np.array(d['center']) for d in disks],np.ones(2),density,adjacent),'carry':{}}
    for name,c in candidates.items():
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in c['cover']['regions']]
        data['carry'][name]=make_roadmap(regions,[r.ChebyshevCenter() for r in regions],WEIGHTS,density)
    return data


def solve(disks,candidates,allowed,config,roadmaps,start,*,navigation_cache=None):
    began=time.perf_counter();initial=q(next(iter(candidates.values()))['rows'][0]);initial[:3]=start
    empty=[Hyperellipsoid.MakeHypersphere(d['radius'],np.array(d['center'])) for d in disks]
    sites=config.get('inspection_sites',[None]);best=None;attempts=[];nav_cache={} if navigation_cache is None else navigation_cache
    def nav(a,b):
        key=tuple(a)+tuple(b)
        if key not in nav_cache:nav_cache[key]=route(empty,roadmaps['empty'],a,b,np.ones(2))
        return nav_cache[key]
    for name,c in candidates.items():
        if name not in allowed or not all(np.isfinite(v) and v>=-1e-8 for d in c['guards'].values() for v in d.values()):continue
        pick=[r for r in c['rows'] if r['phase'] in PICK_PHASES];place=[r for r in c['rows'] if r['phase'] in PLACE_PHASES]
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in c['cover']['regions']]
        carry=route(regions,roadmaps['carry'][name],q(pick[-1]),q(place[0]),WEIGHTS)
        if carry is None:attempts.append({'candidate':name,'failure':'carry_roadmap_or_socp_failed'});continue
        for site in sites:
            goals=[initial[:2]]+([np.array(site['xy'])] if site else [])+[q(pick[0])[:2]]
            navigation=[nav(a,b) for a,b in zip(goals,goals[1:])]
            if any(x is None for x in navigation):attempts.append({'candidate':name,'site':site,'failure':'navigation_roadmap_or_socp_failed'});continue
            value=sum(x['cost'] for x in navigation)+carry['cost']+primitive_cost(pick)+primitive_cost(place)
            attempts.append({'candidate':name,'site':site,'cost':value})
            if best is not None and value>=best['objective']:continue
            path=[{'kind':'start','x':np.r_[initial,initial].tolist()}]
            for nav_result in navigation:
                for k,r in enumerate(nav_result['regions']):
                    a,b=nav_result['points'][k:k+2]
                    path.append({'kind':'empty','disk':r,'x':np.r_[a,initial[2:],b,initial[2:]].tolist()})
            path.append({'kind':'pick','candidate':name,'x':np.r_[q(pick[0]),q(pick[-1])].tolist()})
            for k,r in enumerate(carry['regions']):path.append({'kind':'carry','candidate':name,'region':r,'x':np.r_[carry['points'][k],carry['points'][k+1]].tolist()})
            path.extend([{'kind':'place','candidate':name,'x':np.r_[q(place[0]),q(place[-1])].tolist()},{'kind':'accept','x':[0.]*20}])
            best={'status':'solved','objective':value,'path':path,'selected_candidate':name,'inspection_site':site['name'] if site else None}
    result=best or {'status':'no_roadmap_solution'}
    result.update(attempts=attempts,build_solve_seconds=time.perf_counter()-began,scope='Project same-cover mechanism control; Dijkstra plus SOCP, no GCS call; all action/site branches compared')
    return result,result.get('path')
