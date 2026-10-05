"""Single semantic portal graph over every allowed grasp and placement.

Uses the same projected overlap samples as full_task_portal_socp. One Dijkstra
search chooses a discrete task path; ordinary SOCP refines its region runs.
No GraphOfConvexSets solver is invoked. Not a globally optimal continuous solver.
"""
import heapq
import time
import numpy as np
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.solvers import MathematicalProgram, ClarabelSolver
from .semantic_candidate_gcs import q, WEIGHTS, PICK_PHASES, PLACE_PHASES, primitive_cost


class PortalGraph:
    def __init__(self):
        self.points=[];self.adj=[]

    def node(self,point=None):
        i=len(self.points);self.points.append(None if point is None else np.asarray(point));self.adj.append({});return i

    def edge(self,u,v,cost,meta):
        old=self.adj[u].get(v)
        if old is None or cost<old[0]:self.adj[u][v]=(float(cost),meta)

    def layer(self,regions,roadmap,weights,meta,endpoints):
        ids=[self.node(x) for x in roadmap['points']];members=[[ids[i] for i in m] for m in roadmap['members']]
        for v in endpoints:
            for i,r in enumerate(regions):
                if r.PointInSet(self.points[v],1e-7):members[i].append(v)
        for i,m in enumerate(members):
            tag={**meta,'region':i}
            for j,u in enumerate(m):
                for v in m[j+1:]:
                    cost=np.linalg.norm(weights*(self.points[u]-self.points[v]))
                    self.edge(u,v,cost,tag);self.edge(v,u,cost,tag)

    def shortest(self,source,target):
        distances={source:0.};prev={};queue=[(0.,source)]
        while queue:
            d,u=heapq.heappop(queue)
            if d>distances[u]+1e-12:continue
            if u==target:break
            for v,(cost,meta) in self.adj[u].items():
                new=d+cost
                if new<distances.get(v,np.inf)-1e-12:
                    distances[v]=new;prev[v]=(u,meta);heapq.heappush(queue,(new,v))
        if target not in distances:return None
        edges=[];v=target
        while v!=source:
            u,meta=prev[v];edges.append((u,v,meta));v=u
        return distances[target],list(reversed(edges))


def refine(regions,sequence,start,goal,weights):
    sequence=[r for i,r in enumerate(sequence) if i==0 or r!=sequence[i-1]]
    p=MathematicalProgram();n=len(weights);x=p.NewContinuousVariables((len(sequence)+1)*n).reshape(-1,n)
    p.AddBoundingBoxConstraint(start,start,x[0]);p.AddBoundingBoxConstraint(goal,goal,x[-1])
    for i,r in enumerate(sequence):
        regions[r].AddPointInSetConstraints(p,x[i]);regions[r].AddPointInSetConstraints(p,x[i+1])
        p.AddL2NormCost(np.c_[-np.diag(weights),np.diag(weights)],np.zeros(n),np.r_[x[i],x[i+1]])
    result=ClarabelSolver().Solve(p)
    if not result.is_success():return None
    values=result.GetSolution(x)
    if not all(regions[r].PointInSet(v,1e-5) for i,r in enumerate(sequence) for v in values[i:i+2]):return None
    return sequence,values,float(result.get_optimal_cost())


def solve(disks,candidates,allowed,config,roadmaps,start,goal_options):
    began=time.perf_counter();g=PortalGraph();initial=q(next(iter(candidates.values()))['rows'][0]);initial[:3]=start
    source=g.node(initial[:2]);target=g.node();pre=[source];post=[];sites=config.get('inspection_sites');region_sets={}
    empty=[Hyperellipsoid.MakeHypersphere(d['radius'],np.array(d['center'])) for d in disks]
    region_sets['before']=empty;region_sets['after']=empty
    if sites is not None:
        for site in sites:
            a=g.node(site['xy']);b=g.node(site['xy']);pre.append(a);post.append(b)
            state=initial.copy();state[:2]=site['xy']
            g.edge(a,b,0.,{'kind':'inspect','site':site['name'],'x':np.r_[state,state].tolist()})
    else:post=pre
    pairs=0
    valid=lambda c:all(np.isfinite(v) and v>=-1e-8 for d in c['guards'].values() for v in d.values())
    for name,c in candidates.items():
        if name not in allowed or not valid(c):continue
        goals={k:v for k,v in goal_options.get(name,{}).items() if valid(v)}
        if not goals:continue
        pick=[r for r in c['rows'] if r['phase'] in PICK_PHASES]
        assert np.allclose(q(pick[0])[2:],initial[2:],atol=1e-7)
        a=g.node(q(pick[0])[:2]);b=g.node(q(pick[-1]));post.append(a);endpoints=[b]
        g.edge(a,b,primitive_cost(pick),{'kind':'pick','candidate':name,'x':np.r_[q(pick[0]),q(pick[-1])].tolist()})
        for key,goal in goals.items():
            place=[r for r in goal['rows'] if r['phase'] in PLACE_PHASES];v=g.node(q(place[0]));endpoints.append(v);pairs+=1
            g.edge(v,target,primitive_cost(place),{'kind':'place','candidate':name,'goal':key,'x':np.r_[q(place[0]),q(place[-1])].tolist()})
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in c['cover']['regions']];region_sets[name]=regions
        g.layer(regions,roadmaps['carry'][name],WEIGHTS,{'kind':'carry','candidate':name,'layer':name},endpoints)
    g.layer(empty,roadmaps['empty'],np.ones(2),{'kind':'empty','layer':'before'},pre)
    if sites is not None:g.layer(empty,roadmaps['empty'],np.ones(2),{'kind':'empty','layer':'after'},post)
    compiled=time.perf_counter();selected=g.shortest(source,target)
    info={'nodes':len(g.points),'edges':sum(map(len,g.adj)),'available_pairs':pairs,'graph_seconds':compiled-began,'search_seconds':time.perf_counter()-compiled,'global_searches':1,'socp_calls':0,'scope':'One global semantic Dijkstra path, then fixed-region-sequence SOCP; no GCS. Finite portal discretization is not continuous global optimality.'}
    if selected is None:return {**info,'status':'unreachable_in_portal_graph'},None
    discrete,edges=selected;path=[{'kind':'start','x':np.r_[initial,initial].tolist()}];i=0;cost=0.;site=None
    while i<len(edges):
        u,v,meta=edges[i]
        if meta['kind'] not in ['empty','carry']:
            path.append(meta);cost+=g.adj[u][v][0]
            if meta['kind']=='inspect':site=meta['site']
            i+=1;continue
        j=i+1
        while j<len(edges) and edges[j][2].get('layer')==meta['layer']:j+=1
        weights=np.ones(2) if meta['kind']=='empty' else WEIGHTS
        info['socp_calls']+=1
        result=refine(region_sets[meta['layer']],[e[2]['region'] for e in edges[i:j]],g.points[u],g.points[edges[j-1][1]],weights)
        if result is None:return {**info,'status':'fixed_route_socp_failed'},None
        sequence,values,value=result;cost+=value
        for k,r in enumerate(sequence):
            if meta['kind']=='empty':path.append({'kind':'empty','disk':r,'x':np.r_[values[k],initial[2:],values[k+1],initial[2:]].tolist()})
            else:path.append({'kind':'carry','candidate':meta['candidate'],'region':r,'x':np.r_[values[k],values[k+1]].tolist()})
        i=j
    actions=[v for v in path if v['kind'] in ['pick','place']]
    assert [v['kind'] for v in actions]==['pick','place'] and actions[0]['candidate']==actions[1]['candidate']
    path.append({'kind':'accept','x':[0.]*20})
    info.update(status='solved',objective=cost,discrete_cost=discrete,path=path,selected_candidate=actions[0]['candidate'],selected_goal=actions[1]['goal'],inspection_site=site,build_solve_seconds=time.perf_counter()-began)
    return info,path
