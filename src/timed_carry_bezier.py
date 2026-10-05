"""Cubic space/time Bezier GCS and matched fixed-route convex restrictions.

q(t)=r(h^{-1}(t)). C1 continuity is imposed on BOTH r and h; positive
h' and componentwise r' bounds imply continuous bounded physical velocity.
No acceleration/torque constraints or optimality certificate are claimed.
"""
import heapq,time
import numpy as np
from pydrake.geometry.optimization import CartesianProduct,HPolyhedron,GraphOfConvexSets,GraphOfConvexSetsOptions,Point
from pydrake.solvers import MathematicalProgram,ClarabelSolver
from .compare_carry_planners import LIMITS

DEGREE=3
MIN_CLOCK_INCREMENT=.001
DEADLINES=[10.,20.,40.,80.]


def local_constraints(x,n,limits):
    p=np.asarray(x[:4*n],dtype=object).reshape(4,n);h=x[4*n:];constraints=[]
    for k in range(3):
        dt=h[k+1]-h[k];constraints.append(dt>=MIN_CLOCK_INCREMENT)
        for j in range(n):
            constraints.extend([p[k+1,j]-p[k,j]<=limits[j]*dt,p[k+1,j]-p[k,j]>=-limits[j]*dt])
    return constraints


def continuity(a,b,n):
    p=a[:4*n].reshape(4,n);q=b[:4*n].reshape(4,n);h=a[4*n:];k=b[4*n:]
    return [*[p[3,j]==q[0,j] for j in range(n)],*[p[3,j]-p[2,j]==q[1,j]-q[0,j] for j in range(n)],h[3]==k[0],h[3]-h[2]==k[1]-k[0]]


def boundary(x,q,n,start):
    p=x[:4*n].reshape(4,n);i,j=(0,1) if start else (3,2)
    return [*[p[i,k]==q[k] for k in range(n)],*[p[i,k]==p[j,k] for k in range(n)]]


def curvature_energy(x,n):
    """Exact integral ||r''||² + |h''|² on a unit Bezier interval.

    Section 6 regularization of arXiv:2205.04422. This is parameter-space
    smoothness, NOT a bound on physical acceleration.
    """
    controls=np.asarray(x).reshape(-1)
    p=controls[:4*n].reshape(4,n);h=controls[4*n:]
    energy=0
    for c in [p,h.reshape(4,1)]:
        for j in range(c.shape[1]):
            a=6*(c[2,j]-2*c[1,j]+c[0,j])
            b=6*(c[3,j]-2*c[2,j]+c[1,j])
            energy+=((a+b)**2+a**2+b**2)/6
    return energy


def trajectory_objective(segments,n,smoothness=0.):
    return float(sum(s['x'][-1]-s['x'][-4]+smoothness*curvature_energy(s['x'],n) for s in segments))


def graph_solve(regions,start,goal,limits=LIMITS,horizon=80.,rounded=10,smoothness=0.):
    if smoothness<0:raise ValueError("smoothness must be nonnegative")
    n=len(start);g=GraphOfConvexSets();vertices=[];begin=time.perf_counter();clock=HPolyhedron.MakeBox(np.zeros(4),np.full(4,horizon))
    for i,region in enumerate(regions):
        v=g.AddVertex(CartesianProduct([region]*4+[clock]),str(i));vertices.append(v)
        for c in local_constraints(v.x(),n,limits):v.AddConstraint(c)
        v.AddCost(v.x()[-1]-v.x()[-4])
        if smoothness:v.AddCost(smoothness*curvature_energy(v.x(),n))
    src=g.AddVertex(Point(np.zeros(1)),'source');dst=g.AddVertex(Point(np.zeros(1)),'target')
    for i,region in enumerate(regions):
        if region.PointInSet(start,1e-7):
            e=g.AddEdge(src,vertices[i])
            for c in boundary(e.xv(),start,n,True):e.AddConstraint(c)
            e.AddConstraint(e.xv()[4*n]==0)
        if region.PointInSet(goal,1e-7):
            e=g.AddEdge(vertices[i],dst)
            for c in boundary(e.xu(),goal,n,False):e.AddConstraint(c)
        for j in range(i):
            if region.IntersectsWith(regions[j]):
                for u,v in [(i,j),(j,i)]:
                    e=g.AddEdge(vertices[u],vertices[v])
                    for c in continuity(e.xu(),e.xv(),n):e.AddConstraint(c)
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=rounded;options.rounding_seed=0;options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver();built=time.perf_counter();result=g.SolveShortestPath(src,dst,options)
    info={'solved':result.is_success(),'build_seconds':built-begin,'solve_seconds':time.perf_counter()-built,'horizon':horizon,'rounding':rounded}
    if not result.is_success():return info,None
    edges=g.GetSolutionPath(src,dst,result);ids={v.id():i for i,v in enumerate(vertices)};chosen=[e.v() for e in edges if e.v().id() in ids];segments=[{'region':ids[v.id()],'x':result.GetSolution(v.x()).tolist()} for v in chosen]
    info['duration']=float(segments[-1]['x'][-1]);info['objective']=trajectory_objective(segments,n,smoothness);info['smoothness']=smoothness;return info,segments


def fixed_route_solve(regions,sequence,start,goal,limits=LIMITS,horizon=80.,smoothness=0.):
    if smoothness<0:raise ValueError("smoothness must be nonnegative")
    n=len(start);p=MathematicalProgram();variables=[]
    for i,idx in enumerate(sequence):
        x=p.NewContinuousVariables(4*n+4,'segment'+str(i));variables.append(x)
        for q in x[:4*n].reshape(4,n):regions[idx].AddPointInSetConstraints(p,q)
        p.AddBoundingBoxConstraint(np.zeros(4),np.full(4,horizon),x[4*n:])
        for c in local_constraints(x,n,limits):p.AddConstraint(c)
        if smoothness:p.AddQuadraticCost(smoothness*curvature_energy(x,n),is_convex=True)
    for a,b in zip(variables,variables[1:]):
        for c in continuity(a,b,n):p.AddConstraint(c)
    for c in boundary(variables[0],start,n,True)+boundary(variables[-1],goal,n,False):p.AddConstraint(c)
    p.AddConstraint(variables[0][4*n]==0);p.AddLinearCost(variables[-1][-1]);result=ClarabelSolver().Solve(p)
    if not result.is_success():return None
    return [{'region':idx,'x':result.GetSolution(x).tolist()} for idx,x in zip(sequence,variables)]


def shortest(adj,source,target,banned_nodes=frozenset(),banned_edges=frozenset()):
    heap=[(0.,source)];dist={source:0.};prev={}
    while heap:
        cost,u=heapq.heappop(heap)
        if cost>dist[u]+1e-12:continue
        if u==target:
            path=[u]
            while u!=source:u=prev[u];path.append(u)
            return cost,tuple(reversed(path))
        for v,w in adj[u].items():
            if v in banned_nodes or (u,v) in banned_edges:continue
            d=cost+w
            if d<dist.get(v,np.inf)-1e-12:dist[v]=d;prev[v]=u;heapq.heappush(heap,(d,v))
    return None


def yen_paths(adj,source,target):
    first=shortest(adj,source,target)
    if first is None:return
    accepted=[first[1]];heap=[];queued={first[1]};yield first
    while True:
        previous=accepted[-1]
        for i in range(len(previous)-1):
            prefix=previous[:i+1];banned={(p[i],p[i+1]) for p in accepted if len(p)>i+1 and p[:i+1]==prefix}
            spur=shortest(adj,prefix[-1],target,set(prefix[:-1]),banned)
            if spur is None:continue
            path=prefix[:-1]+spur[1]
            if path in queued:continue
            queued.add(path);cost=sum(adj[u][v] for u,v in zip(path,path[1:]));heapq.heappush(heap,(cost,path))
        if not heap:return
        item=heapq.heappop(heap);accepted.append(item[1]);yield item


def portal_solve(regions,start,goal,limits=LIMITS,horizon=80.,routes=50,raw_path_cap=2000,budget_seconds=280.):
    from .full_task_portal_socp import make_roadmap
    from .global_portal_socp import PortalGraph
    begin=time.perf_counter();maps=make_roadmap(regions,[r.ChebyshevCenter() for r in regions],np.ones(len(start)),5);g=PortalGraph();source=g.node(start);target=g.node(goal);g.layer(regions,maps,np.ones(len(start)),{},[source,target]);adj=[{v:float(np.max(np.abs(g.points[u]-g.points[v])/limits)) for v in edges} for u,edges in enumerate(g.adj)];best=None;seen=set();attempts=[];raw_count=0
    for discrete,path in yen_paths(adj,source,target):
        raw_count+=1
        if time.perf_counter()-begin>=budget_seconds:break
        if raw_count>raw_path_cap:break
        sequence=[g.adj[u][v][1]['region'] for u,v in zip(path,path[1:])];sequence=tuple(r for i,r in enumerate(sequence) if not i or r!=sequence[i-1])
        if sequence in seen:continue
        seen.add(sequence);solution=fixed_route_solve(regions,sequence,start,goal,limits,horizon);duration=solution[-1]['x'][-1] if solution else None;attempts.append({'sequence':sequence,'solved':solution is not None,'duration':duration})
        if solution and (best is None or duration<best[-1]['x'][-1]):best=solution
        if len(attempts)>=routes:break
    return {'solved':best is not None,'duration':best[-1]['x'][-1] if best else None,'total_seconds':time.perf_counter()-begin,'attempts':attempts,'raw_paths':min(raw_count,raw_path_cap),'route_cap':routes,'raw_path_cap':raw_path_cap,'budget_seconds':budget_seconds,'portal_density':5,'scope':'Time-aware portal Yen alternatives, identical cubic space/time restriction constraints; bounded search, not a globally optimal continuous solver.'},best


def audit(segments,regions,start,goal,limits=LIMITS,distance=None):
    n=len(start);valid=True;maximum=0.;minimum=float('inf');frames=[];last=None
    for segment in segments:
        x=np.array(segment['x']);p=x[:4*n].reshape(4,n);h=x[4*n:];valid &= bool(np.all(np.diff(h)>=MIN_CLOCK_INCREMENT-1e-6));valid &= all(regions[segment['region']].PointInSet(v,1e-5) for v in p)
        valid &= bool(np.max(np.abs(np.diff(p,axis=0))-np.diff(h)[:,None]*limits)<=1e-5)
        if last is not None:valid &= bool(np.max(abs(p[0]-last[0]))<1e-5 and abs(h[0]-last[1])<1e-5 and np.max(abs((p[1]-p[0])-(last[0]-last[2])))<1e-5 and abs((h[1]-h[0])-(last[1]-last[3]))<1e-5)
        last=(p[-1],h[-1],p[-2],h[-2])
        for s in np.linspace(0,1,101):
            b=np.array([(1-s)**3,3*s*(1-s)**2,3*s*s*(1-s),s**3]);db=3*np.array([(1-s)**2,2*s*(1-s),s*s]);q=b@p;t=float(b@h);dh=float(db@np.diff(h));v=(db@np.diff(p,axis=0))/dh;maximum=max(maximum,float(np.max(abs(v)/limits)))
            if distance is not None:minimum=min(minimum,float(distance(q)))
            frames.append({'t':t,'q':q.tolist()})
    first=np.array(segments[0]['x']);end=np.array(segments[-1]['x']);valid &= bool(np.max(abs(first[:n]-start))<1e-5 and np.max(abs(end[3*n:4*n]-goal))<1e-5 and abs(first[4*n])<1e-5)
    valid &= bool(np.max(abs(first[n:2*n]-first[:n]))<1e-5 and np.max(abs(end[3*n:4*n]-end[2*n:3*n]))<1e-5)
    return {'kinematics_passed':bool(valid and maximum<=1.001),'maximum_velocity_ratio':maximum,'minimum_sampled_distance':minimum if distance else None,'collision_samples_passed':minimum>=-1e-7 if distance else None,'frames':frames,'scope':'Cubic analytic convex-hull velocity and C1 parameter checks plus sampled geometry. No acceleration/torque constraints or continuous collision certificate.'}


def region_yen_solve(regions,start,goal,limits=LIMITS,horizon=80.,
                     smoothness=0.,budget_seconds=30.,route_cap=None,on_incumbent=None):
    """Yen region routes + identical convex restriction; retain best objective.

    Budget checked between solves/routes; callers must enforce a hard process
    timeout if required. No GCS solution is used to construct the search.
    """
    begin=time.perf_counter();n=len(regions)
    points=[r.ChebyshevCenter() for r in regions]+[start,goal]
    adj=[{} for _ in points]
    def edge(i,j):adj[i][j]=float(np.max(abs(points[i]-points[j])/limits))
    for i,r in enumerate(regions):
        if r.PointInSet(start,1e-7):edge(n,i)
        if r.PointInSet(goal,1e-7):edge(i,n+1)
        for j in range(i):
            if r.IntersectsWith(regions[j]):edge(i,j);edge(j,i)
    best=None;best_cost=float('inf');attempts=[];first=None
    for _,path in yen_paths(adj,n,n+1):
        if time.perf_counter()-begin>=budget_seconds:break
        solution=fixed_route_solve(regions,path[1:-1],start,goal,limits,horizon,smoothness)
        cost=trajectory_objective(solution,len(start),smoothness) if solution else None
        attempts.append({'sequence':list(path[1:-1]),'objective':cost})
        if solution and cost<best_cost:
            best=solution;best_cost=cost
            if first is None:first=time.perf_counter()-begin
            if on_incumbent is not None:on_incumbent(best,best_cost,len(attempts))
        if route_cap is not None and len(attempts)>=route_cap:break
    return {'solved':best is not None,'objective':best_cost if best else None,
            'duration':best[-1]['x'][-1] if best else None,'attempts':attempts,
            'first_feasible_seconds':first,'total_seconds':time.perf_counter()-begin,
            'smoothness':smoothness,'budget_seconds':budget_seconds},best
