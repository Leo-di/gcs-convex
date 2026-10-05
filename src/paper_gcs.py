"""GCS transcription of arXiv:2205.04422, equations (7)--(11).

Default: degree 6, C2, h'_min=.1, epsilon=.1 (Section 7.1 smoothing
configuration). Robot limits/regions are supplied separately. Clarabel replaces
paper's MOSEK; no claim of identical solver or experimental reproduction.
All local linear conditions belong to X_v, including time/velocity constraints.
Cost bounds use Bernstein convex-hull upper bounds as in Section 5.4.
No hard acceleration, torque, contact dynamics, or globally certified C-space.
"""
from dataclasses import dataclass,asdict
from math import factorial,comb
import time
import numpy as np
from pydrake.geometry.optimization import HPolyhedron,Point,GraphOfConvexSets,GraphOfConvexSetsOptions,CartesianProduct,Intersection
from pydrake.solvers import ClarabelSolver,MathematicalProgram,Binding,Cost,L2NormCost,PerspectiveQuadraticCost

@dataclass(frozen=True)
class Config:
    degree:int=6
    continuity:int=2
    clock_derivative_min:float=.1
    horizon:float=80.
    time_weight:float=1.
    length_weight:float=0.
    energy_weight:float=0.
    smoothing_weight:float=.1
    rounded_paths:int=50
    rounding_trials:int=500
    seed:int=0
    def __post_init__(self):
        if self.degree<max(2,self.continuity+1) or self.continuity<0:raise ValueError('Invalid degree/continuity')
        if self.clock_derivative_min<=0 or self.horizon<=0:raise ValueError('Positive clock bounds required')
        if min(self.time_weight,self.length_weight,self.energy_weight,self.smoothing_weight)<0:raise ValueError('Nonnegative weights required')


def split(x,n,cfg):
    k=cfg.degree+1
    return np.asarray(x[:k*n]).reshape(k,n),np.asarray(x[k*n:])


def derivative_matrix(degree,order):
    return factorial(degree)/factorial(degree-order)*np.diff(np.eye(degree+1),n=order,axis=0)


def vertex_set(region,limits,cfg):
    n=len(limits);d=cfg.degree;k=d+1;dim=k*(n+1);A=[];b=[]
    for i in range(k):
        for row,rhs in (zip(region.A(),region.b()) if isinstance(region,HPolyhedron) else []):
            v=np.zeros(dim);v[i*n:(i+1)*n]=row;A.append(v);b.append(rhs)
        v=np.zeros(dim);v[k*n+i]=1;A.extend([v,-v]);b.extend([cfg.horizon,0.])
    for i in range(d):
        dt=np.zeros(dim);dt[k*n+i+1]=1;dt[k*n+i]=-1
        A.append(-dt);b.append(-cfg.clock_derivative_min/d)
        for j in range(n):
            dp=np.zeros(dim);dp[(i+1)*n+j]=1;dp[i*n+j]=-1
            A.extend([dp-limits[j]*dt,-dp-limits[j]*dt]);b.extend([0.,0.])
    local=HPolyhedron(np.asarray(A),np.asarray(b))
    if isinstance(region,HPolyhedron):return local
    clock=HPolyhedron.MakeBox(np.zeros(k),np.full(k,cfg.horizon))
    return Intersection(CartesianProduct([region]*k+[clock]),local)


def seam(a,b,n,cfg):
    pa,ha=split(a,n,cfg);pb,hb=split(b,n,cfg);out=[]
    for order in range(cfg.continuity+1):
        D=derivative_matrix(cfg.degree,order)
        for j in range(n):out.append(sum(D[-1,i]*pa[i,j]-D[0,i]*pb[i,j] for i in range(cfg.degree+1))==0)
        out.append(sum(D[-1,i]*ha[i]-D[0,i]*hb[i] for i in range(cfg.degree+1))==0)
    return out


def boundary(x,q,n,cfg,start):
    p,h=split(x,n,cfg);i=0 if start else cfg.degree;j=1 if start else cfg.degree-1
    out=[p[i,k]==q[k] for k in range(n)]+[p[i,k]==p[j,k] for k in range(n)]
    if start:out.append(h[0]==0)
    return out


def smoothing_matrix(n,cfg):
    # Jensen upper bound of integral squared second derivative (Sec5.4/6).
    D=derivative_matrix(cfg.degree,2);k=cfg.degree+1
    B=np.zeros(((n+1)*(cfg.degree-1),k*(n+1)))
    for j in range(n):
        for t in range(cfg.degree-1):
            for i in range(k):B[j*(cfg.degree-1)+t,i*n+j]=D[t,i]
    B[n*(cfg.degree-1):,k*n:]=D
    return B/np.sqrt(cfg.degree-1)


def add_costs(owner,x,n,cfg,graph):
    p,h=split(x,n,cfg);dim=len(x);k=cfg.degree+1
    linear=cfg.time_weight*(h[-1]-h[0])
    if graph:owner.AddCost(linear)
    else:owner.AddLinearCost(linear)
    if cfg.smoothing_weight:
        B=smoothing_matrix(n,cfg);Q=2*cfg.smoothing_weight*(B.T@B)
        if graph:
            from pydrake.solvers import QuadraticCost
            owner.AddCost(Binding[Cost](QuadraticCost(Q,np.zeros(dim),0.,True),x))
        else:owner.AddQuadraticCost(Q,np.zeros(dim),x,is_convex=True)
    for i in range(cfg.degree):
        M=np.zeros((n,dim))
        for j in range(n):M[j,(i+1)*n+j]=1;M[j,i*n+j]=-1
        if cfg.length_weight:
            if graph:owner.AddCost(Binding[Cost](L2NormCost(cfg.length_weight*M,np.zeros(n)),x))
            else:
                t=owner.NewContinuousVariables(1)[0];owner.AddLorentzConeConstraint(np.array([t]+[p[i+1,j]-p[i,j] for j in range(n)],dtype=object));owner.AddLinearCost(cfg.length_weight*t)
        if cfg.energy_weight:
            V=np.zeros((n+1,dim));V[0,k*n+i+1]=1;V[0,k*n+i]=-1;V[1:]=np.sqrt(cfg.energy_weight)*M
            if graph:owner.AddCost(Binding[Cost](PerspectiveQuadraticCost(V,np.zeros(n+1)),x))
            else:
                t=owner.NewContinuousVariables(1)[0];owner.AddRotatedLorentzConeConstraint(np.array([t,h[i+1]-h[i]]+[float(np.sqrt(cfg.energy_weight))*(p[i+1,j]-p[i,j]) for j in range(n)],dtype=object));owner.AddLinearCost(t)


def objective(segments,n,cfg):
    result=0.
    for s in segments:
        x=np.asarray(s['x']);p,h=split(x,n,cfg);dp=np.diff(p,axis=0);dt=np.diff(h)
        result+=cfg.time_weight*(h[-1]-h[0])+cfg.length_weight*np.linalg.norm(dp,axis=1).sum()+cfg.energy_weight*np.sum(dp**2/dt[:,None])+cfg.smoothing_weight*np.sum((smoothing_matrix(n,cfg)@x)**2)
    return float(result)


def solve(regions,start,goal,limits,cfg=Config(),relaxation_only=False):
    begin=time.perf_counter();n=len(start);g=GraphOfConvexSets();vertices=[]
    for i,r in enumerate(regions):
        v=g.AddVertex(vertex_set(r,limits,cfg),str(i));add_costs(v,v.x(),n,cfg,True);vertices.append(v)
    source=g.AddVertex(Point(np.zeros(1)),'source');target=g.AddVertex(Point(np.zeros(1)),'target')
    for i,r in enumerate(regions):
        if r.PointInSet(start,1e-7):
            e=g.AddEdge(source,vertices[i])
            for c in boundary(e.xv(),start,n,cfg,True):e.AddConstraint(c)
        if r.PointInSet(goal,1e-7):
            e=g.AddEdge(vertices[i],target)
            for c in boundary(e.xu(),goal,n,cfg,False):e.AddConstraint(c)
        for j in range(i):
            if r.IntersectsWith(regions[j]):
                for a,b in [(i,j),(j,i)]:
                    e=g.AddEdge(vertices[a],vertices[b])
                    for c in seam(e.xu(),e.xv(),n,cfg):e.AddConstraint(c)
    opts=GraphOfConvexSetsOptions();opts.convex_relaxation=True;opts.preprocessing=True;opts.max_rounded_paths=0 if relaxation_only else cfg.rounded_paths;opts.max_rounding_trials=cfg.rounding_trials;opts.rounding_seed=cfg.seed;opts.solver=ClarabelSolver();opts.restriction_solver=ClarabelSolver();opts.preprocessing_solver=ClarabelSolver()
    built=time.perf_counter();result=g.SolveShortestPath(source,target,opts)
    info={'solved':result.is_success(),'config':asdict(cfg),'preprocessing':True,'build_seconds':built-begin,'solve_seconds':time.perf_counter()-built,'solver_status':str(result.get_solution_result()),'relaxation_only':relaxation_only}
    if not result.is_success():return info,None
    info['solver_objective']=float(result.get_optimal_cost())
    if relaxation_only:return info,None
    ids={v.id():i for i,v in enumerate(vertices)};path=g.GetSolutionPath(source,target,result)
    segments=[{'region':ids[e.v().id()],'degree':cfg.degree,'x':result.GetSolution(e.v().x()).tolist()} for e in path if e.v().id() in ids]
    info['objective']=objective(segments,n,cfg);info['duration']=segments[-1]['x'][-1]
    return info,segments


def fixed_route(regions,sequence,start,goal,limits,cfg=Config()):
    n=len(start);prog=MathematicalProgram();variables=[]
    for idx in sequence:
        x=prog.NewContinuousVariables((cfg.degree+1)*(n+1));variables.append(x);vertex_set(regions[idx],limits,cfg).AddPointInSetConstraints(prog,x);add_costs(prog,x,n,cfg,False)
    for a,b in zip(variables,variables[1:]):
        for c in seam(a,b,n,cfg):prog.AddConstraint(c)
    for c in boundary(variables[0],start,n,cfg,True)+boundary(variables[-1],goal,n,cfg,False):prog.AddConstraint(c)
    result=ClarabelSolver().Solve(prog)
    if not result.is_success():return None
    return [{'region':i,'degree':cfg.degree,'x':result.GetSolution(x).tolist()} for i,x in zip(sequence,variables)]


def evaluate(controls,u,order=0):
    d=len(controls)-1;D=derivative_matrix(d,order);c=D@controls;m=d-order
    basis=np.array([comb(m,i)*u**i*(1-u)**(m-i) for i in range(m+1)])
    return basis@c


def audit(segments,regions,start,goal,limits,cfg=Config(),distance=None):
    n=len(start);valid=True;maxv=0.;mina=float('inf');maxa=0.;seam_error=0.
    prev=None
    for s in segments:
        x=np.asarray(s['x']);p,h=split(x,n,cfg)
        valid &= vertex_set(regions[s['region']],limits,cfg).PointInSet(x,1e-5)
        if prev is not None:
            for order in range(cfg.continuity+1):
                seam_error=max(seam_error,float(np.max(abs(evaluate(prev[0],1.,order)-evaluate(p,0.,order)))),float(abs(evaluate(prev[1],1.,order)-evaluate(h,0.,order))))
        prev=(p,h)
        for u in np.linspace(0,1,201):
            hp=evaluate(h,u,1);rp=evaluate(p,u,1);v=rp/hp
            a=(evaluate(p,u,2)*hp-rp*evaluate(h,u,2))/hp**3
            maxv=max(maxv,float(max(abs(v)/limits)));maxa=max(maxa,float(max(abs(a))))
            if distance is not None:mina=min(mina,float(distance(evaluate(p,u))))
    p0,h0=split(np.asarray(segments[0]['x']),n,cfg);p1,h1=split(np.asarray(segments[-1]['x']),n,cfg)
    boundary_error=max(float(max(abs(p0[0]-start))),float(max(abs(p1[-1]-goal))),float(max(abs(evaluate(p0,0.,1)))),float(max(abs(evaluate(p1,1.,1)))),abs(float(h0[0])))
    return {'passed':bool(valid and seam_error<=1e-5 and boundary_error<=1e-5 and maxv<=1.001 and (distance is None or mina>=-1e-7)),'set_membership':bool(valid),'seam_derivative_error':seam_error,'boundary_error':boundary_error,'max_velocity_ratio':maxv,'sampled_peak_acceleration_mixed_units':maxa,'minimum_sampled_distance':mina if distance is not None else None,'collision_checked':distance is not None,'scope':'Full Bernstein set/derivative constraints plus sampled physical derivatives and optional geometry; no hard acceleration or certified continuous collision claim.'}
