"""Bounded IRIS bridge-seed heuristic; final motion is still official GCS.

Find closest points between convex components with a convex QP, and propose
their midpoint as an IRIS seed. The chord is NOT an accepted robot trajectory.
"""
import copy
import time
import numpy as np
from pydrake.solvers import MathematicalProgram, ClarabelSolver
from .online_carry_iris import sets, grow_carry_region
from .online_iris_attachment import connection_status
from .iris_region_generator import _base_region_specs, _base_in_region


def component(regions,point):
    found={i for i,r in enumerate(regions) if r.PointInSet(point)};pending=list(found)
    while pending:
        i=pending.pop()
        for j,r in enumerate(regions):
            if j not in found and regions[i].IntersectsWith(r):found.add(j);pending.append(j)
    return found


def closest_points(a,b,weights):
    n=a.ambient_dimension();p=MathematicalProgram();x=p.NewContinuousVariables(n,'a');y=p.NewContinuousVariables(n,'b')
    p.AddLinearConstraint(a.A(),np.full(len(a.b()),-np.inf),a.b(),x)
    p.AddLinearConstraint(b.A(),np.full(len(b.b()),-np.inf),b.b(),y)
    metric=np.diag(np.asarray(weights)**2)
    # Numeric Hessian avoids NumPy ufunc warnings on Drake symbolic arrays.
    p.AddQuadraticCost(2*np.block([[metric,-metric],[-metric,metric]]),
                       np.zeros(2*n),np.r_[x,y])
    result=ClarabelSolver().Solve(p)
    if not result.is_success():raise RuntimeError('Closest-region QP failed')
    left,right=result.GetSolution(x),result.GetSolution(y)
    if max(np.max(a.A()@left-a.b()),np.max(b.A()@right-b.b()))>1e-6:
        raise RuntimeError('Closest-region QP containment audit failed')
    return left,right,float(np.linalg.norm(np.asarray(weights)*(left-right)))


def bridge(config,records,anchor,start,goal,*,seed_base=150000,max_calls=4,grow_fn=grow_carry_region):
    current=copy.deepcopy(records);history=[];tried=[];begin=time.perf_counter();weights=np.array([1,1,.35]+[.12]*7)
    before=connection_status(sets(current),start,goal)
    if not before['endpoints_covered']:
        return current,{'before':before,'after':before,'attempts':[],'reason':'endpoint_not_covered','wall_seconds':0.}
    while len(history)<max_calls:
        regions=sets(current);state=connection_status(regions,start,goal)
        if state['region_graph_connected']:break
        left=component(regions,start);right=component(regions,goal);pairs=[]
        for i in sorted(left):
            for j in sorted(right):
                a,b,d=closest_points(regions[i],regions[j],weights);pairs.append((d,i,j,a,b))
        proposals=[]
        for d,i,j,a,b in sorted(pairs,key=lambda p:(p[0],p[1],p[2])):
            for alpha in [.5,.25,.75]:
                seed=(1-alpha)*a+alpha*b
                if any(np.linalg.norm((seed-old)*weights)<1e-4 for old in tried):continue
                labels=[name for name,spec in _base_region_specs(config).items() if _base_in_region(seed,spec)]
                if labels:proposals.append((seed,labels[0],{'left_region':i,'right_region':j,'weighted_gap':d,'alpha':alpha}))
        if not proposals:break
        seed,label,meta=proposals[0];tried.append(seed.copy())
        try:record=grow_fn(config,anchor,seed,label,seed_base+len(history))
        except RuntimeError as error:record={'accepted':False,'error':str(error)}
        entry={**meta,'seed':seed.tolist(),'region':record};history.append(entry)
        if record['accepted']:current.append(dict(record,online_bridge=True))
    return current,{'before':before,'after':connection_status(sets(current),start,goal),'attempts':history,
                    'wall_seconds':time.perf_counter()-begin,'max_calls':max_calls,
                    'scope':'New bounded closest-component QP midpoint heuristic for choosing IRIS seeds; not a published bridging algorithm, no completeness/safety claim, chord never used as planned motion.'}
