"""Bounded deterministic RRT repair in the original attached-cup 10D model."""
import time
import numpy as np
from .build_scene import set_robot_configuration
from .iris_region_generator import _make_collision_checker,_make_parameterization,_reduced_position_indices,_base_region_specs,_base_in_region


def safe_bootstrap(scene, reference, labels, budget=25., max_iterations=4000):
    reference=np.asarray(reference);start,goal=reference[0],reference[-1]
    set_robot_configuration(scene,start[:3],start[3:],scene.gripper_closed_positions)
    parameterize=_make_parameterization(scene.plant.GetPositions(scene.plant_context).copy(),_reduced_position_indices(scene))
    checker=_make_collision_checker(scene)
    specs=[_base_region_specs(scene.config)[n] for n in labels]
    weights=np.array([1.,1.,.35]+[.2]*7);began=time.perf_counter();calls=0
    def valid(q):
        nonlocal calls
        calls+=1
        return any(_base_in_region(q,s) for s in specs) and checker.CheckConfigCollisionFree(parameterize(q))
    def edge(a,b):
        count=max(2,int(np.ceil(np.max(np.abs(b-a))/.02))+1)
        return all(valid((1-t)*a+t*b) for t in np.linspace(0,1,count))
    report={'seed':20260929,'edge_max_coordinate_step':.02,'budget_seconds':budget,'scope':'Sampled edges; finite search failure is not an infeasibility proof.'}
    def finish(path,status):
        report.update(status=status,seconds=time.perf_counter()-began,collision_queries=calls)
        return path,report
    if not valid(start) or not valid(goal):return finish(None,'endpoint_rejected')
    if all(edge(a,b) for a,b in zip(reference,reference[1:])):return finish(reference,'reference_valid')
    rng=np.random.default_rng(20260929);trees=[([start.copy()],[-1]),([goal.copy()],[-1])]
    lo=np.array(scene.config['whole_body_planning']['joint_lower_limits']);hi=np.array(scene.config['whole_body_planning']['joint_upper_limits'])
    def extend(tree,target):
        nodes,parents=tree;idx=int(np.argmin(np.linalg.norm((np.array(nodes)-target)*weights,axis=1)))
        delta=target-nodes[idx];length=np.linalg.norm(delta*weights);q=nodes[idx]+delta*min(1.,.18/max(length,1e-12))
        if not edge(nodes[idx],q):return None,False
        nodes.append(q);parents.append(idx);return len(nodes)-1,length<=.18
    def ancestry(tree,i):
        out=[]
        while i!=-1:out.append(tree[0][i]);i=tree[1][i]
        return out[::-1]
    for iteration in range(max_iterations):
        if time.perf_counter()-began>budget:break
        side=iteration%2
        if rng.random()<.75:
            target=reference[rng.integers(len(reference))].copy()+rng.normal(size=10)*np.array([.25,.25,.15]+[.2]*7)
            target[3:]=np.clip(target[3:],lo,hi)
        else:
            s=specs[rng.integers(len(specs))];target=np.r_[rng.uniform(s['lower'],s['upper']),rng.uniform(*s['yaw_bounds']),rng.uniform(lo,hi)]
        idx,_=extend(trees[side],target)
        if idx is None:continue
        for _ in range(100):
            if time.perf_counter()-began>budget:break
            other,reached=extend(trees[1-side],trees[side][0][idx])
            if other is None:break
            if reached:
                a=ancestry(trees[side],idx);b=ancestry(trees[1-side],other)
                path=a+b[-2::-1] if side==0 else b+a[-2::-1]
                # Deterministic shortcutting, every shortcut collision checked.
                compact=[path[0]];i=0
                while i<len(path)-1:
                    j=len(path)-1
                    while j>i+1 and not edge(path[i],path[j]):j-=1
                    compact.append(path[j]);i=j
                dense=[]
                for a,b in zip(compact,compact[1:]):
                    n=max(2,int(np.ceil(np.max(np.abs(b-a))/.02))+1)
                    dense.extend((1-t)*a+t*b for t in np.linspace(0,1,n)[:-1])
                dense.append(goal);report['iterations']=iteration+1
                return finish(np.array(dense),'rrt_repaired')
    return finish(None,'search_budget_exhausted')
