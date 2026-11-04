"""Deterministic bounded fixed-yaw guidance; never used as final trajectory."""
import heapq
import time
import numpy as np


def grid_guide(distance,start,goal,lower,upper,*,spacing=.08,margin=.0301,time_budget_s=120.):
    start=np.asarray(start);goal=np.asarray(goal)
    if abs(start[2]-goal[2])>1e-10:raise ValueError('Grid guidance requires equal endpoint yaw')
    xs=np.unique(np.r_[np.arange(lower[0],upper[0]+1e-9,spacing),start[0],goal[0]])
    ys=np.unique(np.r_[np.arange(lower[1],upper[1]+1e-9,spacing),start[1],goal[1]])
    width=len(xs);shape=(len(ys),width);began=time.perf_counter();cache={};edge_calls=0
    def point(i):y,x=divmod(i,width);return np.array([xs[x],ys[y],start[2]])
    def index(q):return int(np.argmin(abs(ys-q[1]))*width+np.argmin(abs(xs-q[0])))
    def clearance(i):
        if i not in cache:
            d,internal=distance(point(i));cache[i]=d if internal>=0 else -1.
        return cache[i]
    def valid_edge(a,b,da=None,db=None):
        nonlocal edge_calls
        edge_calls+=1;length=np.linalg.norm(b[:2]-a[:2])
        # Fixed-yaw translation residual is 1-Lipschitz, including a route witness.
        if da is not None and db is not None and min(da,db)-length/2>=margin:return True
        for t in np.linspace(0,1,max(2,int(np.ceil(length/.002))+1)):
            d,internal=distance((1-t)*a+t*b)
            if min(d-margin,internal)<0:return False
        return True
    source,target=index(start),index(goal);cost={source:0.};parent={};queue=[(np.linalg.norm(start[:2]-goal[:2]),source)]
    if min(clearance(source),clearance(target))<margin:raise RuntimeError('Grid guide endpoint rejected')
    visited=set()
    while queue:
        if time.perf_counter()-began>time_budget_s:raise RuntimeError('Grid guide time budget exhausted; no infeasibility claim')
        _,i=heapq.heappop(queue)
        if i in visited:continue
        visited.add(i)
        if i==target:break
        y,x=divmod(i,width);a=point(i)
        for dy in [-1,0,1]:
            for dx in [-1,0,1]:
                if not(dx or dy) or not(0<=y+dy<shape[0] and 0<=x+dx<width):continue
                j=(y+dy)*width+x+dx
                if j in visited:continue
                b=point(j);new=cost[i]+np.linalg.norm(b[:2]-a[:2])
                if new>=cost.get(j,np.inf) or clearance(j)<margin:continue
                if not valid_edge(a,b,clearance(i),clearance(j)):continue
                cost[j]=new;parent[j]=i;heapq.heappush(queue,(new+np.linalg.norm(b[:2]-goal[:2]),j))
    if target not in visited:raise RuntimeError('Fixed-yaw finite grid disconnected; no global infeasibility claim')
    ids=[target]
    while ids[-1]!=source:ids.append(parent[ids[-1]])
    raw=np.array([point(i) for i in ids[::-1]])
    # Exponential visibility shortening avoids quadratic repeated long rays.
    print(f'Grid guide: {len(visited)} expanded nodes, {len(raw)} raw path nodes',flush=True)
    shortened=[raw[0]];i=0
    while i<len(raw)-1:
        if time.perf_counter()-began>time_budget_s:raise RuntimeError('Grid guide shortening budget exhausted')
        good=i+1;step=2
        while i+step<len(raw) and valid_edge(raw[i],raw[i+step]):
            good=i+step;step*=2
        high=min(i+step,len(raw)-1)
        while high>good:
            mid=(good+high+1)//2
            if valid_edge(raw[i],raw[mid]):good=mid
            else:high=mid-1
        shortened.append(raw[good]);i=good
    return np.array(shortened),dict(method='deterministic_fixed_yaw_grid_Astar',spacing_m=spacing,margin_m=margin,
        tested_nodes=len(cache),expanded_nodes=len(visited),edge_checks=edge_calls,raw_nodes=len(raw),guide_nodes=len(shortened),wall_seconds=time.perf_counter()-began,
        scope='Finite fixed-yaw guidance with 2 mm edge sampling; final motion solved and audited by official GCS; no completeness claim')
