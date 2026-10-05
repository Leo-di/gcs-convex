"""Same-cover GCS removal: portal Dijkstra, then fixed-sequence SOCP.

Neither roadmap construction nor fixed-sequence refinement calls GCS. This
isolates the solver/continuous-variable contribution from region semantics.
"""
import copy
import heapq
import json
from pathlib import Path
import time

import numpy as np
from pydrake.solvers import ClarabelSolver, MathematicalProgram

from .arm_branch_gcs import ROOT as LIBRARY, make_families, region_specs, solve_branches, audit
from .build_scene import build_scene

OUTPUT=Path('outputs/results/gcs_solver_ablation_v1')
WEIGHTS=np.array([1,1,.35]+[.12]*7)


def portal_search(families,covers,case,density):
    began=time.perf_counter();specs=region_specs(families,covers,case,list(families),True,transit_boundary_margin=.0001)
    coordinates=[];adjacency=[];members=[[] for _ in specs]
    def node(q):
        coordinates.append(q);adjacency.append([]);return len(coordinates)-1
    def edge(a,b,region=None):
        cost=0. if region is None else float(np.linalg.norm(WEIGHTS*(coordinates[a]-coordinates[b])))
        adjacency[a].append((b,cost,region))
    home=next(iter(families.values()))
    source=node(home.state(case['start_x'],0));target=node(home.state(.6,0))
    def member(i,q):
        for n in members[i]:
            if np.max(np.abs(coordinates[n]-q))<1e-10:return n
        n=node(q);members[i].append(n);return n
    for i,spec in enumerate(specs):
        a=spec['rect']
        if spec['mode']=='room' and a[0]<=case['start_x']<=a[1] and a[2]<=0<=a[3]:
            edge(source,member(i,coordinates[source]))
        if spec['mode']=='transit' and a[0]<=.6<=a[1] and a[2]<=0<=a[3]:
            edge(member(i,coordinates[target]),target)
        for j,other in enumerate(specs):
            if j<=i or spec['branch']!=other['branch']:continue
            b=other['rect'];lo=[max(a[0],b[0]),max(a[2],b[2])];hi=[min(a[1],b[1]),min(a[3],b[3])]
            if any(l>h+1e-10 for l,h in zip(lo,hi)):continue
            axes=[[(l+h)/2] if density==1 or h-l<1e-10 else np.linspace(l,h,density) for l,h in zip(lo,hi)]
            for x in axes[0]:
                for alpha in axes[1]:
                    q=families[spec['branch']].state(x,alpha);u=member(i,q);v=member(j,q)
                    if not(spec['mode']=='transit' and other['mode']=='room'):edge(u,v)
                    if not(other['mode']=='transit' and spec['mode']=='room'):edge(v,u)
    for i,vertices in enumerate(members):
        for a in vertices:
            for b in vertices:
                if a!=b:edge(a,b,i)
    build_seconds=time.perf_counter()-began;began=time.perf_counter()
    distance={source:0.};parent={};queue=[(0.,source)]
    while queue:
        value,u=heapq.heappop(queue)
        if value>distance[u]+1e-12:continue
        if u==target:break
        for v,cost,region in adjacency[u]:
            candidate=value+cost
            if candidate<distance.get(v,float('inf'))-1e-12:
                distance[v]=candidate;parent[v]=(u,region);heapq.heappush(queue,(candidate,v))
    info=dict(build_seconds=build_seconds,solve_seconds=time.perf_counter()-began,
              nodes=len(coordinates),edges=sum(map(len,adjacency)),density=density)
    if target not in parent:return dict(**info,status='unreachable_in_roadmap',path=[])
    steps=[];v=target
    while v!=source:
        u,region=parent[v]
        if region is not None:
            spec=specs[region]
            segment=dict(**spec,start=coordinates[u].tolist(),end=coordinates[v].tolist())
            steps.append(segment)
        v=u
    path=list(reversed(steps))
    # Merge adjacent intra-region edges before a fair continuous refinement.
    merged=[]
    for segment in path:
        if merged and all(merged[-1][k]==segment[k] for k in ('branch','mode','index')):
            merged[-1]['end']=segment['end']
        else:merged.append(segment)
    return dict(**info,status='solved',path=merged,selected_branch=merged[0]['branch'],
                metric_length=distance[target])


def refine_sequence(families,plan):
    began=time.perf_counter()
    if not plan['path']:return copy.deepcopy(plan)
    prog=MathematicalProgram();path=plan['path'];variables=[]
    for i,segment in enumerate(path):
        region=families[segment['branch']].polytope(*segment['rect'])
        x=prog.NewContinuousVariables(20,f'segment{i}');variables.append(x)
        for half in [x[:10],x[10:]]:prog.AddLinearConstraint(region.A(),np.full(len(region.b()),-np.inf),region.b(),half)
        prog.AddL2NormCost(np.c_[-np.diag(WEIGHTS),np.diag(WEIGHTS)],np.zeros(10),x)
        if i:
            prog.AddLinearEqualityConstraint(np.c_[np.eye(10),-np.eye(10)],np.zeros(10),np.r_[variables[i-1][10:],x[:10]])
    prog.AddBoundingBoxConstraint(path[0]['start'],path[0]['start'],variables[0][:10])
    prog.AddBoundingBoxConstraint(path[-1]['end'],path[-1]['end'],variables[-1][10:])
    build_seconds=time.perf_counter()-began;began=time.perf_counter();solution=ClarabelSolver().Solve(prog)
    result=copy.deepcopy(plan)
    result.update(refinement_build_seconds=build_seconds,refinement_solve_seconds=time.perf_counter()-began,
                  refinement_status=str(solution.get_solution_result()))
    if not solution.is_success():result.update(status='refinement_failed',path=[]);return result
    for segment,x in zip(result['path'],variables):
        values=solution.GetSolution(x);segment['start']=values[:10].tolist();segment['end']=values[10:].tolist()
    result['metric_length']=float(solution.get_optimal_cost())
    return result


def metric_length(plan):
    return sum(float(np.linalg.norm(WEIGHTS*(np.array(p['end'])-p['start']))) for p in plan['path'])


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    fixture=json.loads(Path('tests/fixtures/arm_boundary.json').read_text())
    families=make_families(build_scene(config=fixture['config'],visualize=False),fixture)
    covers=json.loads((LIBRARY/'cover.json').read_text());cases=json.loads((LIBRARY/'protocol.json').read_text())['cases']
    (OUTPUT/'protocol.json').write_text(json.dumps(dict(cases=cases,densities=[1,3,9],
        methods=['gcs','portal_dijkstra','portal_dijkstra_then_socp'],
        source_cover=str(LIBRARY/'cover.json'),shared='same region cover, semantics, full configuration metric and endpoints',
        transit_boundary_margin_m=.0001,
        development_note='Initial run exposed transit cells admitting illegal poses exactly at the gate; retained in gcs_solver_ablation_boundary_diagnostic. All methods now receive the same conservative boundary correction.',
        scope='Reference-path solver ablation; no new dynamics executions; no OMPL PRM reproduction',
        main_outcomes=['valid task path','weighted length','online build plus solve time','GCS numerical relaxation bound']),indent=2))
    records=[]
    for case in cases:
        plans=[('gcs',solve_branches(families,covers,case,list(families),measure_relaxation=True,transit_boundary_margin=.0001))]
        for density in [1,3,9]:
            discrete=portal_search(families,covers,case,density)
            plans.extend([(f'dijkstra_{density}',discrete),(f'dijkstra_{density}_socp',refine_sequence(families,discrete))])
        for method,result in plans:
            check,rows=audit(families,result,case)
            result['metric_length']=metric_length(result) if result['path'] else None
            result['online_seconds']=sum(result.get(k,0.) for k in ['build_seconds','solve_seconds','refinement_build_seconds','refinement_solve_seconds'])
            record=dict(case=case,method=method,result=result,audit=check);records.append(record)
            directory=OUTPUT/case['name']/method;directory.mkdir(parents=True,exist_ok=True)
            (directory/'result.json').write_text(json.dumps(record,indent=2));(directory/'trace.json').write_text(json.dumps(rows))
            print(case['name'],method,check['accepted'],result['metric_length'],result['online_seconds'],flush=True)
    (OUTPUT/'summary.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':main()
