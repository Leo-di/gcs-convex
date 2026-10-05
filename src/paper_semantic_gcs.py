"""Paper Bézier transcription on the fixed-primitive delivery product graph.

No polyline fallback. Empty disks are represented exactly as SOC sets; the
existing library restricts empty navigation to fixed yaw/arm, lifted explicitly.
All variable motion uses degree6/C2 defaults. Pick/place are existing fixed
primitives, NOT claimed to be C2 or newly contact-validated. Cost/time cover
variable motion only; clock pauses through fixed actions. Not total task time.
"""
from dataclasses import asdict
import time
import numpy as np
from pydrake.common import Parallelism
from pydrake.geometry.optimization import GraphOfConvexSets,GraphOfConvexSetsOptions,HPolyhedron,Hyperellipsoid,Point
from pydrake.solvers import ClarabelSolver
from .paper_gcs import Config,vertex_set,add_costs,seam,boundary,split
from .semantic_candidate_gcs import q,PICK_PHASES,PLACE_PHASES
from .compare_carry_planners import LIMITS
from .timed_carry_bezier import yen_paths


def guard_ok(guard):
    return bool(guard) and all(np.isfinite(v) and v>=-1e-8 for v in guard.values())


def options(cfg):
    o=GraphOfConvexSetsOptions();o.convex_relaxation=True;o.preprocessing=True;o.max_rounded_paths=cfg.rounded_paths;o.max_rounding_trials=cfg.rounding_trials;o.rounding_seed=cfg.seed;o.solver=ClarabelSolver();o.restriction_solver=ClarabelSolver();o.preprocessing_solver=ClarabelSolver();o.parallelism=Parallelism(4)
    return o


def compile_graph(data,cfg=Config(),allowed=None):
    if data['config'].get('inspection_sites'):raise NotImplementedError('Inspection automaton must be compiled explicitly; refusing to ignore it')
    candidates=data['candidates'];allowed=set(candidates if allowed is None else allowed)
    if not candidates:raise ValueError('No candidate library')
    initial=q(next(iter(candidates.values()))['rows'][0]);n=len(initial)
    g=GraphOfConvexSets();meta={};edges=[];vertices=[]
    def vertex(set_,name,record):
        v=g.AddVertex(set_,name);meta[v.id()]=record;vertices.append(v);return v
    source=vertex(Point(np.array([0.])), 'start',{'kind':'start','entry':initial,'exit':initial})
    target=vertex(Point(np.array([0.])), 'accept',{'kind':'accept','entry':initial,'exit':initial})
    def motion(region,name,record,limits):
        v=vertex(vertex_set(region,limits,cfg),name,record);add_costs(v,v.x(),len(limits),cfg,True);return v
    def edge(a,b,constraints=()):
        e=g.AddEdge(a,b)
        for c in constraints:e.AddConstraint(c)
        edges.append(e);return e
    nav=[];disks=[]
    for i,d in enumerate(data['disks']):
        region=Hyperellipsoid.MakeHypersphere(d['radius'],np.asarray(d['center']));disks.append(region)
        center=initial.copy();center[:2]=d['center']
        v=motion(region,f'empty_{i}',{'kind':'empty','disk':i,'dimension':2,'entry':center,'exit':center},LIMITS[:2]);nav.append(v)
        if region.PointInSet(initial[:2],1e-7):edge(source,v,boundary(v.x(),initial[:2],2,cfg,True))
        for j in range(i):
            if region.IntersectsWith(disks[j]):
                for a,b in [(v,nav[j]),(nav[j],v)]:edge(a,b,seam(a.x(),b.x(),2,cfg))
    branches=[]
    for name,c in candidates.items():
        if name not in allowed or not guard_ok(c['guards'].get('pick',{})):continue
        pick=[r for r in c['rows'] if r['phase'] in PICK_PHASES]
        if not pick:raise ValueError('Missing pick primitive')
        if not np.allclose(q(pick[0])[2:],initial[2:],atol=1e-7):raise ValueError('Empty-cover posture incompatible with pick entry')
        goals=[(key,v) for key,v in data['goals'].get(name,{}).items() if guard_ok(v['guards'].get('place',{}))]
        if not goals:continue
        pv=vertex(HPolyhedron.MakeBox(np.array([0.]),np.array([cfg.horizon])),f'pick_{name}',{'kind':'pick','candidate':name,'entry':q(pick[0]),'exit':q(pick[-1])})
        for v,r in zip(nav,disks):
            if r.PointInSet(q(pick[0])[:2],1e-7):
                _,h=split(v.x(),2,cfg)
                edge(v,pv,boundary(v.x(),q(pick[0])[:2],2,cfg,False)+[h[-1]==pv.x()[0]])
        regions=[HPolyhedron(np.asarray(r['A']),np.asarray(r['b'])) for r in c['cover']['regions']];carry=[]
        for i,r in enumerate(regions):
            center=r.ChebyshevCenter();v=motion(r,f'carry_{name}_{i}',{'kind':'carry','candidate':name,'region':i,'dimension':n,'entry':center,'exit':center},LIMITS);carry.append(v)
            if r.PointInSet(q(pick[-1]),1e-7):
                p,h=split(v.x(),n,cfg)
                constraints=[p[0,j]==q(pick[-1])[j] for j in range(n)]+[p[1,j]==p[0,j] for j in range(n)]+[h[0]==pv.x()[0]]
                edge(pv,v,constraints)
            for j in range(i):
                if r.IntersectsWith(regions[j]):
                    for a,b in [(v,carry[j]),(carry[j],v)]:edge(a,b,seam(a.x(),b.x(),n,cfg))
        for gid,value in goals:
            place=[r for r in value['rows'] if r['phase'] in PLACE_PHASES]
            if not place:raise ValueError('Missing place primitive')
            lv=vertex(HPolyhedron.MakeBox(np.array([0.]),np.array([cfg.horizon])),f'place_{name}_{gid}',{'kind':'place','candidate':name,'goal':gid,'entry':q(place[0]),'exit':q(place[-1])})
            for v,r in zip(carry,regions):
                if r.PointInSet(q(place[0]),1e-7):
                    _,h=split(v.x(),n,cfg);edge(v,lv,boundary(v.x(),q(place[0]),n,cfg,False)+[h[-1]==lv.x()[0]])
            edge(lv,target);branches.append((name,gid))
    return {'graph':g,'source':source,'target':target,'meta':meta,'edges':edges,'vertices':vertices,'initial':initial,'branches':branches,'config':cfg}


def extract(model,result,path):
    meta=model['meta'];vertices=[model['source']]+[e.v() for e in path];out=[]
    for v in vertices:
        record={k:(val.tolist() if isinstance(val,np.ndarray) else val) for k,val in meta[v.id()].items()};record['x']=result.GetSolution(v.x()).tolist();out.append(record)
    actions=[s for s in out if s['kind'] in ['pick','place']]
    if [s['kind'] for s in actions]!=['pick','place'] or actions[0]['candidate']!=actions[1]['candidate']:raise RuntimeError('Invalid semantic mode path')
    return {'selected_candidate':actions[0]['candidate'],'selected_goal':actions[1]['goal'],'objective':float(result.get_optimal_cost()),'variable_motion_duration':actions[-1]['x'][0],'path':out}


def solve(data,cfg=Config(),allowed=None):
    begin=time.perf_counter();m=compile_graph(data,cfg,allowed);built=time.perf_counter();r=m['graph'].SolveShortestPath(m['source'],m['target'],options(cfg));info={'solved':r.is_success(),'config':asdict(cfg),'build_seconds':built-begin,'solve_seconds':time.perf_counter()-built,'vertices':len(m['vertices']),'edges':len(m['edges']),'legal_branches':m['branches'],'scope':'Joint empty navigation, grasp/goal choice and carrying. C2 within variable-motion modes only; existing fixed actions and fixed empty posture; objective excludes fixed-action execution time.'}
    if r.is_success():info.update(extract(m,r,m['graph'].GetSolutionPath(m['source'],m['target'],r)))
    return info


def region_search(data,cfg=Config(),allowed=None,budget_seconds=60.,route_cap=None):
    """Independent Yen search, each path refined by official identical GCS restriction."""
    begin=time.perf_counter();m=compile_graph(data,cfg,allowed);vs=m['vertices'];ids={v.id():i for i,v in enumerate(vs)};adj=[{} for _ in vs];emap={}
    for e in m['edges']:
        i,j=ids[e.u().id()],ids[e.v().id()];a=m['meta'][e.u().id()];b=m['meta'][e.v().id()]
        adj[i][j]=0. if b['kind']=='accept' else float(np.max(abs(a['exit']-b['entry'])/LIMITS));emap[(i,j)]=e
    best=None;attempts=0
    for _,path in yen_paths(adj,ids[m['source'].id()],ids[m['target'].id()]):
        if time.perf_counter()-begin>=budget_seconds:break
        active=[emap[(i,j)] for i,j in zip(path,path[1:])];r=m['graph'].SolveConvexRestriction(active,options(cfg));attempts+=1
        if r.is_success():
            result=extract(m,r,active)
            if best is None or result['objective']<best['objective']:best=result
        if route_cap is not None and attempts>=route_cap:break
    return {'solved':best is not None,'total_seconds':time.perf_counter()-begin,'attempts':attempts,'budget_seconds':budget_seconds,'scope':'Same compiled graph/costs/constraints; independent region Yen ordering; soft budget checked between restrictions.',**(best or {})}
