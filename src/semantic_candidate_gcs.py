"""One GCS chooses a checked grasp candidate and continuous paths on both sides.

Carrying vertices hold (q_entry,q_exit) in C x C, q=(x,y,yaw,arm[7]).
Empty vertices eliminate fixed posture coordinates and optimize XY endpoints;
edge continuity and exported paths lift them back to the same full 10D state.
Action vertices are fixed checked primitives, not teleportation edges. Each
grasp has its own object transform and carrying cover; branches cannot mix.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry.optimization import CartesianProduct, GraphOfConvexSets, GraphOfConvexSetsOptions, HPolyhedron, Hyperellipsoid, Point
from pydrake.solvers import Binding, Cost, L2NormCost, ClarabelSolver
from pydrake.math import RigidTransform

from .build_scene import build_scene, set_robot_configuration
from .carry_mode_gcs import CarryOracle
from .compare_carry_planners import retime
from .collision_checker import evaluate_collision_status
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .semantic_monitor import monitor_delivery
from .audit_replay import audit_replay


WEIGHTS=np.array([1.,1.,.35]+[.12]*7)
PICK_PHASES=('arm_to_pregrasp','approach_cup','close_gripper','lift_cup')
PLACE_PHASES=('lower_into_bin','release_cup','retreat_arm')


def q(row):return np.array([*row['base_q'],*row['arm_q']])


def primitive_cost(rows):
    return float(np.linalg.norm(np.diff(np.array([q(r) for r in rows]),axis=0)*WEIGHTS,axis=1).sum())


def read_library(root):
    config=json.loads((root/'config.json').read_text())
    report=json.loads((root/'report.json').read_text())
    candidates={}
    for attempt in report['attempts']:
        if attempt['status']!='accepted':continue
        name=attempt['name'];directory=root/name
        candidates[name]={key:json.loads((directory/file).read_text()) for key,file in
                          [('rows','delivery_trace.json'),('cover','cover.json'),('guards','guards.json')]}
    return config,candidates


def build_and_solve(disks,candidates,allowed,config,*,start_base_q=None,rounding_seed=0,max_rounded_paths=10,goal_options=None):
    begin=time.perf_counter();g=GraphOfConvexSets();metadata={};edges=[]
    first=next(iter(candidates.values()))['rows'][0];initial=q(first)
    if start_base_q is not None:
        initial[:3]=np.asarray(start_base_q,dtype=float)
    weight_matrix=np.diag(WEIGHTS)
    def vertex(region,name,meta,cost=None):
        v=g.AddVertex(region,name);metadata[v.id()]=meta
        if cost is not None:v.AddCost(float(cost)+1e-6)
        return v
    def motion(region,name,meta):
        v=vertex(CartesianProduct([region,region]),name,meta)
        n=region.ambient_dimension();weights=weight_matrix[:n,:n]
        v.AddCost(Binding[Cost](L2NormCost(np.c_[-weights,weights],np.zeros(n)),v.x()))
        v.AddCost(1e-6)
        return v
    def edge(a,b,continuity=True):
        e=g.AddEdge(a,b)
        if continuity:
            empty_a=metadata[a.id()]['kind']=='empty';empty_b=metadata[b.id()]['kind']=='empty'
            end=np.r_[e.xu()[2:],initial[2:]] if empty_a else e.xu()[10:]
            start=np.r_[e.xv()[:2],initial[2:]] if empty_b else e.xv()[:10]
            for i in range(2 if empty_a and empty_b else 10):e.AddConstraint(end[i]==start[i])
        edges.append(e)
    source=vertex(Point(np.r_[initial,initial]),'start',{'kind':'start'})
    # Accept has no physical state; each incoming place primitive has its own
    # fixed, checked final configuration at the common dropoff base.
    target=vertex(Point(np.zeros(20)),'accept',{'kind':'accept'})
    nav=[]
    for i,d in enumerate(disks):
        c=Hyperellipsoid.MakeHypersphere(d['radius'],np.array(d['center']))
        nav.append(motion(c,f'empty_{i}',{'kind':'empty','disk':i}))
        if np.linalg.norm(initial[:2]-d['center'])<=d['radius']-1e-7:edge(source,nav[-1])
    for i,a in enumerate(disks):
        for j,b in enumerate(disks):
            if i!=j and np.linalg.norm(np.array(a['center'])-b['center'])<a['radius']+b['radius']-.003:edge(nav[i],nav[j])
    if 'inspection_sites' in config:
        inspected=[]
        for i,d in enumerate(disks):
            c=Hyperellipsoid.MakeHypersphere(d['radius'],np.array(d['center']))
            inspected.append(motion(c,f'inspected_empty_{i}',{'kind':'empty','disk':i,'task_mode':'inspected'}))
        for i,a in enumerate(disks):
            for j,b in enumerate(disks):
                if i!=j and np.linalg.norm(np.array(a['center'])-b['center'])<a['radius']+b['radius']-.003:edge(inspected[i],inspected[j])
        for site in config['inspection_sites']:
            state=initial.copy();state[:2]=site['xy']
            action=vertex(Point(np.r_[state,state]),'inspect_'+site['name'],{'kind':'inspect','site':site['name']},0.)
            for i,d in enumerate(disks):
                if np.linalg.norm(state[:2]-d['center'])<=d['radius']-1e-7:
                    edge(nav[i],action);edge(action,inspected[i])
        nav=inspected
    expected_goal=np.array(config['delivery_task']['transfer_path'][-1])
    for name,candidate in candidates.items():
        rows=candidate['rows'];pick=[r for r in rows if r['phase'] in PICK_PHASES];place=[r for r in rows if r['phase'] in PLACE_PHASES]
        gate=candidate['guards']
        if name not in allowed or not all(np.isfinite(v) and v>=-1e-8 for d in gate.values() for v in d.values()):continue
        assert np.allclose(q(pick[0])[2:],initial[2:],atol=1e-7)
        options_for_grasp=([('default',candidate)] if goal_options is None else list(goal_options.get(name,{}).items()))
        options_for_grasp=[(key,value) for key,value in options_for_grasp if all(np.isfinite(v) and v>=-1e-8 for d in value['guards'].values() for v in d.values())]
        if not options_for_grasp:continue
        if goal_options is None:assert np.allclose(place[-1]['base_q'],expected_goal,atol=1e-7)
        pv=vertex(Point(np.r_[q(pick[0]),q(pick[-1])]),f'pick_{name}',{'kind':'pick','candidate':name},primitive_cost(pick))
        placements=[]
        for goal_id,goal_candidate in options_for_grasp:
            action=[r for r in goal_candidate['rows'] if r['phase'] in PLACE_PHASES]
            meta={'kind':'place','candidate':name}
            if goal_options is not None:meta['goal']=goal_id
            label=f'place_{name}' if goal_options is None else f'place_{name}_{goal_id}'
            lv=vertex(Point(np.r_[q(action[0]),q(action[-1])]),label,meta,primitive_cost(action))
            placements.append((action,lv))
        for i,d in enumerate(disks):
            if np.linalg.norm(q(pick[0])[:2]-d['center'])<=d['radius']-1e-7:edge(nav[i],pv)
        regions=[HPolyhedron(np.array(r['A']),np.array(r['b'])) for r in candidate['cover']['regions']]
        cv=[motion(c,f'carry_{name}_{i}',{'kind':'carry','candidate':name,'region':i}) for i,c in enumerate(regions)]
        for i,c in enumerate(regions):
            if c.PointInSet(q(pick[-1]),1e-7):edge(pv,cv[i])
            for action,lv in placements:
                if c.PointInSet(q(action[0]),1e-7):edge(cv[i],lv)
            for j in range(i):
                if c.IntersectsWith(regions[j]):edge(cv[i],cv[j]);edge(cv[j],cv[i])
        for action,lv in placements:edge(lv,target,False)
    info={'allowed_candidates':list(allowed),'vertices':len(metadata),'edges':len(edges),'build_seconds':time.perf_counter()-begin}
    adjacency={}
    for e in edges:adjacency.setdefault(e.u().id(),[]).append(e.v().id())
    reachable={source.id()};queue=[source.id()]
    for v in queue:
        for w in adjacency.get(v,[]):
            if w not in reachable:reachable.add(w);queue.append(w)
    if target.id() not in reachable:return {**info,'status':'unreachable_in_compiled_graph','solve_seconds':0},None
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=max_rounded_paths;options.rounding_seed=rounding_seed
    options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
    begin=time.perf_counter();result=g.SolveShortestPath(source,target,options)
    info.update(solve_seconds=time.perf_counter()-begin,solver_status=str(result.get_solution_result()))
    if not result.is_success():return {**info,'status':'solver_failed'},None
    selected=g.GetSolutionPath(source,target,result)
    assert all(result.GetSolution(e.phi())>.999 for e in selected)
    vertices=[source]+[e.v() for e in selected]
    path=[]
    for v in vertices:
        values=result.GetSolution(v.x())
        if metadata[v.id()]['kind']=='empty':values=np.r_[values[:2],initial[2:],values[2:],initial[2:]]
        path.append({**metadata[v.id()],'x':values.tolist()})
    actions=[v for v in path if v['kind'] in ('pick','place')]
    assert [v['kind'] for v in actions]==['pick','place']
    assert actions[0]['candidate']==actions[1]['candidate']
    costs={}
    for v in vertices:
        kind=metadata[v.id()]['kind'];costs[kind]=costs.get(kind,0.)+float(v.GetSolutionCost(result))
    info.update(status='solved',selected_candidate=actions[0]['candidate'],objective=float(result.get_optimal_cost()),cost_by_kind=costs,path=path)
    if goal_options is not None:info['selected_goal']=actions[1]['goal']
    info['inspection_site']=next((v['site'] for v in path if v['kind']=='inspect'),None)
    return info,path


def expand(path,candidates,disks,config,output,*,visualize=True):
    selected=next(v['candidate'] for v in path if v['kind']=='pick')
    candidate=candidates[selected];original=candidate['rows']
    pick=[r for r in original if r['phase'] in PICK_PHASES];place=[r for r in original if r['phase'] in PLACE_PHASES]
    oracle=CarryOracle(config,pick[-1]);rows=[]
    # Check every normal edge in the extracted integral path and every convex
    # segment endpoint. The action's internal curve is expanded below.
    for a,b in zip(path,path[1:]):
        if b['kind']!='accept':assert np.max(np.abs(np.array(a['x'][10:])-b['x'][:10]))<2e-5
    for kind in ('empty','carry'):
        points=[]
        for v in path:
            if v['kind']!=kind:continue
            a,b=np.array(v['x'][:10]),np.array(v['x'][10:])
            if kind=='empty':
                d=disks[v['disk']]
                assert max(np.linalg.norm(a[:2]-d['center']),np.linalg.norm(b[:2]-d['center']))<=d['radius']+1e-5
            else:
                assert v['candidate']==selected
                spec=candidate['cover']['regions'][v['region']];A=np.array(spec['A']);bb=np.array(spec['b'])
                assert max(np.max(A@a-bb),np.max(A@b-bb))<1e-5
            points.extend([a,b])
        qs,timing=retime(points)
        for point in qs:
            row=copy.deepcopy(original[0] if kind=='empty' else pick[-1])
            row.update(base_q=point[:3].tolist(),arm_q=point[3:].tolist(),origin='joint_candidate_gcs',phase='navigate_to_pickup' if kind=='empty' else 'whole_body_carry_to_dropoff')
            if kind=='carry':
                _,cup=oracle.state(point[:2],point[2:]);row['cup_pose']=cup.GetAsMatrix4().tolist()
            rows.append(row)
        if kind=='empty':rows.extend(copy.deepcopy(pick))
    rows.extend(copy.deepcopy(place))
    output.mkdir(parents=True,exist_ok=True)
    scene=build_scene(config=config,visualize=visualize)
    if visualize:scene.meshcat.StartRecording()
    for i,r in enumerate(rows):
        set_robot_configuration(scene,r['base_q'],r['arm_q'],r['gripper_q'])
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(r['cup_pose'])))
        r['gripper_pose']=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).GetAsMatrix4().tolist()
        r['collision_residual']=evaluate_collision_status(scene,r['base_q'],arm_q=r['arm_q'],gripper_q=r['gripper_q']).residual
        r['cup_gripper_audit']=measure_cup_gripper(scene);r['index']=i
        if visualize:
            scene.root_context.SetTime(.15*i);scene.diagram.ForcedPublish(scene.root_context)
    if visualize:
        scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
    semantic=monitor_delivery(rows,config);grasp=summarize_grasp_audit(rows);dense=audit_replay(rows,config,10)
    base=np.array([r['base_q'][:2] for r in rows])
    audit={'semantic':semantic,'grasp':grasp,'dense':dense,'frames':len(rows),'base_path_length_m':float(np.linalg.norm(np.diff(base,axis=0),axis=1).sum()),
           'weighted_path_cost':float(np.linalg.norm(np.diff(np.array([q(r) for r in rows]),axis=0)*WEIGHTS,axis=1).sum())}
    audit['accepted']=semantic['satisfied'] and grasp['satisfied'] and dense['satisfied']
    if 'inspection_sites' in config:
        audit['inspection']=inspection_monitor(rows,config['inspection_sites'])
        audit['accepted']=audit['accepted'] and audit['inspection']['satisfied']
    for name,value in [('delivery_trace.json',rows),('scene_config_snapshot.json',config),('audit.json',audit)]:
        (output/name).write_text(json.dumps(value))
    if visualize:(output/'scene.html').write_text(scene.meshcat.StaticHtml())
    return audit


def inspection_monitor(rows,sites,tolerance=.02):
    """Observe pre-grasp reference segments; do not trust task-node labels."""
    from .inspection_polyline_audit import inspect_polyline
    return inspect_polyline(rows,sites,tolerance)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('outputs/results/semantic_candidates'))
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    config,candidates=read_library(root/'library')
    model=json.loads(Path('outputs/results/semantic_joint_gcs/shared_model.json').read_text());disks=model['empty_cover']
    # Hash excludes policy so both solves can be checked against the same library.
    digest=hashlib.sha256(json.dumps({'disks':disks,'candidates':candidates},sort_keys=True).encode()).hexdigest()
    results={}
    for case in ('all_candidates','selected_site_unavailable','none_available'):
        allowed=list(candidates)
        if case=='selected_site_unavailable':allowed.remove(results['all_candidates']['selected_candidate'])
        if case=='none_available':allowed=[]
        print('SOLVE '+case+' '+str(allowed),flush=True)
        result,path=build_and_solve(disks,candidates,allowed,config)
        result['shared_library_sha256']=digest
        if path is not None:result['audit']=expand(path,candidates,disks,config,root/case)
        results[case]=result;(root/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('path','audit')}),flush=True)
        if path is not None:assert result['audit']['accepted'],result['audit']
    assert results['all_candidates']['selected_candidate']!=results['selected_site_unavailable']['selected_candidate']
    assert results['none_available']['status']=='unreachable_in_compiled_graph'


if __name__=='__main__':main()
