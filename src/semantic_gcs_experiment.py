"""Task automaton compiled into one GCS, using the original 3D scene.

XY coordinates are lifted by each vertex's fixed posture/object mode. Pick,
place and posture changes expand to checked fixed motion primitives. This is
a finite-library hybrid planner, not general task-and-motion completeness.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, Hyperellipsoid, Point
from pydrake.math import RigidTransform
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver
from pydrake.symbolic import Expression

from .build_scene import build_scene, set_robot_configuration
from .scene_slice_gcs import SliceOracle
from .carry_mode_gcs import CarryOracle
from .compare_carry_planners import retime
from .collision_checker import evaluate_collision_status
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .semantic_monitor import monitor_delivery
from .task_semantics import pick_guard_residuals, place_guard_residuals
from .audit_replay import audit_replay


CHECKPOINT = [-2.5, -1.0]
CASES = ("geometry_only", "delivery", "inspect_then_delivery", "pick_unavailable")


def containing(disks, xy):
    return [i for i,d in enumerate(disks)
            if np.linalg.norm(np.asarray(xy)-d['center']) <= d['radius']-1e-7]


def checkpoint_monitor(rows, checkpoint=CHECKPOINT, tolerance=.02):
    """Observe raw positions before first attachment, without trusting labels."""
    first_pick=next((i for i,r in enumerate(rows) if r['attached']), len(rows))
    visits=[i for i,r in enumerate(rows[:first_pick])
            if np.linalg.norm(np.asarray(r['base_q'][:2])-checkpoint)<=tolerance]
    return {'satisfied':bool(visits), 'first_visit':visits[0] if visits else None,
            'first_attachment':first_pick if first_pick<len(rows) else None,
            'tolerance_m':tolerance}


def contact_checkpoint_monitor(result, checkpoint=CHECKPOINT, tolerance=.02):
    """Check actual base positions before measured bilateral finger contact."""
    threshold=result['acceptance']['minimum_finger_or_bin_contact_force_N']
    rows=result['trace']
    first_contact=next((i for i,r in enumerate(rows)
                        if all(r['finger_contact_force_N'][f]>threshold
                               for f in ('left_finger','right_finger'))),len(rows))
    visits=[i for i,r in enumerate(rows[:first_contact])
            if np.linalg.norm(np.asarray(r['base_q'][:2])-checkpoint)<=tolerance]
    return {'satisfied':bool(visits) and first_contact<len(rows),
            'first_visit_s':rows[visits[0]]['time_s'] if visits else None,
            'first_bilateral_contact_s':rows[first_contact]['time_s'] if first_contact<len(rows) else None,
            'tolerance_m':tolerance,'contact_force_threshold_N':threshold}


def guards(original, config):
    values={}
    for a,b in zip(original, original[1:]):
        if not a['attached'] and b['attached']:
            values['pick']={f'{side}.{k}':v for side,r in [('pre',a),('post',b)]
                            for k,v in pick_guard_residuals(r,config).items()}
        if a['attached'] and not b['attached']:
            values['place']={f'{side}.{k}':v for side,r in [('pre',a),('post',b)]
                             for k,v in place_guard_residuals(r,config).items()}
    assert set(values)=={'pick','place'}
    assert all(np.isfinite(v) and v>=-1e-8 for d in values.values() for v in d.values())
    return values


def prepare(reference, carry_path, slice_path):
    config=json.loads((reference/'scene_config_snapshot.json').read_text())
    rows=json.loads((reference/'delivery_trace.json').read_text())
    graph=json.loads(carry_path.read_text())
    slice_data=json.loads(slice_path.read_text())
    scene=build_scene(config=config,visualize=False)
    first=rows[0]
    oracle=SliceOracle(scene,first['arm_q'],first['gripper_q'],first['base_q'][2])
    lo=np.array(config['robot']['lower_limits'][:2]);hi=np.array(config['robot']['upper_limits'][:2])
    disks=[]
    # Re-evaluate the original 3D geometry at the exact reference posture.
    for xy in [first['base_q'][:2],graph['start'],graph['goal'],CHECKPOINT]+slice_data['centers']:
        xy=np.array(xy)
        if any(np.linalg.norm(xy-d['center'])<1e-7 for d in disks):continue
        clearance,self_distance,_=oracle.query(xy)
        radius=min(.65,clearance-.0201,np.min(xy-lo),np.min(hi-xy))
        if self_distance>=0 and radius>=.035:
            disks.append({'center':xy.tolist(),'radius':float(radius)})
    for xy in [first['base_q'][:2],graph['start'],graph['goal'],CHECKPOINT]:
        assert containing(disks,xy), f'No empty posture coverage at {xy}'
    assert np.allclose([r for r in rows if r['phase']=='navigate_to_pickup'][-1]['base_q'][:2],graph['start'])
    assert monitor_delivery(rows,config)['satisfied']
    return config,rows,{'empty_cover':disks,'carry':graph,'checkpoint':CHECKPOINT,
                        'guards':guards(rows,config)}


def solve_graph(model, start, case):
    """No external region route or task route is selected before SolvePath."""
    begin=time.perf_counter();planner=GcsTrajectoryOptimization(2)
    metadata={};edges=[];groups={};sets={}
    def group(name,disks,kind,mode=None):
        pairs=[(i,j) for i,a in enumerate(disks) for j,b in enumerate(disks)
               if i!=j and np.linalg.norm(np.array(a['center'])-b['center'])<a['radius']+b['radius']-.003]
        g=planner.AddRegions([Hyperellipsoid.MakeHypersphere(d['radius'],np.array(d['center'])) for d in disks],pairs,order=1,h_min=.01,h_max=20,name=name)
        groups[name]=g;sets[name]=disks;edges.extend(g.Edges())
        for i,v in enumerate(g.Vertices()):metadata[v.id()]={'kind':kind,'mode':mode,'disk':i,'layer':name}
        return g
    def point(name,xy,kind,**kw):
        g=planner.AddRegions([Point(np.array(xy))],order=0,name=name)
        metadata[g.Vertices()[0].id()]={'kind':kind,**kw}
        return g
    def into(name,p,xy):
        edges.extend(planner.AddEdges(groups[name],p,edges_between_regions=[(i,0) for i in containing(sets[name],xy)]).Edges())
    def outof(p,name,xy):
        edges.extend(planner.AddEdges(p,groups[name],edges_between_regions=[(0,i) for i in containing(sets[name],xy)]).Edges())
    group('empty',model['empty_cover'],'empty')
    pickup_layer='empty'
    if case=='inspect_then_delivery':
        group('inspected_empty',model['empty_cover'],'empty')
        p=point('inspect',CHECKPOINT,'inspect');into('empty',p,CHECKPOINT);outof(p,'inspected_empty',CHECKPOINT)
        pickup_layer='inspected_empty'
    graph=model['carry'];postures=np.array(graph['postures'])
    for m,disks in enumerate(graph['covers']):group(f'carry{m}',disks,'carry',m)
    for i,t in enumerate(graph['transitions']):
        p=point(f'switch{i}',t['xy'],'switch',transition=i)
        p.AddVertexCost(Expression(.1+.15*np.linalg.norm(postures[t['to']]-postures[t['from']])))
        into(f"carry{t['from']}",p,t['xy']);outof(p,f"carry{t['to']}",t['xy'])
    pick_allowed=case!='pick_unavailable' and all(v>=-1e-8 for v in model['guards']['pick'].values())
    if pick_allowed:
        p=point('pick_and_lift',graph['start'],'pick');p.AddVertexCost(Expression(.1))
        into(pickup_layer,p,graph['start']);outof(p,'carry0',graph['start'])
    place_allowed=all(v>=-1e-8 for v in model['guards']['place'].values())
    src=point('start',start,'start');dst=point('accept',graph['goal'],'accept')
    outof(src,'empty',start)
    if place_allowed:
        p=point('place_and_retreat',graph['goal'],'place');p.AddVertexCost(Expression(.1))
        into(f'carry{len(postures)-1}',p,graph['goal'])
        edges.extend(planner.AddEdges(p,dst,edges_between_regions=[(0,0)]).Edges())
    if case=='geometry_only':into('empty',dst,graph['goal'])
    # Exact directed reachability of the compiled finite graph, not world infeasibility.
    adjacency={}
    for e in edges:adjacency.setdefault(e.u().id(),[]).append(e.v().id())
    seen={src.Vertices()[0].id()};todo=list(seen)
    for v in todo:
        for w in adjacency.get(v,[]):
            if w not in seen:seen.add(w);todo.append(w)
    info={'case':case,'vertices':len(metadata),'edges':len(edges),'pick_edge_present':pick_allowed,
          'inspection_required':case=='inspect_then_delivery','graph_build_seconds':time.perf_counter()-begin}
    if dst.Vertices()[0].id() not in seen:
        return {**info,'status':'unreachable_in_compiled_graph','solve_seconds':0.,'actions':[],
                'reason':'No accepting task path with the available action library; not a proof of physical infeasibility.'},None
    planner.AddPathLengthCost(1.);planner.AddTimeCost(.01)
    planner.AddVelocityBounds(np.array([-.3,-.3]),np.array([.3,.3]))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10;options.rounding_seed=0
    options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
    begin=time.perf_counter();_,result=planner.SolvePath(src,dst,options)
    info.update(solve_seconds=time.perf_counter()-begin,solver_status=str(result.get_solution_result()))
    if not result.is_success():return {**info,'status':'solver_failed'},None
    outgoing={}
    for e in edges:
        if result.GetSolution(e.phi())>.5:outgoing.setdefault(e.u().id(),[]).append(e)
    v=src.Vertices()[0];path=[];seen=set()
    while True:
        assert v.id() not in seen;seen.add(v.id())
        path.append({**metadata[v.id()],'x':result.GetSolution(v.x()).tolist()})
        if v.id()==dst.Vertices()[0].id():break
        assert len(outgoing.get(v.id(),[]))==1
        v=outgoing[v.id()][0].v()
    actions=[v['kind'] for v in path if v['kind'] in ('inspect','pick','place')]
    return {**info,'status':'solved','actions':actions,'objective':float(result.get_optimal_cost()),'selected_vertices':path},path


def expand_and_audit(path,model,original,config,output):
    output.mkdir(parents=True,exist_ok=True)
    anchor=[r for r in original if r['phase']=='lift_cup'][-1]
    oracle=CarryOracle(config,anchor);rows=[];empty_points=[];carry_points=[]
    graph=model['carry'];postures=np.array(graph['postures'])
    first=original[0]
    for v in path:
        kind=v['kind']
        if kind in ('empty','carry'):
            x=np.array(v['x']);assert len(x)==5
            a,b=x[:2],x[2:4]
            d=(model['empty_cover'] if kind=='empty' else graph['covers'][v['mode']])[v['disk']]
            assert max(np.linalg.norm(a-d['center']),np.linalg.norm(b-d['center']))<=d['radius']+1e-5
            target=empty_points if kind=='empty' else carry_points
            posture=np.r_[first['base_q'][2],first['arm_q']] if kind=='empty' else postures[v['mode']]
            target.extend([np.r_[a,posture],np.r_[b,posture]])
        elif kind=='switch':
            t=graph['transitions'][v['transition']]
            for u in np.linspace(0,1,t['audit']['samples']):
                carry_points.append(np.r_[t['xy'],(1-u)*postures[t['from']]+u*postures[t['to']]])
    def generated(points,attached):
        qs,_=retime(points);out=[]
        for q in qs:
            row=copy.deepcopy(anchor if attached else first)
            row.update(base_q=q[:3].tolist(),arm_q=q[3:].tolist(),origin='joint_semantic_gcs',
                       phase='whole_body_carry_to_dropoff' if attached else 'navigate_to_pickup')
            if attached:
                gripper,cup=oracle.state(q[:2],q[2:]);row['cup_pose']=cup.GetAsMatrix4().tolist()
            out.append(row)
        return out
    rows.extend(generated(empty_points,False))
    actions=[v['kind'] for v in path]
    if 'pick' in actions:
        rows.extend(copy.deepcopy(r) for r in original if r['phase'] in ('arm_to_pregrasp','approach_cup','close_gripper','lift_cup'))
        rows.extend(generated(carry_points,True)[1:])
    if 'place' in actions:
        rows.extend(copy.deepcopy(r) for r in original if r['phase'] in ('lower_into_bin','release_cup','retreat_arm'))
    scene=build_scene(config=config,visualize=True);scene.meshcat.StartRecording()
    for i,r in enumerate(rows):
        set_robot_configuration(scene,r['base_q'],r['arm_q'],r['gripper_q'])
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(r['cup_pose'])))
        r['gripper_pose']=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).GetAsMatrix4().tolist()
        r['collision_residual']=evaluate_collision_status(scene,r['base_q'],arm_q=r['arm_q'],gripper_q=r['gripper_q']).residual
        r['cup_gripper_audit']=measure_cup_gripper(scene);r['index']=i
        scene.root_context.SetTime(.15*i);scene.diagram.ForcedPublish(scene.root_context)
    scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
    semantic=monitor_delivery(rows,config);dense=audit_replay(rows,config,10);grasp=summarize_grasp_audit(rows)
    inspection=checkpoint_monitor(rows)
    base=np.array([r['base_q'][:2] for r in rows])
    audit={'delivery':semantic,'checkpoint_before_pick':inspection,'dense':dense,'grasp':grasp,
           'base_path_length_m':float(np.linalg.norm(np.diff(base,axis=0),axis=1).sum()),'frames':len(rows)}
    assert dense['satisfied'] and grasp['satisfied'],audit
    for name,data in [('delivery_trace.json',rows),('scene_config_snapshot.json',config),('audit.json',audit)]:
        (output/name).write_text(json.dumps(data))
    (output/'scene.html').write_text(scene.meshcat.StaticHtml())
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('outputs/results/semantic_joint_gcs'))
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    reference=Path('outputs/results/full_dynamics/reference')
    config,rows,model=prepare(reference,Path('outputs/results/paired_carry_comparison/disk_run0/graph.json'),Path('outputs/results/grasp_fix_web/planar/slice/result.json'))
    serialized=json.dumps(model,sort_keys=True);digest=hashlib.sha256(serialized.encode()).hexdigest()
    (args.output/'shared_model.json').write_text(serialized)
    results={}
    for case in CASES:
        print('SOLVING '+case,flush=True)
        result,path=solve_graph(model,rows[0]['base_q'][:2],case)
        result['shared_model_sha256']=digest
        if path is not None:result['audit']=expand_and_audit(path,model,rows,config,args.output/case)
        results[case]=result
        (args.output/'results.json').write_text(json.dumps(results,indent=2))
        print(json.dumps({k:v for k,v in result.items() if k not in ('audit','selected_vertices')}),flush=True)
    assert results['geometry_only']['status']=='solved' and not results['geometry_only']['audit']['delivery']['satisfied']
    for case in ('delivery','inspect_then_delivery'):assert results[case]['audit']['delivery']['satisfied']
    assert results['inspect_then_delivery']['audit']['checkpoint_before_pick']['satisfied']
    assert not results['delivery']['audit']['checkpoint_before_pick']['satisfied']
    assert results['pick_unavailable']['status']=='unreachable_in_compiled_graph'
    print('ALL EXPERIMENT EXPECTATIONS PASSED',flush=True)


if __name__=='__main__':main()
