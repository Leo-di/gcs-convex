"""Matched early / geometry / post-hoc pipelines on the functional cafe.

All methods use the same cost-ordered posture family and finite budgets.
Post-hoc checks occur after a complete action reference and before acceptance;
rejected references are never physically executed.
"""
import argparse,copy,hashlib,json,time
from functools import partial
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform
from . import stateful_transport_grounding as core
from .build_scene import build_scene,set_robot_configuration
from .human_passage import HumanPassageEvaluator
from .semantic_plan import configuration,SemanticPlanEvaluator
from .transport_posture_search import PostureSearchRejected
from .scene_spec import PROJECT_ROOT,load_scene_config

def ordered_postures(candidates,screen,evaluate,policy,artifact):
    """Same candidate order for all methods; coarse covers never delete a trial."""
    began=time.monotonic();rows=[];attempts=0
    report=dict(method='matched_cost_order_posture_search',policy=policy,generated_candidates=candidates,
        trials=rows,selected=None,scope='Fixed candidate order, no coarse deletion; full-plans and admission-time budgets shared by all methods')
    def save():
        report.update(full_plans=attempts,wall_seconds=time.monotonic()-began)
        Path(artifact).write_text(json.dumps(report,indent=2))
    for c in candidates:
        if attempts>=policy['max_full_plans'] or time.monotonic()-began>=policy['time_budget_s']:break
        row=dict(id=c['id'],offset_world_m=c['offset_world_m'],full_plan_attempted=False);rows.append(row)
        # Keep the same full-trial ordering independently of screen outcomes.
        try:row['screen']=screen(c)
        except Exception as exc:row['screen_error']=str(exc)
        if time.monotonic()-began>=policy['time_budget_s']:row['status']='admission_budget_exhausted';save();break
        attempts+=1;row['full_plan_attempted']=True;save()
        try:
            answer=evaluate(c)
            if answer is None:raise RuntimeError('No accepted action returned')
        except Exception as exc:
            row.update(status='full_action_rejected',failure_stage=getattr(exc,'stage',type(exc).__name__),
                       reason=str(exc),details=getattr(exc,'details',None));save();continue
        row['status']='fully_accepted';report['selected']=c;save();return answer,report
    save();raise PostureSearchRejected(report)

def reference_passage_audit(scene,frames,*,stop_on_failure=False):
    """Same interpolated configurations as the full-action auditor; both routes.

Held object poses follow gripper FK, free objects retain the propagated world.
Finite checks, not a continuous-time certificate.
"""
    human=HumanPassageEvaluator(scene,force=True);transforms={};samples=0;bad=0;examples=[];counts={}
    began=time.monotonic()
    for i,f in enumerate(frames):
        a=configuration(frames[max(0,i-1)]);b=configuration(f)
        held=f['mode'] in ('lift','hold','carry','lower')
        if held and f['object_id'] not in transforms:
            set_robot_configuration(scene,f['base_q'],f['arm_q'],f['gripper_q'])
            transforms[f['object_id']]=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@RigidTransform(np.asarray(f['world_objects'][f['object_id']]))
        n=max(1,int(np.ceil(np.max(abs(b-a)/np.r_[[.01]*10,[.001]*2]))))
        for alpha in np.linspace(0,1,n+1):
            q=(1-alpha)*a+alpha*b;set_robot_configuration(scene,q[:3],q[3:10],q[10:])
            for key,matrix in f['world_objects'].items():
                pose=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body)@transforms[key] if held and key==f['object_id'] else RigidTransform(np.asarray(matrix))
                scene.plant.SetFreeBodyPose(scene.plant_context,scene.object_bodies[key],pose)
            r=human.current();routes=r.get('connections',{human.spec['id']:r});samples+=1
            for key,v in routes.items():counts[key]=counts.get(key,0)+int(not v['satisfied'])
            if not r['satisfied']:
                bad+=1
                if len(examples)<5:examples.append(dict(phase=f['phase'],edge=i,alpha=float(alpha),
                    failed_connections=[k for k,v in routes.items() if not v['satisfied']],base_q=q[:3].tolist()))
                if stop_on_failure:break
        if stop_on_failure and bad:break
    return dict(satisfied=bad==0,samples=samples,violations=bad,connection_violations=counts,
        counterexamples=examples,complete=not(stop_on_failure and bad),wall_seconds=time.monotonic()-began,
        scope='Interpolated reference nodes/edges; all world objects; same capsule/body-AABB model; no continuous-time certificate')

def run(task_path,output,method,*,execute=True):
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    spec=load_scene_config(task_path);cfg=core.make_config(spec)
    base_method={'branch_early':'early','branch_posthoc':'posthoc','branch_geometry':'geometry'}.get(method,method)
    if method.startswith('branch_'):
        spec=copy.deepcopy(spec);spec['carry']['graph_mode']='multibranch'
    planning=copy.deepcopy(cfg)
    if base_method in ('geometry','posthoc'):planning['human_passage']['enforce']=False
    if base_method=='geometry':cfg['human_passage']['enforce']=False
    protocol=dict(spec=spec,config=cfg,planning_config=planning,method=method,
        rule_check_timing={'early':'candidate/stage planning and reference/execution','early_suffix':'posture-independent destination suffix before transport search; then all early checks','geometry':'audit only; not enforced','posthoc':'complete action reference, then accept/reject/retry; execution enforced'}[base_method],
        source_sha256={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((PROJECT_ROOT/'src').glob('*.py'))})
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));checks=[]
    def posthoc(answer,world,action,folder):
        frames,_,report=answer
        current=core.world_config(cfg,world,action['object_id']);scene=build_scene(config=current,visualize=False)
        audit=reference_passage_audit(scene,frames,stop_on_failure=True)
        (folder/'posthoc_passage.json').write_text(json.dumps(audit,indent=2))
        row=dict(action=action['id'],folder=str(folder.relative_to(root)),**audit);checks.append(row)
        (root/'posthoc_checks.json').write_text(json.dumps(checks,indent=2))
        if not audit['satisfied']:raise core.TransportRejected('posthoc_passage_rejected',row)
        report['posthoc_passage']=audit
    from .destination_suffix import check_destination_suffix
    ground=partial(core.ground_action,pickup_cache={},posthoc_accept=posthoc if base_method=='posthoc' else None,posture_selector=ordered_postures,
        destination_precheck=check_destination_suffix if method in ('early_suffix','branch_early') else None)
    began=time.monotonic();search=core.search(planning,spec,root,ground=ground)
    (root/'search.json').write_text(json.dumps({k:v for k,v in search.items() if k!='frames'},indent=2))
    result=dict(satisfied=False,reference_satisfied=search['satisfied'],method=method)
    if search['satisfied']:
        # Propagate the actually selected order to all downstream adapters.
        by_id={a['id']:a for a in spec['actions']}
        spec['actions']=[by_id[a] for a in search['selected_task_order']]
        protocol['spec']=spec
        (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
        frames=search['frames'];scene=build_scene(config=cfg,visualize=False)
        audit=reference_passage_audit(scene,frames)
        (root/'reference_passage.json').write_text(json.dumps(audit,indent=2))
        if base_method!='geometry' and not audit['satisfied']:
            result.update(reason='assembled_reference_passage_rejected',details=audit)
        else:
            plan=core.semantic_plan(cfg,frames)
            accepted=SemanticPlanEvaluator(build_scene(config=cfg,visualize=False),plan).audit(frames)
            (root/'assembled_semantic.json').write_text(json.dumps(accepted,indent=2))
            if not accepted['satisfied']:raise core.TransportRejected('assembled_semantic_plan_failed',accepted)
            for name,data in [('reference',frames),('plan',plan),('config',cfg)]:
                (root/(name+'.json')).write_text(json.dumps(data,indent=2 if name!='reference' else None))
            if not execute:
                result.update(satisfied=True,reference_satisfied=True,execution_performed=False,scope='Cold generated guide only; complete-task joint optimization and execution are separate, still required.')
                result['total_wall_seconds']=time.monotonic()-began
                (root/'result.json').write_text(json.dumps(result,indent=2))
                return result
            try:
                result,physics,_=core.execute_reference(cfg,spec,frames,plan,search['selected'],visualize=False)
                result.update(reference_satisfied=True,method=method)
                (root/'physics.json').write_text(json.dumps(physics))
            except Exception as exc:result.update(reason=str(exc),details=getattr(exc,'details',{}))
    result['total_wall_seconds']=time.monotonic()-began
    (root/'result.json').write_text(json.dumps(result,indent=2));print({k:result.get(k) for k in ('satisfied','reference_satisfied','reason')},flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--method',required=True,choices=['early','early_suffix','geometry','posthoc','branch_early','branch_posthoc','branch_geometry']);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    if not run(a.config,a.output,a.method,execute=not a.plan_only)['satisfied']:raise SystemExit(1)
