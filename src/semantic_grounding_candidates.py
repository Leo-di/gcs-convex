"""Fixed candidate grounding: incoming motion bridge and contained manipulation.

Uses the existing official Drake GCS / IRIS pipeline. Candidate-level
selection is explicitly an outer finite search, not joint GCS optimization.
"""
import argparse,copy,json,os,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np


def worker(root,index):
    from .build_scene import build_scene,set_robot_configuration
    from .scene_spec import load_scene_config
    from .semantic_refinement import make_candidate_config
    from .geometry_outbound_gcs import plan_geometry_outbound
    from .green_room_gcs import room_oracle,guide_edge_validator,solve_operation,append_motion,audit_reference,region_holdout
    from .green_room_iris import make_checker
    from .grasp_planner import plan_pick,cartesian_arm_path
    from .safe_pregrasp import plan_pregrasp
    root=Path(root);protocol=json.loads((root/'protocol.json').read_text());candidate=protocol['candidates'][index]
    out=root/candidate['id'];out.mkdir(exist_ok=True)
    config=make_candidate_config(load_scene_config(protocol['config']),candidate)
    from .semantic_policy import policy_from_config,accepts
    policy=policy_from_config(config);semantic=policy=='full_motion'
    scene=build_scene(config=config,visualize=False)
    base=np.array(candidate['base_q']);start=np.array(config['robot']['start']);tucked=np.array(config['pick_task']['navigation_arm_q']);oracle=room_oracle(scene);spec=config['green_room']
    row={'id':candidate['id'],'semantic_policy':policy,'base_q':base.tolist(),'semantic_goal':oracle.query(np.r_[base,tucked]),'bridge':{'accepted':False},'operation':{'accepted':False},'physical':{'accepted':False}}
    began=time.perf_counter()
    def save(stage):
        row['stage']=stage;row['elapsed_seconds']=time.perf_counter()-began
        (out/'candidate_result.json').write_text(json.dumps(row,indent=2))
    (out/'scene_config_snapshot.json').write_text(json.dumps(config,indent=2));save('bridge')
    if config.get('grounding_semantic_plan') is not None:
        from .semantic_plan import SemanticPlanEvaluator
        try:
            SemanticPlanEvaluator(scene, config['grounding_semantic_plan'], bindings={'dock': base, 'tucked': tucked})
        except Exception as e:
            row.update(failure_stage='semantic_plan_validation',failure_reason=str(e));save('complete');return
    cached_pick=None;cached_goals=None
    if protocol.get('endpoint_first',False):
        from .joint_grounding_checks import check_pick_endpoints
        save('joint_endpoint_check');precheck_start=time.perf_counter()
        try:
            cached_pick,cached_goals=check_pick_endpoints(scene,base,oracle,enforce_semantic=policy!='geometry_only')
            row['joint_endpoint_check']={'accepted':True,'seconds':time.perf_counter()-precheck_start,'goals':cached_goals}
        except Exception as e:
            row['joint_endpoint_check']={'accepted':False,'seconds':time.perf_counter()-precheck_start,'reason':str(e)}
            row.update(failure_stage='joint_endpoint_check',failure_reason=str(e));save('complete');return
    from .accepted_motion import accept_cup_phase, compose_cup_motion
    repeat=lambda q,n:np.repeat(np.asarray(q)[None,:],n,axis=0)
    nav=None;nav_segment=None
    def phase_segment(name, bases, arms, grips, attachment=None, dwell=0):
        local=[]
        append_motion(local,bases,arms,grips,name)
        local.extend([dict(local[-1]) for _ in range(dwell)])
        return accept_cup_phase(scene,local,base,tucked,attachment,
            start=np.r_[bases[0],arms[0],grips[0]], end=np.r_[bases[-1],arms[-1],grips[-1]])
    def run_navigation():
        nonlocal nav,nav_segment
        # Region holdout probes may leave the planning cup at a lifted pose.
        # Restore navigation collision filters in the planning context.
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,scene.cup_initial_pose)
        save('bridge')
        try:
            tick=time.perf_counter()
            if 'bridge_method' in protocol:
                from .automatic_motion_bridge import plan_bridge
                nav,navreport=plan_bridge(scene,start,base,tucked,protocol['bridge_method'],out/'bridge_events.json',seed=protocol.get('seed',20261019))
            else:
                nav,navreport=plan_geometry_outbound(scene,start,base,allowed_region_names=spec['navigation_regions'],arm_q=tucked,spacing=spec['navigation_spacing_m'],prune_contained=spec['navigation_prune_contained'],refinement_spacing=spec['navigation_refinement_spacing_m'],refinement_radius_threshold=spec['navigation_refinement_radius_threshold_m'])
            indices=sorted(set([*range(0,len(nav.configurations),10),len(nav.configurations)-1]))
            navpath=nav.configurations[indices]
            nav_segment=phase_segment('navigate_to_pickup',navpath,repeat(tucked,len(navpath)),repeat(scene.gripper_open_positions,len(navpath)))
            row['bridge']={'phase_acceptance':nav_segment.report,'accepted':True,'seconds':time.perf_counter()-tick,'report':navreport,'path_length_m':float(np.linalg.norm(np.diff(nav.configurations[:,:2],axis=0),axis=1).sum())}
        except Exception as e:row['bridge']={'accepted':False,'seconds':time.perf_counter()-tick,'reason':str(e)}
    if not protocol.get('operation_first',False):run_navigation()
    save('operation_ik')
    try:
        tick=time.perf_counter()
        if cached_pick is None:
            from .joint_grounding_checks import check_pick_endpoints
            pick,goals=check_pick_endpoints(scene,base,oracle,enforce_semantic=policy!='geometry_only')
        else:pick,goals=cached_pick,cached_goals
        row['operation']['goal_checks']=goals
        row['operation']['sampled_grasp_pose']=pick.grasp_pose
        if config.get('grounding_semantic_plan') is not None:
            from .semantic_plan import check_cup_goals
            row['operation']['supplied_plan_goal_checks'] = check_cup_goals(scene, base, tucked, pick)
            if not row['operation']['supplied_plan_goal_checks']['satisfied']:
                raise RuntimeError('Sampled IK goals violate supplied semantic plan')
        save('operation_guide')
        transition=plan_pregrasp(scene,base,tucked,pick.pregrasp_q,state_constraint=(lambda q:oracle.query(np.r_[base,q])['satisfied']) if semantic else None)
        approach=cartesian_arm_path(scene,base,pick.pregrasp_q,pick.grasp_q,25,stage='grounding_reach',edge_validator=guide_edge_validator(scene,base,pick.pregrasp_q,scene.gripper_open_positions,semantic=semantic))
        guide=np.vstack([transition,approach[1:]]);grasp=guide[-1]
        set_robot_configuration(scene,base,grasp,scene.gripper_closed_positions)
        X_GC=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@scene.cup_initial_pose
        liftguide=cartesian_arm_path(scene,base,grasp,pick.lift_q,20,stage='grounding_lift',edge_validator=guide_edge_validator(scene,base,grasp,scene.gripper_closed_positions,semantic=semantic,X_GC=X_GC))
        save('operation_iris_gcs')
        reach,rr,_=solve_operation(scene,base,guide,scene.gripper_open_positions,'reach',semantic=semantic,artifacts=out/'reach_cover.json')
        lift,lr,_=solve_operation(scene,base,liftguide,scene.gripper_closed_positions,'lift',semantic=semantic,X_GC=X_GC,artifacts=out/'lift_cover.json')
        holdouts = {
            'reach': rr['region_audit'] if rr.get('region_audit') is not None else
                region_holdout(scene, base, rr['boxes'], scene.gripper_open_positions),
            'lift': lr['region_audit'] if lr.get('region_audit') is not None else
                region_holdout(scene, base, lr['boxes'], scene.gripper_closed_positions, X_GC=X_GC)}
        row['operation']['refinement'] = {'reach': rr.get('refinement'), 'lift': lr.get('refinement')}
        row['operation']['holdouts']=holdouts
        if any(v[k] for v in holdouts.values() for k in (['collisions','raw_grasp_violations']+(['semantic_violations'] if semantic else []))):raise RuntimeError('Region holdout rejected')
        mode,checker,mapping=make_checker(config,base,reach[0],scene.gripper_open_positions)
        bad=[];count=0
        for a,b in zip(reach,reach[1:]):
            for t in np.linspace(0,1,max(2,int(np.ceil(np.max(abs(b-a))/.00125))+1)):
                q=(1-t)*a+t*b;count+=1
                if not checker.CheckConfigCollisionFree(mapping(q)) or (semantic and not oracle.query(np.r_[base,q])['satisfied']):bad.append(q.tolist())
        row['operation']['strict_reach_audit']={'samples':count,'violations':len(bad),'counterexamples':bad[:5]}
        if bad:raise RuntimeError('Strict reach audit rejected; no physical execution')
        save('operation_phase_acceptance')
        opened=scene.gripper_open_positions;closed=scene.gripper_closed_positions
        operation_segments=[
            phase_segment('approach_cup',repeat(base,len(reach)),reach,repeat(opened,len(reach))),
            phase_segment('close_gripper',repeat(base,2),repeat(grasp,2),np.array([opened,closed]),dwell=12),
            phase_segment('lift_cup',repeat(base,len(lift)),lift,repeat(closed,len(lift)),X_GC),
            phase_segment('whole_body_carry_to_dropoff',repeat(base,2),repeat(lift[-1],2),repeat(closed,2),X_GC,dwell=12)]
        # Validate manipulation boundaries before spending time on navigation.
        from .accepted_motion import check_cup_boundary
        for left,right in zip(operation_segments,operation_segments[1:]):
            check_cup_boundary(left,right)
        row['operation'].update(accepted=True,phase_acceptance={s.phase:s.report for s in operation_segments},seconds=time.perf_counter()-tick,path_length_rad=float(np.linalg.norm(np.diff(reach,axis=0),axis=1).sum()+np.linalg.norm(np.diff(lift,axis=0),axis=1).sum()))
        if protocol.get('operation_first',False):run_navigation()
        if not row['bridge']['accepted']:save('operation_passed_bridge_failed');return
        save('composition')
        frames,row['composition']=compose_cup_motion([nav_segment,*operation_segments],
            start=np.r_[start,tucked,opened],end=np.r_[base,lift[-1],closed])
        # Redundant regression check; phase acceptance already covered these predicates.
        audit,trace=audit_reference(scene,frames,base,tucked,X_GC);row['assembled_reference_audit']=audit
        audit['policy_satisfied']=accepts(policy,geometry=audit['geometry_satisfied'],goals=audit['goal_semantics_satisfied'],motion=audit['motion_semantics_satisfied'],plan=audit['supplied_plan_satisfied'])
        if not audit['policy_satisfied']:raise RuntimeError('Composition regression: global audit disagrees with phase acceptance')
        (out/'delivery_trace.json').write_text(json.dumps(frames));(out/'result.json').write_text(json.dumps({'dock_q':base.tolist(),'fixed_tucked_arm_q':tucked.tolist()},indent=2))
        experiment=out/'execution';experiment.mkdir();segments=[{'start':a.tolist(),'end':b.tolist()} for a,b in zip(reach,reach[1:])]
        (experiment/'comparison.json').write_text(json.dumps([{'results':[{'method':'grounded_candidate','numerically_valid':True,'segments':segments}]}]))
        save('physical_execution')
        subprocess.run([sys.executable,'-m','src.execute_multiroute_comparison','--source',str(out),'--experiment',str(experiment)],check=True)
        row['physical']=json.loads((experiment/'execution_comparison.json').read_text())[0]
    except Exception as e:
        row['failure_stage']=row['stage'];row['failure_reason']=str(e)
        if not row['operation'].get('accepted'):row['operation'].update(reason=str(e),seconds=time.perf_counter()-tick)
    save('complete')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--worker',type=int);a=p.parse_args()
    if a.worker is not None:return worker(a.output,a.worker)
    root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
    previous=json.loads(Path('outputs/results/semantic_repair_final_20261019/full_motion/summary.json').read_text())
    candidates=[r for r in previous['candidates'] if r['status']!='rejected' and r['grasp_rpy']==[0.,0.,0.]]
    selected=[candidates[i] for i in np.linspace(0,len(candidates)-1,8,dtype=int)]
    protocol={'config':'configs/scene_semantic_choice_v6.yaml','candidates':selected,'selection':'Eight equally spaced indices from 73 geometry/whole-room screened candidates in fixed navigation-distance order; grasp RPY zero. No outcome-based substitution.',
              'methods':['nearest_only','ordered_backtracking','minimum_verified_total_cost'],
              'score':'bridge_length_m + 0.25 * (reach_plus_lift_joint_length_rad)',
              'cost_arm_weight_m_per_rad':.25,'max_candidates_per_method':8,'candidate_timeout_s':600,
              'scope':'Outer finite grounding search; not joint GCS optimization. Same candidates and per-candidate pipeline. Failure is a bounded solver/audit result, not C-space infeasibility.'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def run(i):
        folder=root/selected[i]['id'];folder.mkdir()
        with (folder/'run.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,'-m','src.semantic_grounding_candidates','--output',str(root),'--worker',str(i)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(proc.pid,signal.SIGTERM);proc.wait();code='timeout'
        f=folder/'candidate_result.json';row=json.loads(f.read_text()) if f.exists() else {'id':selected[i]['id'],'base_q':selected[i]['base_q']}
        row['process_exit']=code;row['verified_task_success']=bool(code==0 and row.get('bridge',{}).get('accepted') and row.get('operation',{}).get('accepted') and row.get('physical',{}).get('accepted'))
        f.write_text(json.dumps(row,indent=2));return row
    with ThreadPoolExecutor(max_workers=3) as pool:
        rows=[]
        for r in pool.map(run,range(8)):
            rows.append(r);(root/'comparison.json').write_text(json.dumps(rows,indent=2));print(r['id'],r['verified_task_success'],r.get('failure_reason'),flush=True)

if __name__=='__main__':main()
