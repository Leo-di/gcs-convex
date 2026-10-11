"""Plan and execute doorway pickup with one layered semantic GCS solve.

All free-space, open-cup, grasp-action, and attached-cup convex sets are added
to one 10-D Drake GcsTrajectoryOptimization. Region construction may use
collision-checked guide configurations, but only one SolvePath chooses the
complete start-to-retraction trajectory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from pydrake.geometry.optimization import (
    GraphOfConvexSetsOptions, HPolyhedron, Point,
)
from pydrake.math import RigidTransform
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver

from .build_scene import build_scene, set_robot_configuration
from .execute_delivery_dynamics import execute
from .geometry_outbound_gcs import (
    clipped_disks_intersect, make_cover, select_navigation_arm,
)
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .grasp_planner import plan_pick, cartesian_arm_path, solve_arm_ik
from .manipulation_gcs import (
    plan_manipulation_phase, phase_plan_to_dict, prepare_mode_machinery,
)
from .safe_pregrasp import plan_pregrasp
from .experiment_gcs_bridging import run as build_gcs_bridging_cover
from .scene_slice_gcs import SliceOracle
from .scene_spec import load_scene_config


def _base_cover(scene, start, goal, arm_q, spacing=.15, max_radius=.18):
    from pydrake.geometry.optimization import CartesianProduct, HPolyhedron, Intersection

    cfg = scene.config
    yaw=float(cfg['robot']['start'][2])
    lane_names = set(cfg['pick_task']['outbound_regions'])
    boxes = [r for r in cfg['gcs_base_planning']['regions']
             if r['name'] in lane_names and
             r['yaw_bounds'][0] <= yaw <= r['yaw_bounds'][1]]
    if not boxes:
        raise RuntimeError('No Room 1 navigation lane at the selected yaw.')
    if abs(float(scene.config['pick_task']['candidates'][0]['base_q'][2])-yaw)>1e-10:
        raise RuntimeError('Unified reduced-state GCS requires fixed base yaw.')
    oracle = SliceOracle(scene, arm_q, scene.gripper_open_positions,
                         yaw=yaw, max_distance=max_radius + .0201)
    # A single convex corridor is possible for this obstacle-free Room 1
    # approach. Admit it only after auditing the full robot geometry on a dense
    # 2-D lattice; the resulting convex set still participates in global GCS.
    a=np.asarray(start[:2],float); b=np.asarray(goal[:2],float)
    direction=(b-a)/np.linalg.norm(b-a); normal=np.array([-direction[1],direction[0]])
    length=float(np.linalg.norm(b-a)); half_width=.08
    tube=HPolyhedron(np.vstack([direction,-direction,normal,-normal]),
        np.array([length+direction@a,-direction@a,
                  half_width+normal@a,half_width-normal@a]))
    lane=Intersection([tube,HPolyhedron.MakeBox(
        np.asarray(boxes[0]['lower'],float),np.asarray(boxes[0]['upper'],float))])
    xy_grid=[]
    for s in np.linspace(0.,length,101):
        for lateral in np.linspace(-half_width,half_width,17):
            xy=a+s*direction+lateral*normal
            if lane.PointInSet(xy,1e-9): xy_grid.append(xy)
    audited=[oracle.query(xy) for xy in xy_grid]
    minimum=min(item[0] for item in audited)
    minimum_self=min(item[1] for item in audited)
    if minimum<.02 or minimum_self<0.:
        raise RuntimeError(f'Room 1 convex-tube collision audit failed: {minimum:.4f} m.')
    region=CartesianProduct([lane,Point(arm_q)])
    if not region.PointInSet(start,1e-8) or not region.PointInSet(goal,1e-8):
        raise RuntimeError('Audited Room 1 convex tube excludes start or staging pose.')
    sources=targets=[0]
    audit={'method':'convex tube intersected with configured Room 1 semantic lane',
        'half_width_m':half_width,'samples':len(audited),
        'minimum_environment_clearance_m':minimum,
        'minimum_self_clearance_m':minimum_self,'oracle_calls':oracle.calls,
        'scope':'Fixed arm posture and yaw; dense lattice audit plus full trajectory collision audit.'}
    return [region], [], sources, targets, oracle, audit


def _state9(q10):
    q10=np.asarray(q10,float)
    if abs(q10[2])>1e-9:
        raise ValueError('This frozen scene uses yaw=0 throughout; cannot project a varying yaw.')
    return np.r_[q10[:2],q10[3:]]


def _project_region_to_9d(region):
    """Intersect a 10-D region with the task's fixed yaw=0 subspace."""
    from pydrake.geometry.optimization import HPolyhedron
    keep=[0,1,3,4,5,6,7,8,9]
    return HPolyhedron(region.A()[:,keep],region.b())


def _carry_guide(scene, config, base_q, pick):
    """Generate collision-checked seeds; this is not a trajectory solve."""
    set_robot_configuration(scene, base_q, pick.grasp_q,
                            scene.gripper_closed_positions)
    X_WG=scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
    X_GC=X_WG.inverse() @ scene.cup_initial_pose
    lift=cartesian_arm_path(scene,base_q,pick.grasp_q,pick.lift_q,18,
                            stage='unified_gcs_lift_guide')
    attached=build_scene(config=config,attached_cup_X_GC=X_GC,visualize=False)
    set_robot_configuration(scene,base_q,pick.lift_q,scene.gripper_closed_positions)
    lift_pose=scene.gripper_body.body_frame().CalcPoseInWorld(scene.plant_context)
    cursor=np.asarray((lift_pose @ X_GC).translation(),float)
    waypoints=[np.array([.28,.78,1.02]),np.array([.15,.42,1.02]),np.array([-.07,.42,1.02])]
    arm_refs=[pick.grasp_q.copy(),*lift[1:]]
    seed=pick.lift_q.copy()
    for endpoint in waypoints:
        endpoint=np.asarray(endpoint,float)
        count=max(2,int(np.ceil(np.linalg.norm(endpoint-cursor)/.08)))
        for alpha in np.linspace(0.,1.,count+1)[1:]:
            cup_xyz=(1-alpha)*cursor+alpha*endpoint
            X_WC=RigidTransform(scene.cup_initial_pose.rotation(),cup_xyz)
            X_WG_target=X_WC @ X_GC.inverse()
            point=np.asarray(config['gripper']['grasp_point_in_body'],float)
            ee_target=X_WG_target.multiply(point)
            seed=solve_arm_ik(attached,base_q=base_q,target_position=ee_target,
                target_orientation=X_WG_target.rotation(),initial_arm_q=seed,
                position_tolerance=.002,orientation_tolerance=.06,
                minimum_clearance=.003,stage='unified_gcs_carry_guide')
            arm_refs.append(seed)
        cursor=endpoint
    reference=np.asarray([np.r_[base_q,q] for q in arm_refs])
    return X_GC, reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/scene_doorway_reach_semantic_v9.yaml')
    parser.add_argument('--output',default='outputs/results/doorway_arm_reach_semantic_v12/gcs_bridged')
    parser.add_argument('--reach-method',choices=('reference_corridor','gcs_bridging'),default='reference_corridor')
    args=parser.parse_args()
    began=time.perf_counter()
    config=load_scene_config(args.config)
    if args.reach_method == 'gcs_bridging':
        staging=next(r for r in config['gcs_base_planning']['regions']
                     if r['name']=='doorway_pickup_staging')
        staging['upper'][0]=-.45
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    base_start=np.asarray(config['robot']['start'],float)
    base_q=np.asarray(config['pick_task']['candidates'][0]['base_q'],float)
    scene=build_scene(config=config,visualize=False)

    # Task-space IK and collision-checked guides only seed the convex covers.
    # They do not choose any segment of the final trajectory.
    nav_arm=select_navigation_arm(scene,[base_start,base_q])
    pick=plan_pick(scene,base_q)
    pre_bridge_report=None
    if args.reach_method == 'reference_corridor':
        pre_transition=plan_pregrasp(scene,base_q,nav_arm,pick.pregrasp_q)
        approach=cartesian_arm_path(scene,base_q,pick.pregrasp_q,pick.grasp_q,25,
                                    stage='unified_gcs_approach_guide')
        # Keep the exact contact state as the shared GCS grasp-transition state.
        if not np.allclose(approach[-1],pick.grasp_q,atol=1e-9):
            approach=np.vstack([approach,pick.grasp_q])
        pre_reference=np.vstack([
            np.column_stack([np.repeat(base_q[None,:],len(pre_transition),axis=0),pre_transition]),
            np.column_stack([np.repeat(base_q[None,:],len(approach)-1,axis=0),approach[1:]])])
        pre_cover=plan_manipulation_phase(mode='pregrasp',config=config,
            reference_q10=pre_reference,scene_cache={},solve=False)
    else:
        sparse_pre_reference=np.vstack([np.r_[base_q,nav_arm],np.r_[base_q,pick.grasp_q]])
        bridge_path=output/'gcs_bridging'/'bridge_cover.json'
        pre_bridge_report=build_gcs_bridging_cover(
            args.config,str(bridge_path),solve_local=False)
        if not pre_bridge_report.get('connected_start_goal'):
            raise RuntimeError('GCS-Bridging did not connect the sparse free-cup reach endpoints.')
        pre_regions=[HPolyhedron(np.asarray(item['A']),np.asarray(item['b']))
                     for item in pre_bridge_report['selected_region_halfspaces']]
        pre_edges=[]
        for i,j in pre_bridge_report['selected_overlap_edges']:
            pre_edges.extend([(i,j),(j,i)])
        pre_machinery=prepare_mode_machinery(
            config=config,mode='pregrasp',reference_q10=sparse_pre_reference,
            scene_cache={})
        pre_cover=SimpleNamespace(
            regions=tuple(pre_regions),edges=tuple(pre_edges),
            machinery=pre_machinery,region_method='gcs_bridging_directline_binary_iris_np2',
            iris_solve_seconds=float(pre_bridge_report['initial_iris_seconds'])+
                sum(x['inflation_seconds'] for x in pre_bridge_report['inflated_region_audits']),
            region_results=tuple())
    X_GC,carry_reference=_carry_guide(scene,config,base_q,pick)

    # Each function below constructs regions and semantic-layer edges only.
    # solve=False is the key distinction from the previous staged pipeline.
    carry_cover=plan_manipulation_phase(mode='grasp',config=config,
        reference_q10=carry_reference,X_GC=X_GC,scene_cache={},solve=False)

    nav_start10=np.r_[base_start,nav_arm]
    nav_goal10=np.r_[base_q,nav_arm]
    nav_start=_state9(nav_start10);nav_goal=_state9(nav_goal10)
    nav_regions,nav_edges,nav_sources,nav_targets,nav_oracle,nav_records=_base_cover(
        scene,nav_start,nav_goal,nav_arm)
    grasp_q10=np.r_[base_q,pick.grasp_q]
    grasp_q9=_state9(grasp_q10)
    carry_goal10=np.asarray(carry_reference[-1],float)
    carry_goal=_state9(carry_goal10)
    pre_regions=[_project_region_to_9d(r) for r in pre_cover.regions]
    carry_regions=[_project_region_to_9d(r) for r in carry_cover.regions]

    spec=config['gcs_base_planning']; whole=config['whole_body_planning']
    # All requested base motions use yaw=0; eliminate that fixed coordinate
    # exactly so the shared optimization has nine independent dimensions.
    planner=GcsTrajectoryOptimization(9)
    order=int(spec['bezier_order'])
    nav_graph=planner.AddRegions(nav_regions,nav_edges,order=order,h_min=.01,h_max=20,name='product_room1_base_navigation')
    pre_graph=planner.AddRegions(pre_regions,list(pre_cover.edges),order=order,h_min=.01,h_max=30,name='product_free_cup_arm_reach')
    action_graph=planner.AddRegions([Point(grasp_q9)],order=0,h_min=.6,h_max=.8,name='semantic_action_bilateral_grasp')
    carry_graph=planner.AddRegions(carry_regions,list(carry_cover.edges),order=order,h_min=.01,h_max=30,name='product_attached_cup_retraction')
    source=planner.AddRegions([Point(nav_start)],order=0,name='task_start')
    target=planner.AddRegions([Point(carry_goal)],order=0,name='task_cup_retracted_room1')

    planner.AddEdges(source,nav_graph,edges_between_regions=[(0,i) for i in nav_sources])
    stage_regions=[i for i,r in enumerate(nav_regions) if r.PointInSet(nav_goal)]
    if not stage_regions:
        raise RuntimeError('No base navigation region reaches the arm-reach staging state.')
    planner.AddEdges(nav_graph,pre_graph,edges_between_regions=[(i,0) for i in stage_regions])
    if not pre_regions[-1].PointInSet(grasp_q9):
        raise RuntimeError('The free-cup reach corridor does not end at the grasp state.')
    planner.AddEdges(pre_graph,action_graph,edges_between_regions=[(len(pre_cover.regions)-1,0)])
    grasp_regions=[i for i,r in enumerate(carry_regions) if r.PointInSet(grasp_q9)]
    if not grasp_regions:
        raise RuntimeError('The attached-cup IRIS cover excludes the grasp transition state.')
    planner.AddEdges(action_graph,carry_graph,edges_between_regions=[(0,i) for i in grasp_regions])
    goal_regions=[i for i,r in enumerate(carry_regions) if r.PointInSet(carry_goal)]
    if not goal_regions:
        raise RuntimeError('The attached-cup IRIS cover excludes the Room 1 retraction goal.')
    planner.AddEdges(carry_graph,target,edges_between_regions=[(i,0) for i in goal_regions])

    metric=np.diag([float(spec['path_length_weight']),float(spec['path_length_weight'])]
        +[float(whole['joint_path_weight'])]*7)
    planner.AddPathLengthCost(metric);planner.AddTimeCost(float(spec['time_weight']))
    velocity=np.asarray([float(spec['maximum_speed']),float(spec['maximum_speed'])]
        +[float(whole['maximum_joint_speed'])]*7)
    planner.AddVelocityBounds(-velocity,velocity)
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True
    options.max_rounded_paths=10;options.rounding_seed=0
    options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
    print('Solving the complete layered 9-D semantic GCS once...',flush=True)
    solve_start=time.perf_counter()
    trajectory,result=planner.SolvePath(source,target,options)
    solve_seconds=time.perf_counter()-solve_start
    if not result.is_success():
        raise RuntimeError(f'Unified semantic GCS failed: {result.get_solution_result()}')

    graph=planner.graph_of_convex_sets()
    selected=graph.GetSolutionPath(source.Vertices()[0],target.Vertices()[0],result)
    selected_vertices=[source.Vertices()[0]]+[edge.v() for edge in selected]
    stage_by_id={}
    for subgraph,phase in [(source,'navigate_to_pickup'),(nav_graph,'navigate_to_pickup'),
                           (pre_graph,'approach_cup'),(action_graph,'close_gripper'),
                           (carry_graph,'whole_body_carry_to_dropoff'),(target,'whole_body_carry_to_dropoff')]:
        for vertex in subgraph.Vertices():stage_by_id[vertex.id()]=phase
    if trajectory.get_number_of_segments()!=len(selected_vertices):
        raise RuntimeError(f'Could not align GCS semantic path ({trajectory.get_number_of_segments()} trajectory segments, {len(selected_vertices)} vertices).')
    frames=[]
    hz=float(spec['samples_per_second'])
    first_carry_vertex=True
    for index,vertex in enumerate(selected_vertices):
        phase=stage_by_id[vertex.id()]
        if vertex.id() in {v.id() for v in action_graph.Vertices()}:
            phase='close_gripper'
        elif vertex.id() in {v.id() for v in pre_graph.Vertices()}:
            phase='approach_cup'
        elif vertex.id() in {v.id() for v in carry_graph.Vertices()}:
            # The first attached-mode segment physically lifts the freely
            # resting cup off the pickup table. Preserve that transition in
            # the execution trace so the contact audit treats table support
            # as expected only until lift-off; all later carry segments use
            # the transport mode. This labels the existing GCS segment and
            # does not add a waypoint or alter the optimization.
            phase='lift_cup' if first_carry_vertex else 'whole_body_carry_to_dropoff'
            first_carry_vertex=False
        elif vertex.id() in {v.id() for v in nav_graph.Vertices()}:
            phase='navigate_to_pickup'
        segment=trajectory.segment(index)
        count=max(2,int(np.ceil((segment.end_time()-segment.start_time())*hz))+1)
        for t in np.linspace(segment.start_time(),segment.end_time(),count):
            q9=np.asarray(trajectory.value(t),float).reshape(-1)
            q=np.r_[q9[:2],0.,q9[2:]]
            grip=scene.gripper_open_positions if phase in ('navigate_to_pickup','approach_cup') else scene.gripper_closed_positions
            frames.append({'base_q':q[:3].tolist(),'arm_q':q[3:].tolist(),
                           'gripper_q':np.asarray(grip).tolist(),'phase':phase})
    frames[0]['base_q']=base_start.tolist();frames[0]['arm_q']=nav_arm.tolist()
    frames[-1]['base_q']=carry_goal10[:3].tolist();frames[-1]['arm_q']=carry_goal10[3:].tolist()
    # Independently replay every saved GCS configuration against the frozen
    # collision model belonging to its semantic layer before physical execution.
    nav_bad=0;pre_bad=0;carry_bad=0;nav_min=float('inf')
    for frame in frames:
        q10=np.r_[frame['base_q'],frame['arm_q']]
        if frame['phase']=='navigate_to_pickup':
            distance,self_distance,_=nav_oracle.query(q10[:2])
            nav_min=min(nav_min,distance)
            nav_bad+=int(distance<.02 or self_distance<0.)
        elif frame['phase'] in ('approach_cup','close_gripper'):
            pre_bad+=int(not pre_cover.machinery.checker.CheckConfigCollisionFree(
                pre_cover.machinery.parameterization(q10)))
        else:
            carry_bad+=int(not carry_cover.machinery.checker.CheckConfigCollisionFree(
                carry_cover.machinery.parameterization(q10)))
    trajectory_audit={'samples':len(frames),'navigation_collisions':nav_bad,
        'free_cup_arm_collisions':pre_bad,'attached_cup_arm_collisions':carry_bad,
        'navigation_minimum_environment_clearance_m':nav_min}
    if nav_bad or pre_bad or carry_bad:
        raise RuntimeError(f'Unified GCS trajectory failed independent collision audit: {trajectory_audit}')
    output.mkdir(parents=True,exist_ok=True)
    (output/'scene_config_snapshot.json').write_text(json.dumps(config,indent=2))
    (output/'delivery_trace.json').write_text(json.dumps(frames))
    (output/'gcs_graphviz.dot').write_text(planner.GetGraphvizString())
    np.savez_compressed(output/'unified_trajectory.npz',
        times=np.asarray(trajectory.get_segment_times()),
        configurations=np.asarray([np.r_[trajectory.value(t).reshape(-1)[:2],0.,trajectory.value(t).reshape(-1)[2:]] for t in trajectory.get_segment_times()]))

    print(f'One GCS solve complete ({solve_seconds:.2f}s); executing {len(frames)} frames in SAP...',flush=True)
    report,dyn_scene=execute(config,frames,time_step=.001,time_scale=2.,visualize=True,grip=True)
    trace=report['trace'];door=report['doorway_semantics']
    bilateral=[min(r['finger_contact_force_N'].values()) for r in trace]
    radius=float(config['objects']['cup']['radius'])
    cup_in_room1=[r for r in trace if r['cup_pose'][0][3]+radius<0.]
    audit={
        'base_entered_room2':door['base_entered_room2'],
        'base_min_door_clearance_m':door['minimum_base_geometry_clearance_to_door_plane_m'],
        'end_effector_entered_room2':door['end_effector_entered_room2'],
        'maximum_end_effector_room2_depth_m':door['maximum_end_effector_room2_depth_m'],
        'arm_crossed_doorway':door['arm_crossed_doorway'],
        'cup_grasped_with_bilateral_contact':bool(any(x>.05 for x in bilateral)),
        'cup_retracted_to_room1_while_contacting':bool(any(min(r['finger_contact_force_N'].values())>.05 for r in cup_in_room1)),
        'unexpected_contact_count':report['unexpected_contact_count'],
    }
    audit['satisfied']=bool(not audit['base_entered_room2'] and audit['end_effector_entered_room2'] and audit['arm_crossed_doorway'] and audit['cup_grasped_with_bilateral_contact'] and audit['cup_retracted_to_room1_while_contacting'] and audit['unexpected_contact_count']==0)
    report['execution_task']='one_shot_layered_semantic_gcs_doorway_reach'
    report['generic_delivery_release_and_settle_check_applicable']=False
    report['doorway_reach_task_audit']=audit
    report['satisfied']=bool(audit['satisfied'] and report['lifted_with_bilateral_contact'] and report['carried_to_room1_with_bilateral_contact'] and report['base_workspace_bounds_satisfied'] and report['actuator_effort_limits_satisfied'] and report['carry_slip_satisfied'])
    report['unified_gcs']={
        'algorithm':'One Drake GcsTrajectoryOptimization SolvePath over a 9-D reduced semantic product graph',
        'state':'(x, y, q_arm[0:7]); base yaw is fixed to 0 and eliminated exactly',
        'solve_calls':1,'solver':'Clarabel','convex_relaxation':True,
        'solve_seconds':solve_seconds,'total_region_generation_seconds':time.perf_counter()-began-solve_seconds,
        'regions':{'room1_navigation':len(nav_regions),'free_cup_reach':len(pre_cover.regions),
                   'grasp_action_vertex':1,'attached_cup_retraction':len(carry_cover.regions)},
        'semantic_layers':['base_in_room1','arm_reach_with_free_cup','bilateral_grasp_action','attached_cup_retraction_to_room1'],
        'selected_graph_path':[v.name() for v in selected_vertices],
        'trajectory_collision_audit':trajectory_audit,
        'region_generation':{'pregrasp_method':pre_cover.region_method,
            'pregrasp_iris_seconds':pre_cover.iris_solve_seconds,
            'attached_cup_method':carry_cover.region_method,
            'attached_cup_iris_seconds':carry_cover.iris_solve_seconds,
            'attached_cup_holdout_samples':sum(x.holdout_samples for x in carry_cover.region_results),
            'attached_cup_holdout_collisions':sum(x.holdout_collisions for x in carry_cover.region_results)},
        'scope':'All continuous configurations are chosen in one rounded convex GCS solve. The free-cup reach cover is connected from sparse semantic endpoints by the selected region-generation method; semantic mode transitions and all movement layers are in the same graph. SAP execution is a separate physical validation.'}
    if pre_bridge_report is not None:
        report['unified_gcs']['reach_connectivity_repair']={
            key:pre_bridge_report[key] for key in (
                'initial_region_count','initial_component_count','initial_component_sizes',
                'closest_region_pair_attempts','bridge_method','binary_inflation',
                'final_region_count','final_component_sizes','connected_start_goal',
                'shortest_overlap_chain','added_regions','wall_seconds_before_gcs')}
    report['trace']=[{**r,'semantic':{
        'base_in_room1':r['base_geometry_door_clearance_m']>=0,
        'base_in_room2':r['base_geometry_door_clearance_m']<0,
        'end_effector_in_room2':r['end_effector_position_world_m'][0]>0,
        'arm_crosses_door':r['arm_crossed_doorway_so_far'],
        'cup_in_room2':r['cup_pose'][0][3]-radius>0,
        'cup_grasped':min(r['finger_contact_force_N'].values())>.05,
        'cup_retracted_to_room1':r['cup_pose'][0][3]+radius<0 and min(r['finger_contact_force_N'].values())>.05}}
        for r in trace]
    (output/'semantic_trace.json').write_text(json.dumps([
        {'time_s':r['time_s'],'phase':r['phase'],**r['semantic'],
         'base_q':r['base_q'],'ee_position_world_m':r['end_effector_position_world_m'],
         'base_geometry_door_clearance_m':r['base_geometry_door_clearance_m'],
         'cup_x_m':r['cup_pose'][0][3],
         'left_finger_contact_force_N':r['finger_contact_force_N']['left_finger'],
         'right_finger_contact_force_N':r['finger_contact_force_N']['right_finger']}
        for r in report['trace']],indent=2))
    (output/'dynamics_trace.json').write_text(json.dumps(report['trace']))
    report.pop('trace',None)
    (output/'result.json').write_text(json.dumps(report,indent=2))
    (output/'scene.html').write_text(dyn_scene.meshcat.StaticHtml())
    print(json.dumps({'satisfied':report['satisfied'],'audit':audit,'unified_gcs':report['unified_gcs']},indent=2),flush=True)
    if not report['satisfied']: raise SystemExit(2)


if __name__=='__main__':
    main()
