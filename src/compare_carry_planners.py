"""Paired single-scene comparison with fresh regions and common retiming."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from pydrake.math import RigidTransform
from pydrake.geometry.optimization import GraphOfConvexSetsOptions
from pydrake.solvers import ClarabelSolver

from .build_scene import build_scene
from .iris_region_generator import generate_doorway_iris_region, generate_carrying_iris_cover, cover_to_dict
from .gcs_whole_body_planner import plan_carry_whole_body_gcs, plan_to_dict
from .carry_mode_gcs import plan as plan_modes, CarryOracle
from .audit_replay import audit_replay
from .semantic_monitor import monitor_delivery
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .collision_checker import evaluate_collision_status


LIMITS=np.array([.30,.30,.40]+[.40]*7)  # physical m/s and rad/s, both methods
PHYSICAL_DT=.15  # executor uses .05 reference interval × time_scale 3


def retime(points, limits=LIMITS, dt=PHYSICAL_DT):
    """Remove collinear samples, then quantize each segment duration equally.

    Preserve every corner; no corner-cutting resampling across segment knots.
    """
    knots=[]
    for q in np.asarray(points):
        if knots and np.max(np.abs(q-knots[-1]))<1e-9:continue
        while len(knots)>=2:
            a,b=knots[-2:];v=q-a;t=float((b-a)@v/(v@v)) if v@v>0 else -1
            if 0<=t<=1 and np.max(np.abs(b-(a+t*v)))<1e-8:knots.pop()
            else:break
        knots.append(q)
    out=[knots[0]]
    for a,b in zip(knots,knots[1:]):
        n=max(1,int(np.ceil(np.max(np.abs(b-a)/limits)/dt)))
        out.extend(a+(b-a)*u for u in np.linspace(0,1,n+1)[1:])
    out=np.array(out)
    assert np.max(np.abs(np.diff(out,axis=0))/dt/limits)<=1+1e-8
    return out,{"knots":len(knots),"frames":len(out),"duration_s":(len(out)-1)*dt,
                "maximum_component_speeds":np.max(np.abs(np.diff(out,axis=0))/dt,axis=0).tolist()}


def fresh_baseline(config,original,reference,directory):
    directory.mkdir(parents=True,exist_ok=True)
    start=[r for r in original if r["phase"]=="lift_cup"][-1]
    goal=[r for r in original if r["phase"]=="whole_body_carry_to_dropoff"][-1]
    carry=np.array(json.loads((reference/"iris_np2_doorway_region.json").read_text())["seed"][3:])
    X_GC=RigidTransform(np.array(start["gripper_pose"])).inverse()@RigidTransform(np.array(start["cup_pose"]))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10;options.rounding_seed=0
    options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
    names=["room2_pickup_lane","doorway_lane","room1_central_lane","room1_upper_lane"]
    kwargs=dict(start_base_q=start["base_q"],start_arm_q=start["arm_q"],carry_arm_q=carry,goal_base_q=goal["base_q"],goal_arm_q=goal["arm_q"],allowed_region_names=names,gcs_options=options)
    begin=time.perf_counter()
    scene=build_scene(config=copy.deepcopy(config),attached_cup_X_GC=X_GC,visualize=False)
    doorway=generate_doorway_iris_region(scene,carry)
    bootstrap=plan_carry_whole_body_gcs(config,**kwargs,region_overrides={"doorway_lane":doorway.region})
    cover=generate_carrying_iris_cover(scene,bootstrap.configurations,names)
    preparation=time.perf_counter()-begin
    begin=time.perf_counter()
    result=plan_carry_whole_body_gcs(config,**kwargs,region_overrides=cover.regions_by_semantic_label())
    final_wall=time.perf_counter()-begin
    metrics={"prepare_seconds":preparation,"final_planner_wall_seconds":final_wall,"final_solve_seconds":result.solve_time_seconds,
             "planning_total_seconds":preparation+final_wall,"regions":len(cover.regions),"bootstrap_solve_seconds":bootstrap.solve_time_seconds}
    (directory/"plan.json").write_text(json.dumps(plan_to_dict(result)))
    (directory/"cover.json").write_text(json.dumps(cover_to_dict(cover)))
    (directory/"metrics.json").write_text(json.dumps(metrics))
    print(json.dumps({"baseline":metrics}),flush=True)
    return result.configurations,metrics


def make_execution(method,points,config,original,output):
    output.mkdir(parents=True,exist_ok=True)
    anchor=[r for r in original if r["phase"]=="lift_cup"][-1]
    oracle=CarryOracle(config,anchor)
    points,timing=retime(points)
    generated=[]
    for q in points[1:]:
        gripper,cup=oracle.state(q[:2],q[2:])
        row=copy.deepcopy(anchor);row.update(phase="whole_body_carry_to_dropoff",base_q=q[:3].tolist(),arm_q=q[3:].tolist(),cup_pose=cup.GetAsMatrix4().tolist(),gripper_pose=gripper.GetAsMatrix4().tolist(),origin=method)
        generated.append(row)
    rows=[];inserted=False
    for row in original:
        if row["phase"]=="whole_body_carry_to_dropoff":
            if not inserted:rows.extend(generated);inserted=True
        else:rows.append(copy.deepcopy(row))
    minimum=2.;grasp_rows=[]
    for i,row in enumerate(rows):
        gripper,cup=oracle.state(row["base_q"][:2],np.array([row["base_q"][2],*row["arm_q"]]))
        # Non-carrying stages retain the original free/attached cup reference.
        scene=oracle.scene
        from .build_scene import set_robot_configuration
        set_robot_configuration(scene,row["base_q"],row["arm_q"],row["gripper_q"])
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,RigidTransform(np.array(row["cup_pose"])))
        row["gripper_pose"]=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).GetAsMatrix4().tolist()
        status=evaluate_collision_status(scene,row["base_q"],arm_q=row["arm_q"],gripper_q=row["gripper_q"])
        row["index"]=i;row["collision_residual"]=status.residual
        row["cup_gripper_audit"]=measure_cup_gripper(scene)
        if row["phase"]=="whole_body_carry_to_dropoff":minimum=min(minimum,status.minimum_signed_distance)
        grasp_rows.append(row)
    semantic=monitor_delivery(rows,config);grasp=summarize_grasp_audit(grasp_rows);dense=audit_replay(rows,config,10)
    differences=np.diff(points,axis=0)
    metrics={"base_length_m":float(np.linalg.norm(differences[:,:2],axis=1).sum()),"yaw_travel_rad":float(np.abs(differences[:,2]).sum()),"arm_travel_rad":float(np.linalg.norm(differences[:,3:],axis=1).sum()),"retiming":timing,
             "semantic":semantic["satisfied"],"grasp_geometry":grasp["satisfied"],"dense_audit":dense,"carry_sample_min_all_pair_clearance":minimum}
    for name,data in [("delivery_trace.json",rows),("scene_config_snapshot.json",config),("metrics.json",metrics),("semantic.json",semantic),("grasp.json",grasp)]:
        (output/name).write_text(json.dumps(data))
    print(json.dumps({method:metrics}),flush=True)
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--reference",type=Path,default=Path("outputs/results/full_dynamics/reference"));p.add_argument("--output",type=Path,default=Path("outputs/results/paired_carry_comparison"));p.add_argument("--repeats",type=int,default=3);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    config=json.loads((args.reference/"scene_config_snapshot.json").read_text());original=json.loads((args.reference/"delivery_trace.json").read_text())
    summary={"planning_runs":{"baseline":[],"disks":[]},"common_physical_velocity_limits":LIMITS.tolist(),"common_control_dt":.01,"common_physics_dt":.001,"common_time_scale":3.,"scope":"One fixed scene, repeated timings; shared frozen task/IK anchors. Disk posture library comes from prior baseline trajectory; not independent end-to-end planning."}
    for repeat in range(args.repeats):
        print(f"PAIR {repeat+1}/{args.repeats}",flush=True)
        points,metrics=fresh_baseline(config,original,args.reference,args.output/f"baseline_run{repeat}")
        summary["planning_runs"]["baseline"].append(metrics)
        plan_modes(args.reference,args.output/f"disk_run{repeat}")
        data=json.loads((args.output/f"disk_run{repeat}"/"result.json").read_text());m=data["metrics"]
        summary["planning_runs"]["disks"].append({**m,"planning_total_seconds":m["prepare_seconds"]+m["solve_seconds"]})
        if repeat==0:
            summary["baseline"]=make_execution("baseline",points,config,original,args.output/"baseline_reference")
            samples=data["samples"];q=np.array([[*r["base_q"],*r["arm_q"]] for r in samples])
            summary["disks"]=make_execution("disks",q,config,original,args.output/"disk_reference")
        (args.output/"summary.json").write_text(json.dumps(summary,indent=2))
    print('PLANNING COMPARISON COMPLETE',flush=True)


if __name__=="__main__":main()
