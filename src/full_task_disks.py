"""Complete task replay: new disk-GCS navigation + saved whole-body task.

Transport disks are pose-conditioned geometric audits, not a new global
disk-GCS solution. Intended contacts are audited separately, never inflated
into fictitious free-space disks.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from pydrake.math import RigidTransform

from .build_scene import build_scene, set_robot_configuration
from .collision_checker import evaluate_collision_status
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .semantic_monitor import monitor_delivery
from .audit_replay import audit_replay


def state_clearance(scene, attached):
    """Moving set is robot, plus cup only while attached; use raw cup pairs."""
    p = scene.plant
    robot = {scene.base_instance, scene.arm_instance, scene.gripper_instance}
    query=scene.scene_graph.get_query_output_port().Eval(scene.scene_graph.GetMyContextFromRoot(scene.root_context))
    inspector=query.inspector()
    moving=lambda b:b.model_instance() in robot or (attached and b.index()==scene.cup_body.index())
    distance=2.
    for pair in query.ComputeSignedDistancePairwiseClosestPoints(2.):
        bodies=[p.GetBodyFromFrameId(inspector.GetFrameId(g)) for g in (pair.id_A,pair.id_B)]
        if moving(bodies[0]) != moving(bodies[1]):distance=min(distance,float(pair.distance))
    # Explicitly include cup/table and cup/fingertip pairs hidden by planning
    # contact filters whenever the two geometries are in different moving sets.
    for cup_id in p.GetCollisionGeometriesForBody(scene.cup_body):
        for index in p.GetBodyIndices(scene.gripper_instance)+[p.GetBodyByName("table").index()]:
            body=p.get_body(index)
            if moving(body)==attached:continue
            for gid in p.GetCollisionGeometriesForBody(body):
                distance=min(distance,float(query.ComputeSignedDistancePairClosestPoints(cup_id,gid).distance))
    return distance


def run(reference, navigation, output, carry_plan_path=None):
    config=json.loads((reference/"scene_config_snapshot.json").read_text())
    original=json.loads((reference/"delivery_trace.json").read_text())
    nav=json.loads((navigation/"result.json").read_text())
    nav_config=json.loads((navigation/"config.json").read_text())
    differences={k for k in set(config)|set(nav_config) if config.get(k)!=nav_config.get(k)}
    if differences-{"execution_dynamics"}:raise ValueError(f"Scene mismatch: {differences}")
    # Controller gains do not alter geometry in this kinematic replay.
    scene=build_scene(config=config,visualize=True)
    rows=[]
    for f in nav["frames"]:
        row=copy.deepcopy(original[0]);row.update(base_q=f["base_q"],arm_q=nav["fixed_arm_q"],gripper_q=nav["fixed_gripper_q"],time_s=f["time_s"],origin="new_2d_disk_gcs",nav_disk=f["disk"])
        rows.append(row)
    carry_plan=json.loads(carry_plan_path.read_text()) if carry_plan_path else None
    remaining=[];inserted=False
    for row in original:
        if row["phase"]=="navigate_to_pickup":continue
        if carry_plan and row["phase"]=="whole_body_carry_to_dropoff":
            if not inserted:remaining.extend(carry_plan["samples"]);inserted=True
        else:remaining.append(row)
    # Retain original post-navigation configurations exactly. Timing is a
    # kinematic display choice (0.15 s/reference frame), not a dynamics result.
    for i,row in enumerate(remaining):
        rows.append({**copy.deepcopy(row),"time_s":nav["duration_s"]+.15*(i+1),"origin":row.get("origin", "saved_10d_gcs" if row["phase"]=="whole_body_carry_to_dropoff" else "saved_ik_contact_transition")})
    links=[scene.plant.GetBodyByName(f"iiwa_link_{i}",scene.arm_instance) for i in range(8)]
    lower=np.array(config["robot"]["lower_limits"][:2]);upper=np.array(config["robot"]["upper_limits"][:2])
    frames=[];boundary_samples=0;minimum_boundary=2.
    scene.meshcat.StartRecording()
    for index,row in enumerate(rows):
        set_robot_configuration(scene,row["base_q"],row["arm_q"],row["gripper_q"])
        cup=RigidTransform(np.array(row["cup_pose"]))
        scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,cup)
        status=evaluate_collision_status(scene,row["base_q"],arm_q=row["arm_q"],gripper_q=row["gripper_q"])
        row["index"]=index;row["collision_residual"]=status.residual
        row["gripper_pose"]=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).GetAsMatrix4().tolist()
        row["cup_gripper_audit"]=measure_cup_gripper(scene)
        clearance=state_clearance(scene,row["attached"])
        xy=np.array(row["base_q"][:2]);margin=.002
        radius=max(0.,min(.6,clearance-margin-.0001,np.min(xy-lower),np.min(upper-xy)))  # 0.1 mm numerical reserve
        # Boundary samples rigidly translate both robot and held object;
        # fixed objects, including an ungrasped/released cup, stay fixed.
        if radius>0 and row["phase"]!="navigate_to_pickup":
            for angle in np.linspace(0,2*np.pi,8,endpoint=False):
                shift=np.array([radius*np.cos(angle),radius*np.sin(angle),0.])
                set_robot_configuration(scene,[*(xy+shift[:2]),row["base_q"][2]],row["arm_q"],row["gripper_q"])
                if row["attached"]:
                    moved=RigidTransform(cup.rotation(),cup.translation()+shift)
                    scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,moved)
                value=state_clearance(scene,row["attached"])
                boundary_samples+=1;minimum_boundary=min(minimum_boundary,value)
                if value<margin-1e-6:raise AssertionError(f"Loaded pose disk failed at frame {index}: {value}")
            set_robot_configuration(scene,row["base_q"],row["arm_q"],row["gripper_q"])
            scene.plant.SetFreeBodyPose(scene.plant_context,scene.cup_body,cup)
        xyz=lambda b:scene.plant.EvalBodyPoseInWorld(scene.plant_context,b).translation().tolist()
        scene.root_context.SetTime(row["time_s"]);scene.diagram.ForcedPublish(scene.root_context)
        frames.append({"time_s":row["time_s"],"phase":row["phase"],"origin":row["origin"],"base_q":row["base_q"],"links_xyz":[xyz(b) for b in links],"gripper_xyz":xyz(scene.gripper_body),"cup_pose":row["cup_pose"],"attached":row["attached"],"local_radius":radius,"local_margin":margin,"local_numerical_reserve":.0001,"clearance":clearance,"nav_disk":row.get("nav_disk"),"carry_mode":row.get("carry_mode"),"carry_disk":row.get("carry_disk"),"carry_switch":row.get("carry_switch")})
    scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
    monitor=monitor_delivery(rows,config);grasp=summarize_grasp_audit(rows)
    dense=audit_replay(rows,config,subdivisions=10)
    if not(monitor["satisfied"] and grasp["satisfied"] and dense["satisfied"]):
        raise AssertionError({"monitor":monitor["first_violation"],"grasp":grasp["first_violation"],"dense":dense})
    data={**nav,"carry_plan":carry_plan,"frames":frames,"duration_s":frames[-1]["time_s"],"phases":list(dict.fromkeys(f["phase"] for f in frames)),"monitor":monitor,"grasp_audit":grasp,"dense_audit":dense,"local_disk_audit":{"samples":boundary_samples,"minimum_clearance":minimum_boundary},"scope":("Complete kinematic task: new disk-GCS navigation and finite-mode disk-GCS carrying; IK transitions reused. Switches sampled, no continuous-time certificate. Attachment grasp, no new dynamics validation." if carry_plan else "Complete kinematic task. New disk-GCS navigation; reused 10D GCS carrying and IK transitions. Pose-dependent disks are local audits, not mode-transition certificates. Attachment grasp, no new dynamics validation.")}
    output.mkdir(parents=True,exist_ok=True)
    for name,obj in [("result.json",data),("delivery_trace.json",rows),("scene_config_snapshot.json",config),("dense_replay_audit.json",dense)]:
        (output/name).write_text(json.dumps(obj))
    (output/"scene.html").write_text(scene.meshcat.StaticHtml())
    (output/"index.html").write_text(Path(__file__).with_name("full_task_disks_view.html").read_text().replace("__DATA__",json.dumps(data)))
    print(json.dumps({"frames":len(frames),"phases":data["phases"],"dense":dense,"disk_audit":data["local_disk_audit"],"semantic":monitor["satisfied"],"grasp":grasp["satisfied"]},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=Path("outputs/results/full_dynamics/reference"))
    parser.add_argument("--navigation",type=Path,default=Path("outputs/results/grasp_fix_web/planar/slice"))
    parser.add_argument("--output",type=Path,default=Path("outputs/results/grasp_fix_web/planar/full_task"))
    parser.add_argument("--carry-plan",type=Path)
    args=parser.parse_args();run(args.reference,args.navigation,args.output,args.carry_plan)
