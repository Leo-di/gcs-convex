"""Finite-posture carrying planner: XY disk GCS with audited switch anchors.

Posture anchors come from an existing plan; its transport XY path is not
replayed. GCS selects new translations and discrete switching locations.
Switch trajectories are densely audited, not continuously certified.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry.optimization import Hyperellipsoid, Point, GraphOfConvexSetsOptions
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.math import RigidTransform
from pydrake.solvers import ClarabelSolver
from pydrake.symbolic import Expression

from .build_scene import build_scene, set_robot_configuration
from .full_task_disks import state_clearance
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit


class CarryOracle:
    def __init__(self, config, anchor):
        self.scene=build_scene(config=config,visualize=False)
        self.gripper=anchor["gripper_q"]
        self.X_GC=RigidTransform(np.array(anchor["gripper_pose"])).inverse() @ RigidTransform(np.array(anchor["cup_pose"]))
        self.calls=0

    def state(self, xy, posture):
        s=self.scene
        set_robot_configuration(s,[*xy,posture[0]],posture[1:],self.gripper)
        gripper=s.plant.EvalBodyPoseInWorld(s.plant_context,s.gripper_body)
        cup=gripper @ self.X_GC
        s.plant.SetFreeBodyPose(s.plant_context,s.cup_body,cup)
        return gripper,cup

    def query(self, xy, posture):
        self.calls+=1;self.state(xy,posture)
        s=self.scene
        query=s.scene_graph.get_query_output_port().Eval(s.scene_graph.GetMyContextFromRoot(s.root_context))
        minimum=min((float(p.distance) for p in query.ComputeSignedDistancePairwiseClosestPoints(2.)),default=2.)
        return state_clearance(s,True),minimum

    def switch(self, xy, a, b, step=.035, margin=.002):
        count=max(3,int(np.ceil(np.max(np.abs(b-a))/step))+1)
        minimum=2.
        for alpha in np.linspace(0,1,count):
            clearance,collision=self.query(xy,(1-alpha)*a+alpha*b)
            minimum=min(minimum,clearance)
            if clearance<margin or collision < -1e-8:
                return None
        return {"samples":count,"minimum_clearance":minimum,"max_joint_step":step}


def plan(reference,output):
    begin=time.perf_counter()
    config=json.loads((reference/"scene_config_snapshot.json").read_text())
    original=json.loads((reference/"delivery_trace.json").read_text())
    startrow=[r for r in original if r["phase"]=="lift_cup"][-1]
    old=[r for r in original if r["phase"]=="whole_body_carry_to_dropoff"]
    anchors=[startrow,old[20],old[44],old[-1]]
    postures=[np.array([r["base_q"][2],*r["arm_q"]]) for r in anchors]
    names=["抬杯姿态","过门姿态","房间内转向姿态","放置前姿态"]
    start=np.array(startrow["base_q"][:2]);goal=np.array(old[-1]["base_q"][:2])
    oracle=CarryOracle(config,startrow)
    lower=np.array(config["robot"]["lower_limits"][:2]);upper=np.array(config["robot"]["upper_limits"][:2])
    seeds=[start,goal]+[np.array(r["base_q"][:2]) for r in anchors]
    seeds += [np.array([x,y]) for x in np.arange(-2.6,1.41,.35) for y in np.arange(-.25,1.66,.35)]
    seeds += [np.array([x,y]) for x in np.arange(-.5,.71,.2) for y in [0.,.2]]
    seeds=list({tuple(np.round(c,8)):c for c in seeds}.values())
    covers=[];margin=.004
    for m,posture in enumerate(postures):
        disks=[]
        for xy in seeds:
            clearance,collision=oracle.query(xy,posture)
            r=min(.6,clearance-margin,np.min(xy-lower),np.min(upper-xy))
            if collision>=-1e-8 and r>=.035:
                disks.append({"center":xy.tolist(),"radius":float(r)})
        covers.append(disks)
        print(f"mode {m}: {len(disks)} disks",flush=True)
    def containing(m,xy):
        return [i for i,d in enumerate(covers[m]) if np.linalg.norm(xy-d["center"])<=d["radius"]-1e-7]
    transitions=[];attempts=0
    for a in range(len(postures)):
        for b in range(a+1,len(postures)):
            candidates=[xy for xy in seeds if containing(a,xy) and containing(b,xy)]
            # Limit redundant switches, retain spatial spread deterministically.
            selected=[]
            for xy in candidates:
                if all(np.linalg.norm(xy-c)>.42 for c in selected):selected.append(xy)
            for xy in selected:
                attempts+=1
                audit=oracle.switch(xy,postures[a],postures[b])
                if audit:
                    transitions.append({"from":a,"to":b,"xy":xy.tolist(),"audit":audit})
            print(f"switch {a}->{b}: {sum(t['from']==a and t['to']==b for t in transitions)} accepted",flush=True)
    print(f"{len(transitions)} switches / {attempts} candidates",flush=True)
    output.mkdir(parents=True,exist_ok=True)
    base={"mode_names":names,"postures":[p.tolist() for p in postures],"covers":covers,"transitions":transitions,"switch_attempts":attempts,"start":start.tolist(),"goal":goal.tolist(),"scope":"Finite posture library, forward mode transitions, sampled switch checks; no general completeness or continuous-time switch certificate."}
    (output/"graph.json").write_text(json.dumps(base))
    planner=GcsTrajectoryOptimization(2);groups=[];edges=[];metadata={}
    for m,disks in enumerate(covers):
        pairs=[(i,j) for i,a in enumerate(disks) for j,b in enumerate(disks) if i!=j and np.linalg.norm(np.array(a["center"])-b["center"])<a["radius"]+b["radius"]-.003]
        group=planner.AddRegions([Hyperellipsoid.MakeHypersphere(d["radius"],np.array(d["center"])) for d in disks],pairs,order=1,h_min=.01,h_max=20,name=f"mode{m}")
        groups.append(group);edges.extend(group.Edges())
        for i,v in enumerate(group.Vertices()):metadata[v.id()]={"kind":"disk","mode":m,"disk":i}
    for i,t in enumerate(transitions):
        xy=np.array(t["xy"])
        switch=planner.AddRegions([Point(xy)],order=0,name=f"switch{i}")
        switch.AddVertexCost(Expression(.1+.15*np.linalg.norm(postures[t["to"]]-postures[t["from"]])))
        metadata[switch.Vertices()[0].id()]={"kind":"switch","transition":i}
        edges.extend(planner.AddEdges(groups[t["from"]],switch,edges_between_regions=[(j,0) for j in containing(t["from"],xy)]).Edges())
        edges.extend(planner.AddEdges(switch,groups[t["to"]],edges_between_regions=[(0,j) for j in containing(t["to"],xy)]).Edges())
    src=planner.AddRegions([Point(start)],order=0,name="source");dst=planner.AddRegions([Point(goal)],order=0,name="target")
    metadata[src.Vertices()[0].id()]={"kind":"endpoint"};metadata[dst.Vertices()[0].id()]={"kind":"endpoint"}
    edges.extend(planner.AddEdges(src,groups[0],edges_between_regions=[(0,j) for j in containing(0,start)]).Edges())
    edges.extend(planner.AddEdges(groups[-1],dst,edges_between_regions=[(j,0) for j in containing(len(groups)-1,goal)]).Edges())
    planner.AddPathLengthCost(1.);planner.AddTimeCost(.01);planner.AddVelocityBounds(np.array([-.3,-.3]),np.array([.3,.3]))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10;options.rounding_seed=0
    options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver()
    preparation=time.perf_counter()-begin;begin=time.perf_counter()
    _,result=planner.SolvePath(src,dst,options);elapsed=time.perf_counter()-begin
    if not result.is_success():raise RuntimeError(f"Finite mode graph failed: {result.get_solution_result()}; graph saved for diagnosis")
    # Extract the integral rounded vertex path, not inferred nearest disks.
    outgoing={}
    for e in edges:
        if result.GetSolution(e.phi())>.5:outgoing.setdefault(e.u().id(),[]).append(e)
    vertex=src.Vertices()[0];path=[];seen=set()
    while vertex.id()!=dst.Vertices()[0].id():
        if vertex.id() in seen or len(outgoing.get(vertex.id(),[]))!=1:raise AssertionError("Rounded path is not a simple integral path")
        seen.add(vertex.id());path.append(vertex);vertex=outgoing[vertex.id()][0].v()
    path.append(vertex)
    samples=[];selected_switches=[];selected_disks=[]
    def append(xy,posture,mode,disk=None,switch=None):
        gripper,cup=oracle.state(xy,posture)
        samples.append({"phase":"whole_body_carry_to_dropoff","base_q":[*np.asarray(xy).tolist(),float(posture[0])],"arm_q":posture[1:].tolist(),"gripper_q":oracle.gripper,"gripper_pose":gripper.GetAsMatrix4().tolist(),"cup_pose":cup.GetAsMatrix4().tolist(),"attached":True,"cup_grasped":True,"carry_mode":mode,"carry_disk":disk,"carry_switch":switch,"origin":"new_carry_mode_disk_gcs"})
    append(start,postures[0],0)
    for vertex in path:
        meta=metadata[vertex.id()]
        if meta["kind"]=="disk":
            x=result.GetSolution(vertex.x())
            if len(x)!=5:raise AssertionError(f"Unexpected 2D order-1 vertex layout: {x}")
            a,b=x[:2],x[2:4];m=meta["mode"];disk=meta["disk"];d=covers[m][disk]
            if max(np.linalg.norm(a-d["center"]),np.linalg.norm(b-d["center"]))>d["radius"]+1e-5:raise AssertionError("Incorrect GCS control point extraction")
            selected_disks.append([m,disk])
            for u in np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(b-a)/.02))+1)):append((1-u)*a+u*b,postures[m],m,disk)
        elif meta["kind"]=="switch":
            ti=meta["transition"];t=transitions[ti];selected_switches.append(ti)
            for u in np.linspace(0,1,t["audit"]["samples"]):append(t["xy"],(1-u)*postures[t["from"]]+u*postures[t["to"]],None,switch=ti)
    q=lambda row:np.array([*row["base_q"],*row["arm_q"]])
    if np.linalg.norm(q(samples[-1])-np.array([*old[-1]["base_q"],*old[-1]["arm_q"]]))>1e-5:raise AssertionError("Goal mismatch")
    # Independently audit output and geometric grasp at a denser joint step.
    min_clearance=2.;checks=0
    for a,b in zip(samples,samples[1:]):
        qa,qb=q(a),q(b)
        for u in np.linspace(0,1,max(2,int(np.ceil(np.max(np.abs(qb-qa))/.01))+1)):
            point=(1-u)*qa+u*qb;clearance,collision=oracle.query(point[:2],point[2:]);checks+=1;min_clearance=min(min_clearance,clearance)
            if clearance<.002-1e-6 or collision < -1e-8:raise AssertionError("Final transport interpolation audit failed")
    audit_rows=[]
    for row in samples:
        oracle.state(row["base_q"][:2],np.array([row["base_q"][2],*row["arm_q"]]))
        audit_rows.append({"attached":True,"cup_gripper_audit":measure_cup_gripper(oracle.scene)})
    grasp=summarize_grasp_audit(audit_rows)
    if not grasp["satisfied"]:raise AssertionError(grasp["first_violation"])
    base.update(samples=samples,selected_disks=selected_disks,selected_switches=selected_switches,metrics={"prepare_seconds":preparation,"solve_seconds":elapsed,"disk_count":sum(map(len,covers)),"switch_candidates":attempts,"accepted_switches":len(transitions),"selected_switches":len(selected_switches),"audit_samples":checks,"minimum_environment_clearance":min_clearance,"base_path_length":float(sum(np.linalg.norm(np.array(a["base_q"][:2])-b["base_q"][:2]) for a,b in zip(samples,samples[1:]))),"old_base_path_length":float(sum(np.linalg.norm(np.array(a["base_q"][:2])-b["base_q"][:2]) for a,b in zip([startrow]+old,old)))},grasp_audit=grasp)
    (output/"result.json").write_text(json.dumps(base));print(json.dumps(base["metrics"],indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--reference",type=Path,default=Path("outputs/results/full_dynamics/reference"));p.add_argument("--output",type=Path,default=Path("outputs/results/carry_mode_gcs"));args=p.parse_args();plan(args.reference,args.output)
