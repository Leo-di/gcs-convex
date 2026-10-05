"""Safe XY disks from full 3D Drake distances, at a fixed empty-arm pose."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry.optimization import Hyperellipsoid, Point, GraphOfConvexSetsOptions
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver

from .build_scene import build_scene, set_robot_configuration
from .planar_bubble_demo import graph_connects


class SliceOracle:
    def __init__(self, scene, arm, gripper, yaw=0.):
        self.scene, self.arm, self.gripper, self.yaw = scene, arm, gripper, yaw
        self.robot = {scene.base_instance, scene.arm_instance, scene.gripper_instance}
        self.calls = 0

    def query(self, xy):
        self.calls += 1
        s = self.scene
        set_robot_configuration(s, [*xy, self.yaw], self.arm, self.gripper)
        query = s.scene_graph.get_query_output_port().Eval(s.scene_graph.GetMyContextFromRoot(s.root_context))
        inspector = query.inspector()
        environment, self_distance = 2., float("inf")
        closest = None
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(2.):
            bodies = [s.plant.GetBodyFromFrameId(inspector.GetFrameId(g)) for g in (pair.id_A, pair.id_B)]
            count = sum(b.model_instance() in self.robot for b in bodies)
            if count == 1 and pair.distance < environment:
                environment = float(pair.distance)
                closest = [b.name() for b in bodies]
            elif count == 2:
                self_distance = min(self_distance, float(pair.distance))
        return environment, self_distance, closest


def run(source, output):
    begin = time.perf_counter()
    config = json.loads((source/"config.json").read_text())
    original = json.loads((source/"result.json").read_text())["trace"]
    first = original[0]
    # Original task's navigation endpoints; arm/gripper held at initial values.
    start = np.array(first["base_q"][:2])
    goal = np.array([f for f in original if f["phase"] == "navigate_to_pickup"][-1]["base_q"][:2])
    scene = build_scene(config=config, visualize=True)
    yaw = float(first["base_q"][2])
    oracle = SliceOracle(scene, first["arm_q"], first["gripper_q"], yaw)
    lower = np.array(config["robot"]["lower_limits"][:2])
    upper = np.array(config["robot"]["upper_limits"][:2])
    margin = .02
    seeds = [start, goal]
    seeds += [np.array([x, y]) for x in np.arange(lower[0]+.2, upper[0], .4)
              for y in np.arange(lower[1]+.2, upper[1], .4)]
    # Doorway refinement in original scene coordinates; not a forced path.
    seeds += [np.array([x, y]) for x in np.arange(-.8, .81, .2) for y in [-.2, 0., .2]]
    centers, radii, evidence = [], [], []
    for xy in seeds:
        if any(np.linalg.norm(xy-c)<1e-7 for c in centers):
            continue
        distance, self_distance, closest = oracle.query(xy)
        if self_distance < 0:
            raise ValueError("Fixed posture has self-collision")
        radius = min(.65, distance-margin, np.min(xy-lower), np.min(upper-xy))
        if radius >= .065:
            centers.append(xy); radii.append(radius)
            evidence.append({"environment_distance": distance, "self_distance": self_distance, "closest_bodies": closest})
    centers, radii = np.array(centers), np.array(radii)
    edges = [(i,j) for i in range(len(centers)) for j in range(len(centers))
             if i!=j and np.linalg.norm(centers[i]-centers[j]) < radii[i]+radii[j]-.005]
    if not graph_connects((centers,radii,edges), start, goal):
        raise RuntimeError("Slice cover is disconnected; this does not prove task infeasibility")
    planner = GcsTrajectoryOptimization(2)
    regions = planner.AddRegions([Hyperellipsoid.MakeHypersphere(r,c) for c,r in zip(centers,radii)], edges, order=1, h_min=.01,h_max=20,name="original_scene_xy_slice")
    src=planner.AddRegions([Point(start)],order=0,name="start")
    dst=planner.AddRegions([Point(goal)],order=0,name="pickup_approach")
    containing=lambda xy:[i for i,(c,r) in enumerate(zip(centers,radii)) if np.linalg.norm(xy-c)<=r]
    planner.AddEdges(src,regions,edges_between_regions=[(0,i) for i in containing(start)])
    planner.AddEdges(regions,dst,edges_between_regions=[(i,0) for i in containing(goal)])
    planner.AddPathLengthCost(1.);planner.AddTimeCost(.01)
    planner.AddVelocityBounds(np.array([-.3,-.3]),np.array([.3,.3]))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10
    options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver();options.rounding_seed=0
    preparation=time.perf_counter()-begin;begin=time.perf_counter()
    trajectory,result=planner.SolvePath(src,dst,options)
    elapsed=time.perf_counter()-begin
    if not result.is_success():raise RuntimeError(str(result.get_solution_result()))
    # Numerical audit of disk boundaries against full 3D geometry.
    boundary_min=2.
    for c,r in zip(centers,radii):
        for a in np.linspace(0,2*np.pi,16,endpoint=False):
            distance,_,_=oracle.query(c+r*np.array([np.cos(a),np.sin(a)]))
            boundary_min=min(boundary_min,distance)
    if boundary_min < margin-1e-6:raise AssertionError("Disk boundary audit failed")
    segments=[]
    for i in range(trajectory.get_number_of_segments()):
        seg=trajectory.segment(i);a=seg.value(seg.start_time()).ravel();b=seg.value(seg.end_time()).ravel()
        matches=[j for j,(c,r) in enumerate(zip(centers,radii)) if max(np.linalg.norm(a-c),np.linalg.norm(b-c))<=r+1e-6]
        if not matches:raise AssertionError("Segment outside all disks")
        segments.append({"a":a.tolist(),"b":b.tolist(),"disk":matches[0],"start":seg.start_time(),"end":seg.end_time()})
    links=[scene.plant.GetBodyByName(f"iiwa_link_{i}",scene.arm_instance) for i in range(8)]
    frames=[];scene.meshcat.StartRecording()
    for t in np.linspace(0,trajectory.end_time(),max(2,int(trajectory.end_time()*30)+1)):
        xy=trajectory.value(t).ravel();distance,self_distance,_=oracle.query(xy)
        if distance<margin-1e-6 or self_distance<0:raise AssertionError("3D path audit failed")
        scene.root_context.SetTime(t);scene.diagram.ForcedPublish(scene.root_context)
        xyz=lambda b:scene.plant.EvalBodyPoseInWorld(scene.plant_context,b).translation().tolist()
        disk=next(s["disk"] for s in segments if s["start"]-1e-8<=t<=s["end"]+1e-8)
        frames.append({"time_s":float(t),"base_q":[*xy.tolist(),yaw],"links_xyz":[xyz(b) for b in links],"gripper_xyz":xyz(scene.gripper_body),"cup_pose":scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.cup_body).GetAsMatrix4().tolist(),"environment_clearance":distance,"disk":disk})
    scene.meshcat.StopRecording();scene.meshcat.PublishRecording()
    objects=[{"name":w["name"],"kind":"wall",**w} for w in config["world"]["walls"]]
    objects += [{"name":n,"kind":"furniture",**v} for n,v in config["objects"].items() if isinstance(v,dict) and "size" in v and "center" in v]
    objects += [{"kind":"distractor",**v} for v in config["objects"].get("distractors",[])]
    data={"floor":config["world"]["floor"],"cup":config["objects"]["cup"],"frames":frames,"segments":segments,"centers":centers.tolist(),"radii":radii.tolist(),"edges":edges,"evidence":evidence,"objects":objects,"start":start.tolist(),"goal":goal.tolist(),"base_size":config["robot"]["size"],"fixed_arm_q":first["arm_q"],"fixed_gripper_q":first["gripper_q"],"fixed_yaw":yaw,"margin":margin,"duration_s":float(trajectory.end_time()),"source":str(source),"config_sha256":hashlib.sha256((source/"config.json").read_bytes()).hexdigest(),"metrics":{"regions":len(centers),"directed_edges":len(edges),"prepare_seconds":preparation,"solve_seconds":elapsed,"boundary_samples":len(centers)*16,"boundary_min_clearance":boundary_min,"path_samples":len(frames),"path_min_clearance":min(f["environment_clearance"] for f in frames),"query_calls":oracle.calls},"scope":"Empty fixed-posture XY slice; original 3D collision geometry and filters; new GCS kinematic replay, not the previous dynamics trajectory or complete manipulation."}
    output.mkdir(parents=True,exist_ok=True)
    (output/"result.json").write_text(json.dumps(data));(output/"config.json").write_text(json.dumps(config,indent=2))
    (output/"scene.html").write_text(scene.meshcat.StaticHtml())
    (output/"index.html").write_text(Path(__file__).with_name("scene_slice_view.html").read_text().replace("__DATA__",json.dumps(data)))
    print(json.dumps(data["metrics"],indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("outputs/results/full_dynamics/validated"))
    parser.add_argument("--output",type=Path,default=Path("outputs/results/grasp_fix_web/planar/slice"))
    args=parser.parse_args();run(args.source,args.output)
