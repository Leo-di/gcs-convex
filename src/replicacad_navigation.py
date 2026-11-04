"""Fixed-yaw GCS navigation using full imported geometry and clearance disks."""
import json
import time
import numpy as np
from pydrake.geometry.optimization import Hyperellipsoid, Point, GraphOfConvexSetsOptions
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver
from .scene_slice_gcs import SliceOracle
from .green_room_gcs import room_oracle


def plan(scene,start,goal,arm,artifact):
    began=time.perf_counter();start=np.asarray(start);goal=np.asarray(goal)
    if abs(start[2]-goal[2])>1e-10:raise ValueError('Fixed-yaw navigation requires equal endpoint yaw')
    oracle=SliceOracle(scene,arm,scene.gripper_open_positions,max_distance=.8)
    oracle.yaw=float(start[2]);room=room_oracle(scene);margin=.02
    def clearance(xy):
        d,sd,_=oracle.query(xy)
        if sd<0:return -1.
        return min(d,room.query(np.r_[xy,start[2],arm])['residual_m'])
    centers=[];radii=[]
    count=max(2,int(np.ceil(np.linalg.norm(goal[:2]-start[:2])/.04))+1)
    for alpha in np.linspace(0,1,count):
        xy=(1-alpha)*start[:2]+alpha*goal[:2]
        d=clearance(xy)
        if d<=margin:raise RuntimeError('Checked endpoint connection leaves the clearance slice')
        radius=min(.25,d-margin-1e-6)
        if centers and np.linalg.norm(xy-centers[-1])<.65*radii[-1] and alpha<1:continue
        centers.append(xy);radii.append(radius)
    regions=[Hyperellipsoid.MakeHypersphere(r,c) for c,r in zip(centers,radii)]
    edges=[(i,j) for i in range(len(regions)) for j in range(len(regions)) if i!=j
           and np.linalg.norm(centers[i]-centers[j])<radii[i]+radii[j]-1e-8]
    planner=GcsTrajectoryOptimization(2)
    free=planner.AddRegions(regions,edges,order=1,h_min=.01,h_max=20,name='replicacad_clearance_cover')
    src=planner.AddRegions([Point(start[:2])],order=0,name='start')
    dst=planner.AddRegions([Point(goal[:2])],order=0,name='goal')
    containing=lambda xy:[i for i,r in enumerate(regions) if r.PointInSet(xy,1e-9)]
    planner.AddEdges(src,free,edges_between_regions=[(0,i) for i in containing(start[:2])])
    planner.AddEdges(free,dst,edges_between_regions=[(i,0) for i in containing(goal[:2])])
    planner.AddPathLengthCost(1.);planner.AddTimeCost(.01)
    planner.AddVelocityBounds(np.full(2,-.3),np.full(2,.3))
    opts=GraphOfConvexSetsOptions();opts.convex_relaxation=True;opts.max_rounded_paths=10
    opts.rounding_seed=0;opts.solver=ClarabelSolver();opts.restriction_solver=ClarabelSolver()
    traj,result=planner.SolvePath(src,dst,opts)
    if not result.is_success():raise RuntimeError('Imported scene GCS solve failed')
    xy=np.array([traj.value(t).ravel() for t in np.linspace(0,traj.end_time(),201)])
    minimum=min(clearance(q) for q in xy)
    boundary=min(clearance(c+r*np.array([np.cos(a),np.sin(a)])) for c,r in zip(centers,radii)
                 for a in np.linspace(0,2*np.pi,12,endpoint=False))
    if min(minimum,boundary)<margin-1e-7:raise RuntimeError('Imported geometry rejected the GCS trajectory/cover')
    path=np.column_stack([xy,np.full(len(xy),start[2])])
    report=dict(satisfied=True,method='Official Drake GCS; fixed-yaw full-robot clearance disks',
        regions=len(regions),edges=len(edges),centers=[c.tolist() for c in centers],radii=radii,
        minimum_path_clearance_m=minimum,minimum_boundary_clearance_m=boundary,
        margin_m=margin,wall_seconds=time.perf_counter()-began,
        scope='Guide-seeded fixed-yaw slice; model-based translation bound and finite audits; reference replay')
    artifact.write_text(json.dumps(report,indent=2))
    return path,report
