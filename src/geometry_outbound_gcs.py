"""Current-geometry, fixed-posture outbound navigation using official Drake GCS.

Disks are a translation-only slice, clipped to the existing allowed lane boxes.
They are not configuration-space balls permitting yaw or arm motion.
"""
import time
import numpy as np
from pydrake.geometry.optimization import Hyperellipsoid,HPolyhedron,Intersection,Point,GraphOfConvexSetsOptions,CartesianProduct
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver
from .gcs_base_planner import GcsBasePlan,_label_region_sequence
from .scene_slice_gcs import SliceOracle


def make_cover(query,boxes,start,goal,lower,upper,spacing=.20,margin=.0201,max_radius=.65,refinement_spacing=None,refinement_radius_threshold=None,extra_seeds=None):
    seeds=[np.asarray(start),np.asarray(goal)]
    for seed in (() if extra_seeds is None else extra_seeds):
        seed=np.asarray(seed,float)
        if seed.shape!=(2,) or not np.all(np.isfinite(seed)):
            raise ValueError("Extra navigation seeds must be finite XY points")
        seeds.append(seed)
    for box in boxes:
        lo=np.maximum(box['lower'],lower);hi=np.minimum(box['upper'],upper)
        x_samples=np.arange(lo[0],hi[0]+spacing*.1,spacing)
        y_samples=np.arange(lo[1],hi[1]+spacing*.1,spacing)
        # Include the lane centerline as well as the lower-anchored lattice.
        # Otherwise a narrow lane can leave the endpoint disk isolated even
        # when a collision-free centerline corridor exists between samples.
        y_samples=np.unique(np.r_[y_samples,0.5*(lo[1]+hi[1])])
        seeds.extend(np.array([x,y]) for x in x_samples for y in y_samples)
    coarse_keys={tuple(np.round(xy,8)) for xy in seeds}
    if refinement_spacing is not None:
        if not 0 < refinement_spacing < spacing:
            raise ValueError('Refinement spacing must be positive and smaller than coarse spacing')
        for box in boxes:
            lo=np.maximum(box['lower'],lower);hi=np.minimum(box['upper'],upper)
            seeds.extend(np.array([x,y]) for x in np.arange(lo[0],hi[0]+1e-8,refinement_spacing)
                         for y in np.arange(lo[1],hi[1]+1e-8,refinement_spacing))
    seen=set();regions=[];records=[]
    for xy in seeds:
        key=tuple(np.round(xy,8))
        if key in seen:continue
        seen.add(key)
        distance,self_distance,_=query(xy)
        if self_distance < -1e-7:raise RuntimeError('Outbound fixed posture has self collision')
        radius=min(float(max_radius),float(distance)-margin)
        if radius<.025:continue
        # Fine seeds only in narrow-clearance areas. All coarse seeds and both
        # endpoints remain; graph connectivity is checked explicitly below.
        if key not in coarse_keys and radius >= (spacing if refinement_radius_threshold is None else refinement_radius_threshold):
            continue
        disk=Hyperellipsoid.MakeHypersphere(radius,xy)
        for box in boxes:
            lo=np.maximum(box['lower'],lower);hi=np.minimum(box['upper'],upper)
            if np.any(lo>hi) or np.linalg.norm(xy-np.clip(xy,lo,hi))>=radius:continue
            region=Intersection(disk,HPolyhedron.MakeBox(lo,hi))
            regions.append(region);records.append({'center':xy.tolist(),'radius':radius,'lower':lo.tolist(),'upper':hi.tolist(),'label':box['name']})
    return regions,records


def prune_contained_disks(regions, records):
    """Remove only sets provably contained in a retained disk/box set.

    Both disk containment and clipping-box containment are required. Thus the
    union is preserved (up to floating point), without approximating obstacles
    or replacing GCS by graph search. Large disks are considered first.
    """
    kept=[]
    for i in sorted(range(len(records)), key=lambda i: (-records[i]['radius'], i)):
        a=records[i]
        if any(
            (a.get('yaw_lo',0.) >= records[j].get('yaw_lo',0.))
            and (a.get('yaw_hi',0.) <= records[j].get('yaw_hi',0.))
            and np.all(np.asarray(a['lower']) >= np.asarray(records[j]['lower']))
            and np.all(np.asarray(a['upper']) <= np.asarray(records[j]['upper']))
            and np.linalg.norm(np.asarray(a['center'])-np.asarray(records[j]['center']))
                + a['radius'] <= records[j]['radius']
            for j in kept
        ):
            continue
        kept.append(i)
    kept.sort()
    return [regions[i] for i in kept], [records[i] for i in kept]


def clipped_disks_intersect(a,b):
    """Exact 2-D feasibility candidates: circle/box boundary intersections."""
    lo=np.maximum(a['lower'],b['lower']);hi=np.minimum(a['upper'],b['upper'])
    if np.any(lo>hi):return False
    centers=[np.array(a['center']),np.array(b['center'])];radii=[a['radius'],b['radius']]
    points=[*centers,*[np.array([x,y]) for x in [lo[0],hi[0]] for y in [lo[1],hi[1]]]]
    for c,r in zip(centers,radii):
        for axis in [0,1]:
            for v in [lo[axis],hi[axis]]:
                disc=r*r-(v-c[axis])**2
                if disc>=0:
                    for sign in [-1,1]:
                        q=c.copy();q[axis]=v;q[1-axis]+=sign*np.sqrt(disc);points.append(q)
    delta=centers[1]-centers[0];distance=np.linalg.norm(delta)
    if distance>1e-12 and abs(radii[0]-radii[1])<=distance<=sum(radii):
        t=(radii[0]**2-radii[1]**2+distance**2)/(2*distance);h=np.sqrt(max(0.,radii[0]**2-t*t));v=delta/distance;normal=np.array([-v[1],v[0]])
        points.extend([centers[0]+t*v+h*normal,centers[0]+t*v-h*normal])
    return any(np.all(q>=lo-1e-10) and np.all(q<=hi+1e-10) and all(np.linalg.norm(q-c)<=r+1e-10 for c,r in zip(centers,radii)) for q in points)


def select_navigation_arm(scene, base_positions):
    """Pick a compact empty-hand arm posture for base navigation.

    The home posture tilts the arm forward (shoulder +0.45, wrist +0.75), which
    makes iiwa_link_6 reach across the doorway plane and collide with the
    shared wall when the base detours above a relocated cart. These candidates
    keep the arm tucked; the first one that is collision-free (with margin) at
    every given base configuration wins, falling back to home. This is a
    finite deterministic search, not a posture certificate: the sampled
    outbound audit and the dense replay audit remain the gate.
    """
    configured = scene.config.get("pick_task", {}).get("navigation_arm_q")
    candidates = []
    if configured is not None:
        configured = np.asarray(configured, dtype=float)
        if configured.shape != (7,):
            raise ValueError("pick_task.navigation_arm_q must contain seven joint positions")
        candidates.append(configured)
    candidates.extend([
        np.array([0.0, -0.3, 0.0, -1.2, 0.0, 0.0, 0.0]),
        np.array([-0.5, -0.3, 0.0, -1.2, 0.0, 0.0, 0.0]),
        np.array([-0.5, -0.5, 0.0, -1.4, 0.0, 0.0, 0.0]),
        np.array([0.0, -0.7, 0.0, -1.4, 0.0, 0.0, 0.0]),
    ])
    for arm in candidates:
        # Only the 3 cm acceptance threshold matters here; all penetrations
        # remain included when Drake is queried with this positive cutoff.
        oracle = SliceOracle(scene, arm, scene.gripper_open_positions, max_distance=.03)
        ok = True
        for q in base_positions:
            q = np.asarray(q, float)
            oracle.yaw = float(q[2])
            distance, self_distance, _ = oracle.query(q[:2])
            if distance < 0.03 or self_distance < 0.0:
                ok = False
                break
        if ok:
            return arm
    return scene.arm_home_positions


def plan_geometry_outbound(scene,start,goal,*,allowed_region_names,arm_q=None,spacing=.30,max_radius=.65,prune_contained=False,refinement_spacing=None,refinement_radius_threshold=None,extra_seeds=None):
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    if abs(start[2]-goal[2])>1e-10:raise RuntimeError('Geometry outbound slice requires identical start/goal yaw')
    config=scene.config;spec=config['gcs_base_planning'];boxes=[r for r in spec['regions'] if r['name'] in allowed_region_names and r['yaw_bounds'][0]<=start[2]<=r['yaw_bounds'][1]]
    if not boxes:raise RuntimeError('No allowed semantic lane at the fixed yaw')
    arm=scene.arm_home_positions if arm_q is None else np.asarray(arm_q,float)
    # The cover caps radii at .65 m and subtracts a .0201 m margin.  Distances
    # above .6701 m cannot change any generated region, so bound the Drake
    # query at that exact threshold to avoid materializing irrelevant pairs.
    margin=.0201
    oracle=SliceOracle(scene,arm,scene.gripper_open_positions,yaw=float(start[2]),max_distance=float(max_radius)+margin);begin=time.perf_counter()
    regions,records=make_cover(oracle.query,boxes,start[:2],goal[:2],np.array(config['robot']['lower_limits'][:2]),np.array(config['robot']['upper_limits'][:2]),spacing=spacing,margin=margin,max_radius=max_radius,refinement_spacing=refinement_spacing,refinement_radius_threshold=refinement_radius_threshold,extra_seeds=extra_seeds)
    raw_region_count=len(regions)
    if prune_contained:
        regions,records=prune_contained_disks(regions,records)
    print(f'Navigation cover: {raw_region_count} raw, {len(regions)} retained regions',flush=True)
    sources=[i for i,r in enumerate(regions) if r.PointInSet(start[:2],1e-9)];targets=[i for i,r in enumerate(regions) if r.PointInSet(goal[:2],1e-9)]
    if not sources or not targets:raise RuntimeError('Current-geometry outbound cover does not cover start or goal')
    # Geometric intersection pruning precedes official GCS optimization.
    edges=[]
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            if np.linalg.norm(np.array(a['center'])-b['center'])>a['radius']+b['radius']:continue
            if np.any(np.maximum(a['lower'],b['lower'])>np.minimum(a['upper'],b['upper'])):continue
            if clipped_disks_intersect(a,b):edges.extend([(i,j),(j,i)])
    adjacency={i:[] for i in range(len(regions))}
    for i,j in edges:adjacency[i].append(j)
    reached=set(sources);stack=list(sources)
    while stack:
        for j in adjacency[stack.pop()]:
            if j not in reached:reached.add(j);stack.append(j)
    if not reached.intersection(targets):raise RuntimeError('Current-geometry fixed-posture lane cover is disconnected; no whole-body infeasibility claim')
    planner=GcsTrajectoryOptimization(2);free=planner.AddRegions(regions,edges,order=1,h_min=.01,h_max=20,name='current_geometry_outbound')
    src=planner.AddRegions([Point(start[:2])],order=0,name='start');dst=planner.AddRegions([Point(goal[:2])],order=0,name='pickup')
    planner.AddEdges(src,free,edges_between_regions=[(0,i) for i in sources]);planner.AddEdges(free,dst,edges_between_regions=[(i,0) for i in targets]);planner.AddPathLengthCost(float(spec['path_length_weight']));planner.AddTimeCost(float(spec['time_weight']));speed=float(spec['maximum_speed']);planner.AddVelocityBounds(np.full(2,-speed),np.full(2,speed))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10;options.rounding_seed=0;options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver();prepared=time.perf_counter()-begin;tick=time.perf_counter();trajectory,result=planner.SolvePath(src,dst,options);elapsed=time.perf_counter()-tick
    if not result.is_success():raise RuntimeError('Official GCS found no path in the current-geometry fixed-posture lane cover; not a task infeasibility proof')
    duration=float(trajectory.end_time());times=np.linspace(0,duration,max(1001,int(np.ceil(duration*spec['samples_per_second']))+1));xy=np.array([trajectory.value(t).ravel() for t in times]);xy[0]=start[:2];xy[-1]=goal[:2]
    distances=[oracle.query(x) for x in xy];minimum=min(d[0] for d in distances)
    if minimum<.02-1e-6 or min(d[1] for d in distances)<-1e-7:raise RuntimeError('Current-geometry outbound trajectory failed independent numerical audit')
    qs=np.column_stack([xy,np.full(len(xy),start[2])]);plan=GcsBasePlan(qs,elapsed,duration,float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum()),0.,float(result.get_optimal_cost()),result.get_solver_id().name(),trajectory.get_number_of_segments(),_label_region_sequence(qs,boxes))
    return plan,{'algorithm':'Drake GcsTrajectoryOptimization on current-geometry disks intersected with allowed lane boxes','fixed_arm_q':arm.tolist(),'fixed_yaw':float(start[2]),'region_count':len(regions),'raw_region_count':raw_region_count,'prune_contained':bool(prune_contained),'refinement_spacing':refinement_spacing,'refinement_radius_threshold':refinement_radius_threshold,'spacing':float(spacing),'margin_m':.0201,'prepare_seconds':prepared,'solve_seconds':elapsed,'audit_samples':len(xy),'minimum_distance_m':minimum,'oracle_calls':oracle.calls,'scope':'Fixed-posture translation slice, original semantic lanes retained; failure does not prove whole-body infeasibility.'}


def _se2_yaw_slices(boxes,start_yaw,goal_yaw,yaw_step,yaw_window):
    cmin=min(start_yaw,goal_yaw);cmax=max(start_yaw,goal_yaw)
    lo=max(min(b['yaw_bounds'][0] for b in boxes),cmin-yaw_window)
    hi=min(max(b['yaw_bounds'][1] for b in boxes),cmax+yaw_window)
    if lo>hi:lo=hi=cmin
    slices={round(float(start_yaw),4),round(float(goal_yaw),4)}
    slices.update(round(float(v),4) for v in np.arange(lo,hi+yaw_step*.5,yaw_step))
    return sorted(slices)


def plan_geometry_outbound_se2(scene,start,goal,*,allowed_region_names,yaw_step=.6,spacing=.35,yaw_window=.9,margin=.0201,arm_q=None,prune_contained=False,cover_artifact=None):
    """SE(2) current-geometry outbound navigation.

    Extends the fixed-yaw translation slice by slicing yaw so the base may
    rotate in place (or while translating) to route around obstacles such as a
    relocated cart. Each region is the Cartesian product of a current-geometry
    XY disk (clipped to its semantic lane box) and a yaw interval. The disk
    radius is sized from signed distance at the central yaw, not the entire
    interval; safety of the entire product under rotation is unproven. A finite
    sample audit at each waypoint's actual yaw remains the gate; this is not a
    continuous-time safety certificate.
    """
    start=np.asarray(start,float);goal=np.asarray(goal,float)
    config=scene.config;spec=config['gcs_base_planning']
    boxes=[r for r in spec['regions'] if r['name'] in allowed_region_names]
    if not boxes:raise RuntimeError('No allowed semantic lane')
    lower=np.array(config['robot']['lower_limits'][:2]);upper=np.array(config['robot']['upper_limits'][:2])
    slices=_se2_yaw_slices(boxes,start[2],goal[2],yaw_step,yaw_window)
    arm=scene.arm_home_positions if arm_q is None else np.asarray(arm_q,float)
    oracle=SliceOracle(scene,arm,scene.gripper_open_positions,yaw=float(start[2]),max_distance=.6701);begin=time.perf_counter()
    cache={}
    def dist_at(xy,yaw):
        key=(round(float(xy[0]),5),round(float(xy[1]),5),round(float(yaw),4))
        got=cache.get(key)
        if got is None:
            oracle.yaw=yaw;got=oracle.query(xy);cache[key]=got
        return got
    regions=[];records=[]
    # Doorway refinement plus a fine symmetric grid over the whole doorway
    # approach corridor, including the x-range a relocated cart can occupy and
    # both the above-cart and below-cart detour bands. Without this, disks in
    # the tight gap shrink below the coarse grid spacing and the cover tears.
    refine=[np.array([x,y]) for x in np.arange(-.8,.81,.2) for y in [-.2,0.,.2]]
    refine+=[np.array([x,y]) for x in np.arange(-1.7,.01,.1) for y in np.arange(-.35,.56,.1)]
    for box in boxes:
        blo,bhi=box['yaw_bounds']
        lo=np.maximum(box['lower'],lower);hi=np.minimum(box['upper'],upper)
        if np.any(lo>hi):continue
        box_seeds=[np.array([x,y]) for x in np.arange(lo[0],hi[0]+spacing*.1,spacing) for y in np.arange(lo[1],hi[1]+spacing*.1,spacing)]
        box_seeds += [start[:2],goal[:2]]+refine
        for psi in slices:
            if psi<blo-1e-9 or psi>bhi+1e-9:continue
            y_lo=max(blo,psi-yaw_step);y_hi=min(bhi,psi+yaw_step)
            if y_hi-y_lo<1e-9:continue
            for xy in box_seeds:
                if np.linalg.norm(xy-np.clip(xy,lo,hi))>=.65:continue
                d,sd,_=dist_at(xy,float(psi))
                if sd<-1e-7:continue
                radius=min(.65,d-margin,float(np.min(xy-lower)),float(np.min(upper-xy)))
                if radius<.025:continue
                if np.linalg.norm(xy-np.clip(xy,lo,hi))>=radius:continue
                xy_set=Intersection(Hyperellipsoid.MakeHypersphere(radius,xy),HPolyhedron.MakeBox(lo,hi))
                yaw_set=HPolyhedron.MakeBox(np.array([y_lo]),np.array([y_hi]))
                regions.append(CartesianProduct([xy_set,yaw_set]))
                records.append({'center':xy.tolist(),'radius':radius,'lower':lo.tolist(),'upper':hi.tolist(),'label':box['name'],'yaw_lo':float(y_lo),'yaw_hi':float(y_hi),'slice':float(psi)})
    raw_region_count=len(regions)
    if prune_contained:
        regions,records=prune_contained_disks(regions,records)
    if cover_artifact is not None:
        import json
        from pathlib import Path
        Path(cover_artifact).write_text(json.dumps({'raw_region_count':raw_region_count,'region_count':len(regions),'regions':records,'start':start.tolist(),'goal':goal.tolist()},indent=2))
    t_cover=time.perf_counter()
    sources=[i for i,r in enumerate(regions) if r.PointInSet(start,1e-9)];targets=[i for i,r in enumerate(regions) if r.PointInSet(goal,1e-9)]
    if not sources or not targets:raise RuntimeError('SE(2) outbound cover does not cover start or goal')
    edges=[]
    for i,a in enumerate(records):
        for j in range(i):
            b=records[j]
            if max(a['yaw_lo'],b['yaw_lo'])>min(a['yaw_hi'],b['yaw_hi'])-1e-10:continue
            if np.linalg.norm(np.array(a['center'])-b['center'])>a['radius']+b['radius']:continue
            if np.any(np.maximum(a['lower'],b['lower'])>np.minimum(a['upper'],b['upper'])):continue
            if clipped_disks_intersect(a,b):edges.extend([(i,j),(j,i)])
    adjacency={i:[] for i in range(len(regions))}
    for i,j in edges:adjacency[i].append(j)
    reached=set(sources);stack=list(sources)
    while stack:
        for j in adjacency[stack.pop()]:
            if j not in reached:reached.add(j);stack.append(j)
    if not reached.intersection(targets):raise RuntimeError('SE(2) current-geometry lane cover is disconnected; no whole-body infeasibility claim')
    t_edges=time.perf_counter()
    planner=GcsTrajectoryOptimization(3);free=planner.AddRegions(regions,edges,order=1,h_min=.01,h_max=20,name='se2_outbound')
    src=planner.AddRegions([Point(start)],order=0,name='start');dst=planner.AddRegions([Point(goal)],order=0,name='pickup')
    planner.AddEdges(src,free,edges_between_regions=[(0,i) for i in sources]);planner.AddEdges(free,dst,edges_between_regions=[(i,0) for i in targets])
    planner.AddPathLengthCost(np.diag([float(spec['path_length_weight']),float(spec['path_length_weight']),float(spec['yaw_path_weight'])]));planner.AddTimeCost(float(spec['time_weight']))
    speed=float(spec['maximum_speed']);ang=float(spec['maximum_angular_speed']);planner.AddVelocityBounds(np.array([-speed,-speed,-ang]),np.array([speed,speed,ang]))
    options=GraphOfConvexSetsOptions();options.convex_relaxation=True;options.max_rounded_paths=10;options.rounding_seed=0;options.solver=ClarabelSolver();options.restriction_solver=ClarabelSolver();prepared=time.perf_counter()-begin;tick=time.perf_counter();trajectory,result=planner.SolvePath(src,dst,options);elapsed=time.perf_counter()-tick
    if not result.is_success():raise RuntimeError('Official GCS found no path in the SE(2) current-geometry lane cover; not a task infeasibility proof')
    duration=float(trajectory.end_time());times=np.linspace(0,duration,max(1001,int(np.ceil(duration*spec['samples_per_second']))+1));qs=np.array([trajectory.value(t).ravel() for t in times]);qs[0]=start;qs[-1]=goal
    distances=[];self_dists=[]
    for q in qs:
        d,sd,_=dist_at(q[:2],float(q[2]));distances.append(d);self_dists.append(sd)
    minimum=min(distances)
    if minimum<.02-1e-6 or min(self_dists)<-1e-7:raise RuntimeError('SE(2) outbound trajectory failed independent numerical audit')
    plan=GcsBasePlan(qs,elapsed,duration,float(np.linalg.norm(np.diff(qs[:,:2],axis=0),axis=1).sum()),float(np.abs(np.diff(qs[:,2])).sum()),float(result.get_optimal_cost()),result.get_solver_id().name(),trajectory.get_number_of_segments(),_label_region_sequence(qs,boxes))
    return plan,{'algorithm':'Drake GcsTrajectoryOptimization on SE(2) current-geometry disks (XY x yaw interval) intersected with allowed lane boxes','fixed_arm_q':arm.tolist(),'yaw_slices':slices,'region_count':len(regions),'raw_region_count':raw_region_count,'prune_contained':prune_contained,'spacing':spacing,'yaw_step':yaw_step,'margin_m':margin,'prepare_seconds':prepared,'solve_seconds':elapsed,'cover_seconds':t_cover-begin,'edge_seconds':t_edges-t_cover,'audit_samples':len(qs),'minimum_distance_m':minimum,'oracle_calls':oracle.calls,'scope':'SE(2) fixed-arm-posture candidate cover; disks sized at central yaw, not certified over the yaw interval; finite trajectory audit required. Failure does not prove whole-body infeasibility.'}
