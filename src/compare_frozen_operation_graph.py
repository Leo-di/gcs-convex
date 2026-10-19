"""Paired GCS vs minimum-hop route + official convex restriction.

Replays frozen IRIS covers. This is a route-selection ablation, not an
end-to-end planner benchmark or an independent physical execution trial.
"""
import argparse
from collections import deque
import hashlib
import heapq
import json
from pathlib import Path
import time

import numpy as np
from pydrake.geometry.optimization import HPolyhedron, Point, GraphOfConvexSetsOptions
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver


def minimum_hop_route(edges, sources, targets):
    adjacency = {}
    for a, b in sorted(edges):
        adjacency.setdefault(a, []).append(b)
    previous = {i: None for i in sorted(sources)}
    queue = deque(sorted(sources))
    target_set = set(targets)
    while queue:
        node = queue.popleft()
        if node in target_set:
            route = []
            while node is not None:
                route.append(node)
                node = previous[node]
            return route[::-1]
        for child in adjacency.get(node, []):
            if child not in previous:
                previous[child] = node
                queue.append(child)
    return None


def center_weighted_route(regions, edges, sources, targets, start, goal):
    """Dijkstra geometric proxy, followed by unrestricted portal optimization.

    Centers only score routes; they are NOT trajectory waypoints.
    """
    centers = [r.ChebyshevCenter() for r in regions]
    adjacency = {}
    for a, b in sorted(edges):
        adjacency.setdefault(a, []).append(b)
    distance = {i: float(np.linalg.norm(start-centers[i])) for i in sources}
    previous = {i: None for i in sources}
    queue = [(d, i) for i, d in distance.items()]
    heapq.heapify(queue)
    while queue:
        cost, node = heapq.heappop(queue)
        if cost > distance[node]:
            continue
        for child in adjacency.get(node, []):
            candidate = cost + float(np.linalg.norm(centers[node]-centers[child]))
            if candidate < distance.get(child, float('inf')):
                distance[child] = candidate
                previous[child] = node
                heapq.heappush(queue, (candidate, child))
    reachable = [i for i in targets if i in distance]
    if not reachable:
        return None
    node = min(reachable, key=lambda i: (distance[i]+np.linalg.norm(centers[i]-goal), i))
    route = []
    while node is not None:
        route.append(node)
        node = previous[node]
    return route[::-1]


def compare(folder, stage, methods=('gcs', 'minimum_hops_then_convex', 'center_weighted_then_convex')):
    folder = Path(folder)
    cover_file = folder / f'{stage}_cover.json'
    data = json.loads(cover_file.read_text())
    config = json.loads((folder / 'scene_config_snapshot.json').read_text())
    records = data['regions']
    regions = [HPolyhedron(np.array(r['A']), np.array(r['b'])) for r in records]
    start, goal = np.array(records[0]['guide_start']), np.array(records[-1]['guide_end'])
    rows = []
    for method in methods:
        tick = time.perf_counter()
        planner = GcsTrajectoryOptimization(7)
        cover = planner.AddRegions(regions, data['edges'], order=1, h_min=.01, h_max=20, name=stage)
        source = planner.AddRegions([Point(start)], order=0, name='start')
        target = planner.AddRegions([Point(goal)], order=0, name='goal')
        planner.AddEdges(source, cover, edges_between_regions=[(0,i) for i in data['sources']])
        planner.AddEdges(cover, target, edges_between_regions=[(i,0) for i in data['targets']])
        planner.AddPathLengthCost(1.)
        planner.AddTimeCost(.05)
        speed = config['whole_body_planning']['maximum_joint_speed']
        planner.AddVelocityBounds(np.full(7,-speed),np.full(7,speed))
        options = GraphOfConvexSetsOptions()
        options.convex_relaxation = True
        options.max_rounded_paths = 5
        options.rounding_seed = 0
        options.solver = ClarabelSolver()
        options.restriction_solver = ClarabelSolver()
        build_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        route = None
        if method == 'gcs':
            trajectory, result = planner.SolvePath(source,target,options)
        else:
            route = (minimum_hop_route(data['edges'],data['sources'],data['targets'])
                     if method == 'minimum_hops_then_convex' else
                     center_weighted_route(regions,data['edges'],data['sources'],data['targets'],start,goal))
            if route is None:
                rows.append({'method':method,'success':False,'reason':'disconnected graph'})
                continue
            vertices = [source.Vertices()[0]] + [cover.Vertices()[i] for i in route] + [target.Vertices()[0]]
            trajectory,result = planner.SolveConvexRestriction(vertices,options)
        row = {'method':method,'success':bool(result.is_success()),'solver_status':str(result.get_solution_result()),
               'build_seconds':build_seconds,'solve_seconds':time.perf_counter()-tick,'fixed_route':route}
        if result.is_success():
            segments=[]; length=0.;continuity=0.; previous=None; contained=True
            for i in range(trajectory.get_number_of_segments()):
                s=trajectory.segment(i)
                a=s.value(s.start_time()).ravel();b=s.value(s.end_time()).ravel()
                length+=float(np.linalg.norm(b-a))
                if previous is not None:
                    continuity=max(continuity,float(np.max(abs(previous-a))))
                previous=b
                contained &= any(r.PointInSet(a,1e-6) and r.PointInSet(b,1e-6) for r in regions)
                segments.append({'start':a.tolist(),'end':b.tolist(),'duration':s.end_time()-s.start_time()})
            endpoint=max(float(np.max(abs(np.array(segments[0]['start'])-start))),float(np.max(abs(previous-goal))))
            row.update(objective=float(result.get_optimal_cost()),path_length_rad=length,
                       duration_s=trajectory.end_time()-trajectory.start_time(),continuity_error_rad=continuity,
                       endpoint_error_rad=endpoint,segments_contained=bool(contained),segments=segments)
            row['numerically_valid']=bool(contained and max(endpoint,continuity)<1e-6)
        rows.append(row)
    return {'input':str(folder),'stage':stage,'cover_sha256':hashlib.sha256(cover_file.read_bytes()).hexdigest(),
            'region_count':len(regions),'edge_count':len(data['edges']),'results':rows,
            'scope':'Frozen guide-seeded IRIS cover; identical objectives and constraints; minimum-hop route ablation, not RRT baseline; no new physics execution; not independent success-rate trials'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidates',nargs='+',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for candidate in args.candidates:
        for stage in ('reach','lift'):
            print(candidate,stage,flush=True)
            rows.append(compare(candidate,stage))
            (out/'comparison.json').write_text(json.dumps(rows,indent=2))
            for r in rows[-1]['results']:
                print({k:v for k,v in r.items() if k not in ('segments','fixed_route')},flush=True)

if __name__=='__main__':
    main()
