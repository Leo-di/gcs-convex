"""Research reproduction: repair a disconnected doorway IRIS graph.

This experiment applies the DirectLine -> LocalRRTC fallback -> binary
IRIS-NP2 inflation pattern from arXiv:2608.22326 to the sparse free-cup reach
layer. It does not modify the production semantic GCS planner.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import deque
from pathlib import Path

import numpy as np
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, Point
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver

from .build_scene import build_scene, set_robot_configuration
from .geometry_outbound_gcs import select_navigation_arm
from .grasp_planner import plan_pick
from .manipulation_gcs import (
    _manipulation_spec,
    _phase_domain,
    _site_region_spec,
    prepare_mode_machinery,
)
from .iris_region_generator import _grow_cover_region
from .online_carry_bridge import closest_points
from .scene_spec import load_scene_config


def _components(regions):
    unseen = set(range(len(regions)))
    result = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        comp = {root}
        queue = [root]
        while queue:
            i = queue.pop()
            for j in tuple(unseen):
                if regions[i].IntersectsWith(regions[j]):
                    unseen.remove(j)
                    comp.add(j)
                    queue.append(j)
        result.append(comp)
    return result


def _shortest_region_chain(regions, start, goal):
    starts = [i for i, r in enumerate(regions) if r.PointInSet(start, 1e-7)]
    goals = {i for i, r in enumerate(regions) if r.PointInSet(goal, 1e-7)}
    parents = {i: None for i in starts}
    queue = deque(starts)
    target = None
    while queue:
        i = queue.popleft()
        if i in goals:
            target = i
            break
        for j, r in enumerate(regions):
            if j not in parents and regions[i].IntersectsWith(r):
                parents[j] = i
                queue.append(j)
    if target is None:
        return None
    chain = []
    while target is not None:
        chain.append(target)
        target = parents[target]
    return chain[::-1]


def _valid_witness(q, seed, machinery):
    if machinery.checker.CheckConfigCollisionFree(machinery.parameterization(q)):
        return np.asarray(q, float)
    # Keep the point in its convex region while moving it toward the known
    # collision-free IRIS seed, as in the paper's witness validation step.
    for alpha in np.linspace(.02, 1., 50):
        candidate = (1 - alpha) * q + alpha * seed
        if machinery.checker.CheckConfigCollisionFree(
            machinery.parameterization(candidate)
        ):
            return candidate
    return None


def _local_rrtc(a, b, machinery, config, site_spec, *, random_seed=260810):
    """Bounded bidirectional RRT-Connect in the paper's local C-space box."""
    n = len(a)
    arm_lower = np.asarray(config['whole_body_planning']['joint_lower_limits'], float)
    arm_upper = np.asarray(config['whole_body_planning']['joint_upper_limits'], float)
    global_lower = np.r_[np.asarray(site_spec['lower'], float), arm_lower]
    global_upper = np.r_[np.asarray(site_spec['upper'], float), arm_upper]
    widths = global_upper - global_lower
    delta = np.abs(b - a)
    distance = float(np.linalg.norm(delta))
    rho = np.minimum(.1 * widths, np.maximum.reduce([
        np.full(n, distance), .5 * delta, .005 * widths,
    ]))
    lower = np.maximum(global_lower, np.minimum(a, b) - rho)
    upper = np.minimum(global_upper, np.maximum(a, b) + rho)
    lower = np.maximum(lower, global_lower)
    upper = np.minimum(upper, global_upper)
    rng = np.random.default_rng(random_seed)
    checker = machinery.checker
    parameterization = machinery.parameterization
    step = .22
    max_iterations = 6000
    restarts = 6
    collision_queries = 0

    def valid_edge(x, y):
        nonlocal collision_queries
        collision_queries += 1
        return checker.CheckEdgeCollisionFree(parameterization(x), parameterization(y))

    def extend(tree, target):
        nodes, parents = tree
        dists = np.linalg.norm(np.asarray(nodes) - target, axis=1)
        near = int(np.argmin(dists))
        d = float(dists[near])
        nxt = nodes[near] + (target - nodes[near]) * min(1., step / max(d, 1e-12))
        if not valid_edge(nodes[near], nxt):
            return None, d <= step
        nodes.append(nxt)
        parents.append(near)
        return len(nodes) - 1, d <= step

    def trace(tree, index):
        nodes, parents = tree
        path = []
        while index is not None:
            path.append(nodes[index])
            index = parents[index]
        return path[::-1]

    def connect(tree, target):
        last = None
        while True:
            idx, reached = extend(tree, target)
            if idx is None:
                return last, False
            last = idx
            if reached:
                return last, True

    for restart in range(restarts):
        ta = ([a.copy()], [None])
        tb = ([b.copy()], [None])
        for iteration in range(max_iterations):
            if rng.random() < .08:
                sample = b.copy() if iteration % 2 == 0 else a.copy()
            else:
                sample = rng.uniform(lower, upper)
            side_a = iteration % 2 == 0
            first, second = (ta, tb) if side_a else (tb, ta)
            new, _ = extend(first, sample)
            if new is None:
                continue
            connected, reached = connect(second, first[0][new])
            if reached:
                p_first = trace(first, new)
                p_second = trace(second, connected)
                path = p_first + p_second[-2::-1]
                if side_a:
                    path_a_to_b = path
                else:
                    path_a_to_b = path[::-1]
                return path_a_to_b, {
                    'restarts': restart + 1,
                    'iterations': iteration + 1,
                    'collision_edge_queries': collision_queries,
                    'local_box_lower': lower.tolist(),
                    'local_box_upper': upper.tolist(),
                }
    return None, {
        'restarts': restarts,
        'iterations_per_restart': max_iterations,
        'collision_edge_queries': collision_queries,
        'local_box_lower': lower.tolist(),
        'local_box_upper': upper.tolist(),
    }


def _path_at_s(path, s):
    path = np.asarray(path, float)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    if arc[-1] <= 1e-12:
        return path[0].copy()
    target = float(s) * arc[-1]
    i = min(int(np.searchsorted(arc, target, side='right') - 1), len(path) - 2)
    alpha = (target - arc[i]) / max(arc[i + 1] - arc[i], 1e-12)
    return (1 - alpha) * path[i] + alpha * path[i + 1]


def _solve_gcs(regions, start, goal, config, machinery):
    edges = []
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if regions[i].IntersectsWith(regions[j]):
                edges.extend([(i, j), (j, i)])
    spec = config['gcs_base_planning']
    whole = config['whole_body_planning']
    planner = GcsTrajectoryOptimization(10, continuous_revolute_joints=[2])
    order = int(spec['bezier_order'])
    free = planner.AddRegions(regions, edges, order=order, h_min=.05,
                              h_max=30., name='gcs_bridged_free_cspace')
    source = planner.AddRegions([Point(start)], order=0, name='sparse_semantic_start')
    target = planner.AddRegions([Point(goal)], order=0, name='sparse_semantic_grasp')
    start_regions = [i for i, r in enumerate(regions) if r.PointInSet(start, 1e-7)]
    goal_regions = [i for i, r in enumerate(regions) if r.PointInSet(goal, 1e-7)]
    planner.AddEdges(source, free,
                     edges_between_regions=[(0, i) for i in start_regions])
    planner.AddEdges(free, target,
                     edges_between_regions=[(i, 0) for i in goal_regions])
    metric = np.diag([float(spec['path_length_weight'])] * 2 +
                     [float(spec['yaw_path_weight'])] +
                     [float(whole['joint_path_weight'])] * 7)
    planner.AddPathLengthCost(metric)
    planner.AddTimeCost(float(spec['time_weight']))
    velocity = np.asarray([float(spec['maximum_speed'])] * 2 +
                          [float(spec['maximum_angular_speed'])] +
                          [float(whole['maximum_joint_speed'])] * 7)
    planner.AddVelocityBounds(-velocity, velocity)
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.max_rounded_paths = 10
    options.rounding_seed = 0
    options.solver = ClarabelSolver()
    options.restriction_solver = ClarabelSolver()
    solve_start = time.perf_counter()
    trajectory, result = planner.SolvePath(source, target, options)
    solve_seconds = time.perf_counter() - solve_start
    if not result.is_success():
        return None, {'success': False, 'solution_result': str(result.get_solution_result()),
                      'edge_count': len(edges)}
    duration = float(trajectory.end_time() - trajectory.start_time())
    sample_times = np.linspace(trajectory.start_time(), trajectory.end_time(),
                               max(201, int(np.ceil(duration * 100)) + 1))
    collision_bad = 0
    previous = None
    sampled_q = []
    ee_x = []
    for t in sample_times:
        q = np.asarray(trajectory.value(t), float).reshape(-1)
        sampled_q.append(q)
        set_robot_configuration(machinery.scene, q[:3], q[3:],
                                machinery.scene.gripper_open_positions)
        ee_x.append(float(machinery.scene.arm_end_effector_frame
                          .CalcPoseInWorld(machinery.scene.plant_context)
                          .translation()[0]))
        if previous is not None and not machinery.checker.CheckEdgeCollisionFree(
            machinery.parameterization(previous), machinery.parameterization(q)
        ):
            collision_bad += 1
        previous = q
    return trajectory, {
        'success': True,
        'edge_count': len(edges),
        'solve_seconds': solve_seconds,
        'duration_s': duration,
        'trajectory_edge_audit_count': len(sample_times) - 1,
        'trajectory_collision_edges': collision_bad,
        'sampled_configurations': len(sample_times),
        'base_always_room1': bool(max(q[0] for q in sampled_q) < 0.),
        'base_center_x_max_m': float(max(q[0] for q in sampled_q)),
        'base_min_door_plane_clearance_m': float(min(
            -q[0] - (0.5 * float(config['robot']['size'][0]) * abs(np.cos(q[2]))
                     + 0.5 * float(config['robot']['size'][1]) * abs(np.sin(q[2])))
            for q in sampled_q
        )),
        'end_effector_max_room2_depth_m': float(max(ee_x)),
        'end_effector_entered_room2': bool(max(ee_x) > 0.),
    }


def run(config_path: str, output_path: str, max_inflations: int = 12,
        solve_local: bool = True):
    config = copy.deepcopy(load_scene_config(config_path))
    config['manipulation_gcs']['pregrasp_region_method'] = 'iris_np2'
    # Keep all arm-reach base poses inside the Room 1 half-plane while retaining
    # the configured 0.10 m footprint clearance for this particular doorway.
    staging = next(r for r in config['gcs_base_planning']['regions']
                   if r['name'] == 'doorway_pickup_staging')
    staging['upper'][0] = -0.45
    began = time.perf_counter()
    scene = build_scene(config=config, visualize=False)
    base = np.asarray(config['pick_task']['candidates'][0]['base_q'], float)
    nav_arm = select_navigation_arm(scene, [np.asarray(config['robot']['start']), base])
    pick = plan_pick(scene, base)
    start = np.r_[base, nav_arm]
    goal = np.r_[base, pick.grasp_q]
    reference = np.vstack([start, goal])
    machinery = prepare_mode_machinery(config=config, mode='pregrasp',
                                       reference_q10=reference, scene_cache={})
    spec = _manipulation_spec(config)
    site = _site_region_spec(config, 'pickup')
    iris_delta = float(spec['total_delta']) / (2 + int(spec['max_bridges']))
    initial_regions = []
    initial_results = []
    initial_seeds = [start, goal]
    init_started = time.perf_counter()
    for index, seed in enumerate(initial_seeds):
        grown = _grow_cover_region(
            scene=machinery.scene, checker=machinery.checker,
            parameterization=machinery.parameterization, seed=seed,
            domain=_phase_domain(machinery.scene, seed, site, spec),
            spec=spec, delta=iris_delta, index=index,
        )
        initial_regions.append(grown.region)
        initial_results.append(grown)
    initial_seconds = time.perf_counter() - init_started
    initial_components = _components(initial_regions)
    regions = list(initial_regions)
    region_seeds = list(initial_seeds)
    added = []
    pair_history = []
    inflation_results = []
    bridge_path = None
    route_method = None
    direct_attempts = 0
    rrt_metadata = None

    if len(initial_components) > 1:
        # The research task is one sparse start/goal component pair. For more
        # components the same loop ranks all cross-component region pairs.
        weights = np.ones(10)
        candidates = []
        for i in sorted(initial_components[0]):
            for j in sorted(initial_components[1]):
                qa, qb, gap = closest_points(regions[i], regions[j], weights)
                candidates.append((gap, i, j, qa, qb))
        for gap, i, j, qa, qb in sorted(candidates, key=lambda x: (x[0], x[1], x[2])):
            qa = _valid_witness(qa, region_seeds[i], machinery)
            qb = _valid_witness(qb, region_seeds[j], machinery)
            if qa is None or qb is None:
                pair_history.append({'pair': [i, j], 'gap': gap,
                                     'status': 'witness_validation_failed'})
                continue
            direct_attempts += 1
            direct_ok = machinery.checker.CheckEdgeCollisionFree(
                machinery.parameterization(qa), machinery.parameterization(qb)
            )
            local_meta = None
            if direct_ok:
                bridge_path = [qa, qb]
                route_method = 'DirectLine'
            else:
                bridge_path, local_meta = _local_rrtc(
                    qa, qb, machinery, config, site,
                    random_seed=260810 + i * 100 + j
                )
                if bridge_path is not None:
                    route_method = 'DirectLine+LocalRRTC_fallback'
            pair_history.append({'pair': [i, j], 'weighted_gap': gap,
                                 'direct_line_collision_free': bool(direct_ok),
                                 'local_rrtc': local_meta,
                                 'status': 'path_found' if bridge_path is not None else 'no_path'})
            if bridge_path is not None:
                break

    inflation_history = []
    if bridge_path is not None:
        parameters = [0., 1.]
        for k in range(max_inflations):
            chain = _shortest_region_chain(regions, start, goal)
            if chain is not None:
                break
            intervals = sorted(zip(parameters[:-1], parameters[1:]),
                               key=lambda pair: (-(pair[1] - pair[0]), pair[0]))
            left_s, right_s = intervals[0]
            s = .5 * (left_s + right_s)
            seed = _path_at_s(bridge_path, s)
            try:
                grown = _grow_cover_region(
                    scene=machinery.scene, checker=machinery.checker,
                    parameterization=machinery.parameterization, seed=seed,
                    domain=_phase_domain(machinery.scene, seed, site, spec),
                    spec=spec, delta=iris_delta, index=len(regions),
                )
                regions.append(grown.region)
                region_seeds.append(seed)
                inflation_results.append(grown)
                parameters.append(s)
                parameters.sort()
                status = 'inflated'
            except RuntimeError as error:
                status = 'inflation_failed'
                inflation_history.append({'iteration': k + 1, 's': s,
                                          'seed': seed.tolist(), 'status': status,
                                          'error': str(error)})
                # Remove this interval from future consideration by stopping;
                # paper-style budgeted failure is reported explicitly.
                break
            chain = _shortest_region_chain(regions, start, goal)
            inflation_history.append({'iteration': k + 1, 's': s,
                                      'seed': seed.tolist(), 'status': status,
                                      'connected': chain is not None,
                                      'region_count': len(regions)})
            if chain is not None:
                break

    chain = _shortest_region_chain(regions, start, goal)
    result = {
        'algorithm': 'GCS-Bridging paper pattern: component-pair DirectLine -> LocalRRTC -> binary IRIS-NP2 inflation -> Drake GCS',
        'paper': 'https://arxiv.org/abs/2608.22326',
        'scene_config': config_path,
        'semantic_mode': 'base_in_room1 + free_cup_arm_reach; grasp transition excluded from this local connectivity test',
        'state': '(x, y, yaw, q_arm[0:7]); base remains in the Room 1 staging region with x <= -0.45 m',
        'semantic_region_override': {'doorway_pickup_staging_x_upper_m': -0.45,
                                     'reason': 'preserve the required minimum base-footprint clearance to the doorway plane'},
        'initial_region_count': len(initial_regions),
        'initial_component_sizes': [len(c) for c in initial_components],
        'initial_component_count': len(initial_components),
        'initial_iris_seconds': initial_seconds,
        'initial_region_audits': [
            {'seed': seed.tolist(),
             'inflation_seconds': float(item.solve_time_seconds),
             'holdout_samples': int(item.holdout_samples),
             'holdout_collisions': int(item.holdout_collisions),
             'faces': int(item.number_of_faces)}
            for seed, item in zip(initial_seeds, initial_results)
        ],
        'closest_region_pair_attempts': pair_history,
        'direct_line_attempts': direct_attempts,
        'bridge_method': route_method,
        'bridge_path_waypoints': len(bridge_path) if bridge_path else 0,
        'binary_inflation': inflation_history,
        'inflated_region_audits': [
            {'inflation_seconds': float(item.solve_time_seconds),
             'holdout_samples': int(item.holdout_samples),
             'holdout_collisions': int(item.holdout_collisions),
             'faces': int(item.number_of_faces)}
            for item in inflation_results
        ],
        'final_region_count': len(regions),
        'final_component_sizes': [len(c) for c in _components(regions)],
        'connected_start_goal': chain is not None,
        'shortest_overlap_chain': chain,
        'added_regions': len(regions) - len(initial_regions),
        'wall_seconds_before_gcs': time.perf_counter() - began,
    }
    if chain is not None:
        selected_regions = [regions[i] for i in chain]
        result['selected_region_halfspaces'] = [
            {'A': region.A().tolist(), 'b': region.b().tolist()}
            for region in selected_regions
        ]
        result['selected_overlap_edges'] = [
            [i, i + 1] for i in range(len(selected_regions) - 1)
        ]
        if solve_local:
            trajectory, gcs = _solve_gcs(selected_regions, start, goal, config, machinery)
            result['gcs'] = gcs
            if trajectory is not None:
                result['gcs']['continuous_path_found'] = True
                result['gcs']['solve_calls'] = 1
                result['gcs']['solver'] = 'Clarabel'
                result['gcs']['convex_relaxation'] = True
                result['gcs']['trajectory_start'] = np.asarray(trajectory.value(trajectory.start_time())).reshape(-1).tolist()
                result['gcs']['trajectory_goal'] = np.asarray(trajectory.value(trajectory.end_time())).reshape(-1).tolist()
        else:
            result['gcs'] = {'success': None, 'reason': 'local solve skipped; regions are intended for a larger unified GCS graph'}
    else:
        result['gcs'] = {'success': False, 'reason': 'region graph remains disconnected; no GCS solve attempted'}
    result['total_wall_seconds'] = time.perf_counter() - began
    result['scope'] = ('A connectivity-repair reproduction for the simple DirectLine case. Collision checking is Drake edge checking at configured resolution; IRIS-NP2 has probabilistic region guarantees. This is not a continuous-time safety certificate or a full task SAP execution.')
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/scene_doorway_reach_semantic_v9.yaml')
    parser.add_argument('--output', default='outputs/results/sparse_gcs_bridging_v1/result.json')
    parser.add_argument('--max-inflations', type=int, default=12)
    args = parser.parse_args()
    run(args.config, args.output, args.max_inflations)


if __name__ == '__main__':
    main()
