"""Official Drake IRIS-NP2 in the fixed-base seven-joint operation space.

The semantic domain is supplied by a conservative whole-robot room bound.
IRIS removes collision portions of that domain; no box fallback is allowed.
Training checks may refine domains; independent held-out samples gate use.
"""
from __future__ import annotations
import time
import numpy as np
from pydrake.common import Parallelism, RandomGenerator
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.planning import IrisNp2, IrisNp2Options, IrisParameterizationFunction
from pydrake.solvers import IpoptSolver
from .build_scene import build_scene, set_robot_configuration
from .iris_region_generator import _make_collision_checker, _make_parameterization, _reduced_position_indices
from .contact_boundary import interior_witness, close_boundary, support_boundary, contact_domain


def make_checker(config, base, arm, gripper, X_GC=None):
    scene = build_scene(config=config, visualize=False, attached_cup_X_GC=X_GC)
    # Reaching is not grasping: enable ALL cup/gripper collision pairs. This
    # also makes the filters uniform per body as required by Drake's checker.
    if X_GC is None:
        gids = [gid for bi in scene.plant.GetBodyIndices(scene.gripper_instance)
                for gid in scene.plant.GetCollisionGeometriesForBody(scene.plant.get_body(bi))]
        scene.scene_graph.collision_filter_manager().Apply(
            CollisionFilterDeclaration().AllowBetween(
                GeometrySet(scene.plant.GetCollisionGeometriesForBody(scene.cup_body)), GeometrySet(gids)))
    # Attached mode keeps payload/table and payload/wall pairs enabled.
    # Only the rigid, intended payload/gripper pair is excluded by build_scene.
    set_robot_configuration(scene, base, arm, gripper)
    mapping = _make_parameterization(scene.plant.GetPositions(scene.plant_context).copy(),
                                     _reduced_position_indices(scene)[3:])
    return scene, _make_collision_checker(scene), mapping


def grow_cover(scene, base, reference, gripper, *, X_GC=None, semantic=True):
    from .green_room_gcs import make_operation_cover
    domains, records, reach = make_operation_cover(scene, base, reference, gripper, semantic=semantic)
    mode_scene, checker, mapping = make_checker(scene.config, base, reference[0], gripper, X_GC)
    spec = scene.config['green_room']['iris']
    regions, output = [], []
    def grow(domain, record, seed=None):
        i = len(regions)
        cap_radius=None
        if X_GC is not None and 'guide_start' in record and support_boundary(mode_scene,mapping(np.asarray(record['guide_start']))):
            # Taper the domain at the support manifold instead of asking IRIS
            # to cut a padded box containing the underside of the table.
            payload_reach=reach+np.linalg.norm(X_GC.translation())+float(scene.config['objects']['cup']['height'])
            domain,seed,cap_radius=contact_domain(mode_scene,checker,mapping,
                record['guide_start'],record['guide_end'],record['lower'],record['upper'],payload_reach)
        if seed is None:
            seed = (np.array(record['lower']) + record['upper']) / 2
        if not checker.CheckConfigCollisionFree(mapping(seed)):
            raise RuntimeError(f'IRIS {i} midpoint collides; requires guide/domain refinement, no box fallback')
        options = IrisNp2Options()
        options.parameterization = IrisParameterizationFunction(mapping, 7)
        options.solver = IpoptSolver()
        options.add_hyperplane_if_solve_fails = bool(spec.get('add_hyperplane_if_solve_fails', False))
        options.ray_sampler_options.num_particles_to_walk_towards = int(spec['ray_particles'])
        sampled = options.sampled_iris_options
        sampled.num_particles = int(spec['particles'])
        sampled.max_iterations = int(spec['max_iterations'])
        sampled.max_iterations_separating_planes = int(spec['separating_iterations'])
        sampled.max_separating_planes_per_iteration = 10
        sampled.parallelism = Parallelism(1)
        sampled.sample_particles_in_parallel = False
        sampled.verbose = False
        sampled.delta = float(spec['total_delta']) / (2*len(domains)+2)
        sampled.epsilon = float(spec['epsilon'])
        sampled.configuration_space_margin = float(spec['configuration_space_margin'])
        sampled.mixing_steps = 20
        sampled.random_seed = int(spec['seed']) + i
        # Preserve real overlaps and endpoints. A supported cup may have a
        # signed distance of -1e-16 from transform roundoff. Do not seed IRIS
        # on that boundary or delete its table collision pair.
        points=[np.asarray(record[k]) for k in ('guide_start','guide_end') if k in record]
        witnesses=[seed]
        boundary=[]
        for point in points:
            witness, on_boundary=interior_witness(mode_scene,checker,mapping,point,seed)
            witnesses.append(witness)
            if on_boundary:
                boundary.append(point)
        sampled.containment_points=np.column_stack(witnesses)
        if boundary:
            sampled.relax_margin=True
        radius = min(float(spec['starting_radius']), float(np.min((np.array(record['upper'])-record['lower']) / 4)))
        if cap_radius is not None:
            radius=min(radius,cap_radius*.1)
        tick = time.perf_counter()
        region = IrisNp2(checker, Hyperellipsoid.MakeHypersphere(radius, seed), domain, options)
        region, closure_shift=close_boundary(region,boundary)
        if not region.PointInSet(seed, 1e-8):
            raise RuntimeError(f'IRIS {i} excluded its seed')
        regions.append(region)
        output.append(dict(record, A=region.A().tolist(), b=region.b().tolist(),
                           method='Drake IrisNp2', seed=seed.tolist(),
                           iris_seconds=time.perf_counter()-tick,
                           attached_payload=X_GC is not None,
                           geometry_filter_scope='All cup/gripper pairs active in reach; rigid cup/gripper excluded in lift; payload/table active',
                           delta=sampled.delta, epsilon=sampled.epsilon))
        output[-1].update(support_boundary_points=[p.tolist() for p in boundary],
                          numerical_closure_shift_rad=closure_shift,
                          contact_domain_cap_radius_rad=cap_radius,
                          containment_points=np.column_stack(witnesses).tolist())
        print(f'IRIS region {i+1}: {region.A().shape[0]} faces', flush=True)

    for domain, record in zip(domains, records):
        grow(domain, record)
    # A real convex-overlap graph is required. Cutting a colliding portion
    # can disconnect two original domains; grow additional official regions
    # at their shared guide state, never add a fictitious graph edge.
    from .green_room_gcs import room_oracle, semantic_box_bound
    oracle = room_oracle(scene)
    limits = scene.config['whole_body_planning']
    def bridge(seed, reason):
        radius = float(spec.get('bridge_domain_radius', .03))
        for _ in range(12):
            lo = np.maximum(seed-radius, limits['joint_lower_limits'])
            hi = np.minimum(seed+radius, limits['joint_upper_limits'])
            bound = semantic_box_bound(oracle, base, lo, hi, gripper, reach)
            if not semantic or bound['whole_box_semantic_lower_bound_m'] >= 1e-6:
                break
            radius /= 2
        else:
            raise RuntimeError('Cannot form a semantic-valid IRIS bridge domain')
        record = dict(lower=lo.tolist(), upper=hi.tolist(), bridge_reason=reason, **bound)
        grow(HPolyhedron.MakeBox(lo, hi), record, seed)
    for i in range(len(domains)-1):
        if not regions[i].IntersectsWith(regions[i+1]):
            bridge(np.asarray(records[i]['guide_end']), f'overlap_gap_{i}_{i+1}')
    for label, seed in [('start', reference[0]), ('goal', reference[-1])]:
        if not any(r.PointInSet(seed, 1e-7) for r in regions):
            bridge(np.asarray(seed), f'missing_{label}')
    return regions, output, reach
