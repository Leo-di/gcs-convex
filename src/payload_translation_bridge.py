"""Official GCS bridge for a rigidly held payload at fixed arm/yaw.

Both the robot and target payload translate; every other object stays fixed.
Clearance disks include payload/environment and room faces. This is a bounded
translation slice, not an arbitrary carrying or contact-transition planner.
"""
import json
from pathlib import Path
import numpy as np
from pydrake.geometry import CollisionFilterDeclaration, GeometrySet
from pydrake.geometry.optimization import GraphOfConvexSetsOptions, Point
from pydrake.planning import GcsTrajectoryOptimization
from pydrake.solvers import ClarabelSolver

from .build_scene import set_robot_configuration
from .green_room_gcs import room_oracle, cup_room_residual
from .geometry_outbound_gcs import make_cover, prune_contained_disks, clipped_disks_intersect, select_region_chain


class PayloadBridgeRejected(RuntimeError):
    """A bounded-cover failure with evidence retained by grounding search."""
    def __init__(self, stage, details):
        super().__init__(stage)
        self.stage, self.details = stage, details


class PayloadTranslationOracle:
    def __init__(self, scene, arm, gripper, yaw, X_GC, *, action_id=None, max_distance=2.):
        self.scene, self.arm, self.gripper, self.yaw = scene, arm, gripper, yaw
        self.X_GC = X_GC
        self.max_distance = float(max_distance)
        if not np.isfinite(self.max_distance) or self.max_distance <= 0:raise ValueError("Positive distance-query cap required")
        self.room = room_oracle(scene)
        from .object_exclusion import ObjectExclusionEvaluator
        self.exclusions = ObjectExclusionEvaluator(scene)
        from .object_corridor import ObjectCorridorEvaluator
        self.corridors = ObjectCorridorEvaluator(scene)
        from .object_relative import ObjectRelativeEvaluator
        self.relative = ObjectRelativeEvaluator(scene)
        from .conditional_permission import ConditionalPermissionEvaluator
        self.permissions = ConditionalPermissionEvaluator(scene)
        from .robot_keep_clear import RobotKeepClearEvaluator
        self.keep_clear = RobotKeepClearEvaluator(scene)
        self.held_object = scene.config.get('active_object_id', 'cup')
        self.grasp_checked = False
        if (self.corridors.specs or self.relative.specs) and action_id is None:
            raise ValueError('Payload corridor planning requires an action ID')
        self.phase = (action_id or 'payload')+'.carry'
        self.calls = 0
        self.robot_instances = {scene.base_instance, scene.arm_instance, scene.gripper_instance}
        # Planning support filters must not hide the moving payload's table.
        from .scene_spec import support_names
        supports = [scene.plant.GetBodyByName(name) for name in support_names(scene.config)]
        scene.scene_graph.collision_filter_manager().Apply(CollisionFilterDeclaration().AllowBetween(
            GeometrySet(scene.plant.GetCollisionGeometriesForBody(scene.cup_body)),
            GeometrySet([gid for body in supports for gid in scene.plant.GetCollisionGeometriesForBody(body)])))

    def query(self, xy):
        self.calls += 1
        s = self.scene
        set_robot_configuration(s, [*xy, self.yaw], self.arm, self.gripper)
        pose = s.plant.EvalBodyPoseInWorld(s.plant_context, s.gripper_body) @ self.X_GC
        s.plant.SetFreeBodyPose(s.plant_context, s.cup_body, pose)
        if self.permissions.specs and not self.grasp_checked:
            from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
            raw = summarize_grasp_audit([{'attached': True, 'cup_gripper_audit': measure_cup_gripper(s)}])
            if not raw['satisfied']:
                raise PayloadBridgeRejected('permission_requires_valid_rigid_grasp', raw)
            self.grasp_checked = True
        query = s.scene_graph.get_query_output_port().Eval(s.scene_graph.GetMyContextFromRoot(s.root_context))
        inspector = query.inspector()
        environment, internal, closest = self.max_distance, float('inf'), None
        for pair in query.ComputeSignedDistancePairwiseClosestPoints(self.max_distance):
            bodies = [s.plant.GetBodyFromFrameId(inspector.GetFrameId(g)) for g in (pair.id_A, pair.id_B)]
            moving = [b.model_instance() in self.robot_instances or b.index() == s.cup_body.index() for b in bodies]
            if sum(moving) == 1 and pair.distance < environment:
                environment, closest = float(pair.distance), [b.name() for b in bodies]
            elif sum(moving) == 2:
                internal = min(internal, float(pair.distance))
        robot_room = self.room.query_current()['residual_m']
        payload_room = cup_room_residual(s, self.room, pose)
        semantic = min(robot_room, payload_room)
        excluded = self.exclusions.current(s)['minimum_residual_m']
        if excluded is not None:
            semantic = min(semantic, excluded)
        contained = self.corridors.current(s, self.phase)['minimum_residual_m']
        if contained is not None:
            semantic = min(semantic, contained)
        relative = self.relative.current(s, self.phase)['minimum_residual_m']
        if relative is not None:
            semantic = min(semantic, relative)
        locked = self.permissions.current([self.held_object])['minimum_locked_residual_m']
        if locked is not None:
            semantic = min(semantic, locked)
        reserved = self.keep_clear.current()['minimum_residual_m']
        if reserved is not None:
            semantic = min(semantic, reserved)
        self.last_residuals = {'environment': environment, 'internal': internal,
            'robot_room': robot_room, 'payload_room': payload_room,
            'object_exclusion': excluded, 'object_corridor': contained, 'conditional_permission': locked,
            'robot_keep_clear': reserved, 'object_relative': relative}
        return min(environment, semantic), internal, closest


def overlap_edges(records):
    """Cheap disk bounds prune candidates before exact clipped intersections."""
    centers = np.array([r['center'] for r in records])
    radii = np.array([r['radius'] for r in records])
    edges = []
    for i, a in enumerate(records):
        # Exact membership allows 1e-10 on each disk. Keep tangent pairs
        # within that tolerance, including floating-point roundoff.
        candidates = np.flatnonzero(np.linalg.norm(centers-centers[i], axis=1) <= radii+radii[i]+2e-10+1e-14)
        for j in candidates[candidates != i]:
            if clipped_disks_intersect(a, records[int(j)]):
                edges.append((i, int(j)))
    return edges


def plan_payload_translation(scene, start, goal, arm, gripper, X_GC, spec, artifact, *, action_id=None):
    if spec.get('graph_mode')=='multibranch':
        from .multibranch_payload import plan
        return plan(scene,start,goal,arm,gripper,X_GC,spec,artifact,action_id=action_id)
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    if (start.shape != (3,) or goal.shape != (3,) or not np.all(np.isfinite([start, goal]))
            or abs(start[2]-goal[2]) > 1e-10):
        raise ValueError('Payload bridge needs finite base triples and equal yaw')
    oracle = PayloadTranslationOracle(scene, arm, gripper, start[2], X_GC, action_id=action_id, max_distance=float(spec['max_radius_m'])+float(spec['margin_m']))
    lower, upper = np.asarray(spec['lower_xy']), np.asarray(spec['upper_xy'])
    margin = float(spec['margin_m'])
    if np.any(start[:2] < lower) or np.any(start[:2] > upper) or np.any(goal[:2] < lower) or np.any(goal[:2] > upper):
        raise RuntimeError('Payload bridge endpoint outside declared translation envelope')
    for label, endpoint in (('start', start), ('goal', goal)):
        clearance, internal, _ = oracle.query(endpoint[:2])
        if clearance < margin-1e-7 or internal < -1e-7:
            raise RuntimeError(f'Payload bridge {label} violates geometry or semantic clearance')
    # Lattice coverage is generated independently of a chosen guide/path.
    boxes = [{'name': 'held_translation_envelope', 'lower': lower, 'upper': upper}]
    rounds = spec.get('connectivity_refinement_rounds', 0)
    if type(rounds) is not int or not 0 <= rounds <= 2:
        raise ValueError('Connectivity refinement rounds must be an integer from 0 to 2')
    fine_spacing = spec.get('refinement_spacing_m')
    cover_attempts = []
    for level in range(rounds+1):
        regions, records = make_cover(oracle.query, boxes, start[:2], goal[:2], lower, upper,
            spacing=spec['spacing_m'], margin=margin, max_radius=spec['max_radius_m'],
            min_radius=spec.get('min_radius_m', .025), refinement_spacing=fine_spacing,
            refinement_radius_threshold=spec.get('refinement_radius_threshold_m'))
        regions, records = prune_contained_disks(regions, records)
        edges = overlap_edges(records)
        sources = [i for i, r in enumerate(regions) if r.PointInSet(start[:2], 1e-9)]
        targets = [i for i, r in enumerate(regions) if r.PointInSet(goal[:2], 1e-9)]
        cover = {'start': start.tolist(), 'goal': goal.tolist(), 'regions': records, 'edges': edges,
            'sources': sources, 'targets': targets, 'spacing_m': spec['spacing_m'],
            'margin_m': margin, 'min_radius_m': spec.get('min_radius_m', .025),
            'refinement_spacing_m': fine_spacing,
            'refinement_radius_threshold_m': spec.get('refinement_radius_threshold_m')}
        text = json.dumps(cover, indent=2)
        Path(artifact).with_name('payload_cover.json').write_text(text)
        Path(artifact).with_name(f'payload_cover_round_{level}.json').write_text(text)
        attempt = dict(level=level, refinement_spacing_m=fine_spacing,
            region_count=len(regions), edge_count=len(edges), connected=False)
        cover_attempts.append(attempt)
        try:
            chain = select_region_chain(records, edges, sources, targets)
            attempt['connected'] = True
            break
        except RuntimeError as exc:
            details = {'reason': str(exc), 'region_count': len(regions), 'edge_count': len(edges),
                'source_region_count': len(sources), 'target_region_count': len(targets),
                'object_exclusion_ids': [s['id'] for s in oracle.exclusions.specs],
                'active_object_corridor_ids': [s['id'] for s in oracle.corridors.specs
                    if s['action_id']+'.'+s['mode'] == oracle.phase], 'cover_attempts': cover_attempts,
                'scope': 'No connection in this finite fixed-arm/yaw cover; not a general task infeasibility proof'}
            Path(artifact).write_text(json.dumps({'satisfied': False, 'stage': 'payload_cover_disconnected', **details}, indent=2))
            if level == rounds:
                raise PayloadBridgeRejected('payload_cover_disconnected', details) from exc
            # Refine the planner's cover, retaining the same safety margin,
            # minimum disk radius, scene, grasp and physical constraints.
            fine_spacing = (fine_spacing or spec['spacing_m'])/2
    count_before = len(regions)
    remap = {old: new for new, old in enumerate(chain)}
    regions = [regions[i] for i in chain]
    edges = [(remap[i], remap[j]) for i, j in edges if i in remap and j in remap]
    planner = GcsTrajectoryOptimization(2)
    free = planner.AddRegions(regions, edges, order=1, h_min=.01, h_max=20, name='held_payload')
    src = planner.AddRegions([Point(start[:2])], order=0, name='lift_end')
    dst = planner.AddRegions([Point(goal[:2])], order=0, name='placement_start')
    planner.AddEdges(src, free, edges_between_regions=[(0, remap[i]) for i in sources if i in remap])
    planner.AddEdges(free, dst, edges_between_regions=[(remap[i], 0) for i in targets if i in remap])
    planner.AddPathLengthCost(1.)
    planner.AddTimeCost(.05)
    planner.AddVelocityBounds(np.full(2, -.15), np.full(2, .15))
    options = GraphOfConvexSetsOptions()
    options.convex_relaxation = True
    options.max_rounded_paths = 10
    options.rounding_seed = 0
    options.solver = options.restriction_solver = ClarabelSolver()
    trajectory, result = planner.SolvePath(src, dst, options)
    if not result.is_success():
        raise RuntimeError('Official payload GCS solve failed in bounded cover')
    duration = float(trajectory.end_time())
    xy = np.array([trajectory.value(t).ravel() for t in np.linspace(0, duration, max(101, int(duration*20)+1))])
    if max(np.max(abs(xy[0]-start[:2])), np.max(abs(xy[-1]-goal[:2]))) > 1e-6:
        raise RuntimeError('Payload GCS endpoint mismatch')
    xy[0], xy[-1] = start[:2], goal[:2]
    # Audit every interpolated final edge; no straight-line fallback.
    samples = []
    for a, b in zip(xy, xy[1:]):
        samples.extend((1-u)*a+u*b for u in np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(b-a)/.0025))+1)))
    values, constraint_samples = [], []
    for p in samples:
        values.append(oracle.query(p))
        constraint_samples.append(dict(oracle.last_residuals))
    minimum = min(v[0] for v in values)
    if minimum < margin-1e-7 or min(v[1] for v in values) < -1e-7:
        raise RuntimeError('Payload GCS independent path audit rejected')
    report = {'satisfied': True, 'method': 'official_Drake_GCS_fixed_arm_yaw_payload_translation',
        'cover_region_count': count_before, 'selected_region_chain': chain,
        'cover_attempts': cover_attempts,
        'relative_reference_scope': 'Reference stationary in this action, current world state; 1-Lipschitz translation residual',
        'regions': [records[i] for i in chain], 'minimum_clearance_m': minimum,
        'margin_m': margin, 'audit_samples': len(samples), 'oracle_calls': oracle.calls,
        'base_path_length_m': float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()),
        'minimum_constraint_residuals_m': {key: min((r[key] for r in constraint_samples if r[key] is not None), default=None)
            for key in constraint_samples[0]},
        'scope': 'Payload and robot translated together; other objects active; bounded real overlap chain restricts coverage/optimality; finite numerical checks'}
    Path(artifact).write_text(json.dumps(report, indent=2))
    return np.c_[xy, np.full(len(xy), start[2])], report
