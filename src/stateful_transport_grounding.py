"""Stateful finite grounding of pick/transport/place sequences.

Search rolls back planning branches before execution. A successful complete
reference then runs once in one continuous free-object contact simulation.
No physical history is undone and no measured object state is overwritten.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from pydrake.math import RigidTransform

from .scene_spec import PROJECT_ROOT, load_scene_config, support_names
from .build_scene import build_scene, set_robot_configuration
from .semantic_plan import configuration, SemanticPlanEvaluator
from .multiobject_grounding import target_config, plan_action
from .green_room_gcs import append_motion, room_oracle
from .whole_robot_room import shape_support
from .meshcat_playback import add_playback_controls


class TransportRejected(RuntimeError):
    def __init__(self, stage, details):
        super().__init__(stage)
        self.stage, self.details = stage, details


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def operation_refinement_policy(spec):
    """Expose the bounded manipulation audit budget without skipping checks."""
    defaults = {'max_rounds': 2, 'max_regions': 512, 'time_budget_s': 60.}
    supplied = spec.get('operation_refinement', {})
    if not isinstance(supplied, dict) or set(supplied)-set(defaults):
        raise ValueError('Unknown operation refinement budget fields')
    policy = dict(defaults, **supplied)
    if (type(policy['max_rounds']) is not int or policy['max_rounds'] < 0
            or type(policy['max_regions']) is not int or policy['max_regions'] < 1
            or isinstance(policy['time_budget_s'], bool)
            or not isinstance(policy['time_budget_s'], (int, float))
            or not np.isfinite(policy['time_budget_s']) or policy['time_budget_s'] <= 0):
        raise ValueError('Finite positive operation budgets and nonnegative integer rounds required')
    return policy


def pickup_candidates(action, spec):
    candidates = action.get('pickup_candidates', [
        {'base_q': action['pickup_base'], 'grasp_rpy': spec['grasp_rpy']}])
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('Nonempty pickup candidate list required')
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != {'base_q', 'grasp_rpy'}:
            raise ValueError('Pickup candidates require explicit base_q and grasp_rpy')
        for key in ('base_q', 'grasp_rpy'):
            value = np.asarray(candidate[key], float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError('Finite pickup candidate triples required')
    return candidates


def validate_task(spec):
    from .task_order_search import validate_order
    validate_order(spec)
    if type(spec.get('placement_lookahead', False)) is not bool:
        raise ValueError('placement_lookahead must be a Boolean')
    operation_refinement_policy(spec)
    from .transport_posture_search import validate_search
    validate_search(spec.get("transport_posture_search"), spec.get("actions", []))
    from .human_passage import validate_passage
    validate_passage(spec.get("human_passage"))
    from .room_entry import validate_entry
    validate_entry(spec)
    from .robot_keep_clear import validate_keep_clear
    validate_keep_clear(spec.get('robot_keep_clear', []))
    names, ids = set(spec['objects']), set()
    from .conditional_permission import validate_permissions
    validate_permissions(spec.get('conditional_permissions', []), names)
    from .object_exclusion import validate_exclusions
    validate_exclusions(spec.get('object_exclusions', []), names)
    from .object_corridor import validate_corridors
    validate_corridors(spec.get('object_corridors', []), names,
                      {a['id']: a['object_id'] for a in spec['actions']})
    from .object_relative import validate_relative
    validate_relative(spec.get('object_relative', []), names,
                      {a['id']: a['object_id'] for a in spec['actions']})
    from .ordered_visitation import validate_visits
    validate_visits(spec.get('ordered_visits', []), names,
                    {a['id']: a['object_id'] for a in spec['actions']})
    if 'cup' not in names or not spec['actions']:
        raise ValueError('Named cups and ordered actions required')
    for center in spec['objects'].values():
        if np.asarray(center).shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError('Finite object center triples required')
    acted = set()
    for a in spec['actions']:
        if a['primitive'] != 'pick_transport_place':
            raise ValueError('Unsupported transport primitive')
        if a['id'] in ids or not a['id'] or any(c in a['id'] for c in './\\') or a['object_id'] not in names:
            raise ValueError('Unique safe action IDs and known object IDs required')
        if a['object_id'] in acted:
            raise ValueError('Current adapter requires one transport per object')
        acted.add(a['object_id'])
        known_supports = {'table', 'dropoff_table', *spec.get('support_surfaces', {})}
        if a['source_support'] not in known_supports or a['destination_support'] not in known_supports:
            raise ValueError('Known named table support required')
        bases, placements = np.asarray(a['pickup_base']), np.asarray(a['placements'])
        if (bases.shape != (3,) or placements.ndim != 2 or placements.shape[1] != 3 or not len(placements)
                or not np.all(np.isfinite(bases)) or not np.all(np.isfinite(placements))):
            raise ValueError('Finite pickup base and nonempty placement triples required')
        if 'transport_offset_m' in a:
            value=np.asarray(a['transport_offset_m'],float)
            if value.shape!=(3,) or not np.all(np.isfinite(value)):raise ValueError('Finite transport posture offset required')
        ids.add(a['id'])
        pickup_candidates(a, spec)
    if (type(spec['max_expansions']) is not int or spec['max_expansions'] < 1
            or not np.isfinite(spec['time_budget_s']) or spec['time_budget_s'] <= 0
            or not np.isfinite(spec['time_scale']) or spec['time_scale'] <= 0):
        raise ValueError('Positive finite search/timing budgets required')
    carry = spec['carry']
    from .ordered_visitation import checkpoint_policy
    checkpoint_policy(carry)
    refinement_rounds = carry.get('connectivity_refinement_rounds', 0)
    if type(refinement_rounds) is not int or not 0 <= refinement_rounds <= 2:
        raise ValueError('Connectivity refinement rounds must be an integer from 0 to 2')
    lower, upper = np.asarray(carry['lower_xy']), np.asarray(carry['upper_xy'])
    if (lower.shape != (2,) or upper.shape != (2,) or not np.all(np.isfinite([lower, upper]))
            or np.any(lower >= upper) or any(not np.isfinite(carry[k]) or carry[k] <= 0
                for k in ('spacing_m', 'margin_m', 'max_radius_m'))):
        raise ValueError('Finite positive carrying envelope and spacings required')
    for key in ('min_radius_m', 'refinement_spacing_m', 'refinement_radius_threshold_m'):
        if key in carry and (not np.isfinite(carry[key]) or carry[key] <= 0):
            raise ValueError('Optional cover radii and spacings must be finite and positive')
    if (carry.get('min_radius_m', .025) > carry['max_radius_m']
            or ('refinement_spacing_m' in carry and carry['refinement_spacing_m'] >= carry['spacing_m'])):
        raise ValueError('Minimum radius exceeds maximum or refinement is not finer than coarse spacing')


def make_config(spec):
    validate_task(spec)
    cfg = load_scene_config(PROJECT_ROOT/spec['scene'])
    primary = copy.deepcopy(cfg['objects']['cup'])
    cfg['objects']['cup']['center'] = spec['objects']['cup']
    cfg['objects']['additional_cups'] = {k: dict(copy.deepcopy(primary), center=v)
                                        for k, v in spec['objects'].items() if k != 'cup'}
    cfg['objects']['dropoff_table'].update(spec['destination_table'])
    if spec.get('support_surfaces'):
        cfg['support_surfaces'] = copy.deepcopy(spec['support_surfaces'])
    cfg['robot']['start'] = spec['start_base']
    if 'entry_navigation' in spec:
        cfg['entry_navigation'] = copy.deepcopy(spec['entry_navigation'])
    cfg['pick_task'].update(gripper_rpy=spec['grasp_rpy'], lift_offset=spec['lift_offset'])
    cfg['green_room']['operation_refinement'] = operation_refinement_policy(spec)
    cfg['execution_dynamics'] = {'hydroelastic_support': True, 'reference_time_scale': spec['time_scale']}
    cfg['initial_supports'] = {a['object_id']: a['source_support'] for a in spec['actions']}
    cfg['execution_object_supports'] = {k: support_names(cfg) for k in spec['objects']}
    cfg['object_exclusions'] = copy.deepcopy(spec.get('object_exclusions', []))
    cfg['object_exclusion_display'] = copy.deepcopy(spec.get('object_exclusion_display', cfg['object_exclusions']))
    cfg['object_relative'] = copy.deepcopy(spec.get('object_relative', []))
    cfg['object_relative_display'] = copy.deepcopy(spec.get('object_relative_display', cfg['object_relative']))
    cfg['object_corridors'] = copy.deepcopy(spec.get('object_corridors', []))
    cfg['object_corridor_display'] = copy.deepcopy(spec.get('object_corridor_display', cfg['object_corridors']))
    cfg['ordered_visits'] = copy.deepcopy(spec.get('ordered_visits', []))
    cfg['conditional_permissions'] = copy.deepcopy(spec.get('conditional_permissions', []))
    if 'human_passage' in spec:cfg['human_passage'] = copy.deepcopy(spec['human_passage'])
    cfg['robot_keep_clear'] = copy.deepcopy(spec.get('robot_keep_clear', []))
    cfg['robot_keep_clear_display'] = copy.deepcopy(spec.get('robot_keep_clear_display', cfg['robot_keep_clear']))
    for region in cfg['gcs_base_planning']['regions']:
        region['yaw_bounds'] = [-.4, .4]
    return cfg


def initial_world(config):
    scene = build_scene(config=config, visualize=False)
    return {'poses': {k: X.GetAsMatrix4().tolist() for k, X in scene.object_initial_poses.items()},
        'supports': {k: config.get('initial_supports',{}).get(k,'table') for k in scene.object_bodies},
        'robot': {'base_q': config['robot']['start'], 'arm_q': config['pick_task']['navigation_arm_q'],
                  'gripper_q': scene.gripper_open_positions.tolist()}}


def world_config(config, world, active):
    cfg = target_config(config, active)
    for key, spec in {active: cfg['objects']['cup'], **cfg['objects']['additional_cups']}.items():
        spec['center'] = np.array(world['poses'][key])[:3, 3].tolist()
    cfg['object_initial_poses'] = copy.deepcopy(world['poses'])
    cfg['planning_object_supports'] = {k: [v] for k, v in world['supports'].items()}
    cfg['pickup_support_body'] = world['supports'][active]
    return cfg


def successor(world, object_id, pose, support, robot):
    result = copy.deepcopy(world)
    result['poses'][object_id] = np.asarray(pose).tolist()
    result['supports'][object_id] = support
    result['robot'] = {k: copy.deepcopy(robot[k]) for k in ('base_q', 'arm_q', 'gripper_q')}
    return result


def destination_occupancy(scene, pose):
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, pose)
    query = scene.scene_graph.get_query_output_port().Eval(scene.scene_graph.GetMyContextFromRoot(scene.root_context))
    pairs = []
    for name, body in scene.object_bodies.items():
        if body.index() == scene.cup_body.index():
            continue
        for gid in scene.plant.GetCollisionGeometriesForBody(scene.cup_body):
            for other in scene.plant.GetCollisionGeometriesForBody(body):
                distance = float(query.ComputeSignedDistancePairClosestPoints(gid, other).distance)
                pairs.append({'other_object': name, 'distance_m': distance})
    return {'satisfied': all(p['distance_m'] >= .005 for p in pairs), 'pairs': pairs,
            'required_gap_m': .005, 'method': 'unfiltered_Drake_object_pair_distance'}


def body_room_residual(scene, body, pose, oracle):
    inspector = scene.scene_graph.model_inspector()
    values = []
    for gid in scene.plant.GetCollisionGeometriesForBody(body):
        X = pose @ inspector.GetPoseInFrame(gid)
        for n, b in zip(oracle.normals, oracle.offsets):
            values.append(float(b-n@X.translation()-shape_support(inspector.GetShape(gid), X.rotation().matrix().T@n)-oracle.margin))
    if not values:
        raise ValueError('Object without collision geometry')
    return min(values)


def audit_action(scene, frames, X_GC, *, max_samples=150000):
    from .collision_checker import evaluate_collision_status
    from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
    room = room_oracle(scene)
    from .robot_keep_clear import RobotKeepClearEvaluator
    keep_clear = RobotKeepClearEvaluator(scene)
    reserved_min = None
    from .object_exclusion import ObjectExclusionEvaluator
    exclusions = ObjectExclusionEvaluator(scene)
    exclusion_min = None
    from .object_corridor import ObjectCorridorEvaluator
    corridors = ObjectCorridorEvaluator(scene)
    from .object_relative import ObjectRelativeEvaluator
    relatives = ObjectRelativeEvaluator(scene)
    relative_min = None
    from .conditional_permission import ConditionalPermissionEvaluator
    permissions = ConditionalPermissionEvaluator(scene)
    corridor_min = None
    minimum, bad, count, examples = None, 0, 0, []
    for i, frame in enumerate(frames):
        a, b = configuration(frames[max(0, i-1)]), configuration(frame)
        n = max(1, int(np.ceil(np.max(abs(b-a)/np.r_[[.01]*10, [.001]*2]))))
        if count+n+1 > max_samples:
            raise RuntimeError('Stateful reference audit budget exhausted')
        for alpha in np.linspace(0, 1, n+1):
            q = (1-alpha)*a+alpha*b
            set_robot_configuration(scene, q[:3], q[3:10], q[10:])
            held = frame['mode'] in ('lift', 'hold', 'carry', 'lower')
            for object_id, matrix in frame['world_objects'].items():
                pose = (scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body) @ X_GC
                        if held and object_id == frame['object_id'] else RigidTransform(np.array(matrix)))
                scene.plant.SetFreeBodyPose(scene.plant_context, scene.object_bodies[object_id], pose)
            status = evaluate_collision_status(scene, q[:3], arm_q=q[3:10], gripper_q=q[10:])
            raw = summarize_grasp_audit([{'attached': held, 'cup_gripper_audit': measure_cup_gripper(scene)}])
            permission = permissions.current([frame['object_id']] if held and raw['satisfied'] else [])
            reserved = keep_clear.current()
            if reserved['minimum_residual_m'] is not None:
                v = reserved['minimum_residual_m']
                reserved_min = v if reserved_min is None else min(reserved_min, v)
            values = [status.minimum_signed_distance, room.query_current()['residual_m']]
            # Raw support queries prevent support filters hiding penetration.
            query = scene.scene_graph.get_query_output_port().Eval(scene.scene_graph.GetMyContextFromRoot(scene.root_context))
            for object_id, body in scene.object_bodies.items():
                pose = scene.plant.EvalBodyPoseInWorld(scene.plant_context, body)
                values.append(body_room_residual(scene, body, pose, room))
                for support in support_names(scene.config):
                    for cg in scene.plant.GetCollisionGeometriesForBody(body):
                        for tg in scene.plant.GetCollisionGeometriesForBody(scene.plant.GetBodyByName(support)):
                            values.append(float(query.ComputeSignedDistancePairClosestPoints(cg, tg).distance))
            value = min(values)
            exclusion = exclusions.current(scene)
            corridor = corridors.current(scene, frame['phase'])
            relative = relatives.current(scene, frame['phase'])
            if relative['minimum_residual_m'] is not None:
                v = relative['minimum_residual_m']
                relative_min = v if relative_min is None else min(relative_min, v)
            if corridor['minimum_residual_m'] is not None:
                v = corridor['minimum_residual_m']
                corridor_min = v if corridor_min is None else min(corridor_min, v)
            if exclusion['minimum_residual_m'] is not None:
                v = exclusion['minimum_residual_m']
                exclusion_min = v if exclusion_min is None else min(exclusion_min, v)
            minimum = value if minimum is None else min(minimum, value)
            count += 1
            if (value < -1e-6 or not raw['satisfied'] or not exclusion['satisfied']
                    or not corridor['satisfied'] or not permission['satisfied'] or not reserved['satisfied']
                    or not relative['satisfied']):
                bad += 1
                if len(examples) < 5:
                    examples.append({'phase': frame['phase'], 'edge': i, 'alpha': float(alpha),
                        'minimum_residual_m': value, 'exclusion':exclusion, 'corridor':corridor, 'permission':permission,
                        'closest': [status.closest_geometry_a, status.closest_geometry_b], 'raw': raw,
                        'robot_keep_clear':reserved, 'object_relative':relative})
    return {'satisfied': bad == 0, 'samples': count, 'violations': bad,
        'minimum_residual_m': minimum, 'counterexamples': examples,
        'minimum_object_exclusion_residual_m': exclusion_min,
        'minimum_object_corridor_residual_m': corridor_min,
        'minimum_object_relative_residual_m': relative_min,
        'minimum_robot_keep_clear_residual_m': reserved_min,
        'scope': 'Interpolated numerical collision/raw-grasp/support/room audit of robot and every world object'}


def translated_destination_suffix(source_frames, delta):
    """Shared fixed suffix: transport postures restore before these frames."""
    result=[]
    for f in source_frames:
        if f['mode'] in ('lower','release','retreat'):
            shifted=copy.deepcopy(f)
            shifted['base_q']=(np.asarray(f['base_q'])+np.r_[np.asarray(delta)[:2],0.]).tolist()
            result.append(shifted)
    return result


def ground_action(config, world, action, placement, spec, folder, *, pickup_cache=None, posthoc_accept=None, posture_selector=None, destination_precheck=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    if world['supports'][action['object_id']] != action['source_support']:
        raise TransportRejected('symbolic_source_support_mismatch', {'world_supports': world['supports']})
    cfg = world_config(config, world, action['object_id'])
    if 'grasp_rpy' in action:
        cfg['pick_task']['gripper_rpy'] = copy.deepcopy(action['grasp_rpy'])
    scene = build_scene(config=cfg, visualize=False)
    source_pose = RigidTransform(np.array(world['poses'][action['object_id']]))
    target = np.asarray(placement)
    if abs(target[2]-source_pose.translation()[2]) > 1e-6:
        raise ValueError('Current adapter requires equal-height tables')
    target_pose = RigidTransform(source_pose.rotation(), target)
    occupancy = destination_occupancy(scene, target_pose)
    (folder/'destination_occupancy.json').write_text(json.dumps(occupancy, indent=2))
    if not occupancy['satisfied']:
        raise TransportRejected('destination_occupied_by_previous_action', occupancy)
    scene.plant.SetFreeBodyPose(scene.plant_context, scene.cup_body, source_pose)
    pickup = folder/'pickup'
    pickup.mkdir()
    cache_key = digest({'config': cfg, 'world': world, 'action': action['id'], 'pickup_base': action['pickup_base']})
    if pickup_cache is not None and cache_key in pickup_cache:
        source_frames, pickup_report, cached_folder = copy.deepcopy(pickup_cache[cache_key])
        (pickup/'cache_source.json').write_text(json.dumps({'key': cache_key, 'source': cached_folder}, indent=2))
    else:
        source_frames, pickup_report = plan_action(cfg, action, action['pickup_base'], world['robot'], pickup)
        (pickup/'reference.json').write_text(json.dumps(source_frames))
        (pickup/'result.json').write_text(json.dumps(pickup_report, indent=2))
        if pickup_cache is not None:
            pickup_cache[cache_key] = copy.deepcopy((source_frames, pickup_report, str(pickup)))
    if destination_precheck is not None:
        destination_precheck(build_scene(config=copy.deepcopy(cfg),visualize=False),world,action,placement,spec,source_frames,folder)
    policy = spec.get('transport_posture_search')
    if policy is None:
        answer=complete_transport_candidate(scene, world, action, placement, spec, folder, source_frames, pickup_report)
        if posthoc_accept is not None:posthoc_accept(answer,world,action,folder)
        return answer
    from .transport_posture_search import generate_candidates, screen_candidate, select_posture
    grasp = next(f for f in reversed(source_frames) if f['mode']=='close')
    held = next(f for f in reversed(source_frames) if f['mode']=='hold')
    base = np.asarray(action['pickup_base']);destination=base+np.r_[target[:2]-source_pose.translation()[:2],0.]
    set_robot_configuration(scene,base,grasp['arm_q'],grasp['gripper_q'])
    X_GC=scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.gripper_body).inverse()@source_pose
    candidates=generate_candidates(scene,base,held['arm_q'],policy)
    trial_root=folder/'transport_candidates';trial_root.mkdir()
    def screen(candidate):
        # A fresh model prevents a prior payload query's support-filter changes
        # and a rejected candidate's poses from contaminating IK or another trial.
        fresh=build_scene(config=copy.deepcopy(cfg),visualize=False)
        return screen_candidate(fresh,base,destination,held['arm_q'],X_GC,candidate['offset_world_m'],
            spec['carry'],action['id'],policy['screen_spacing_m'])
    def evaluate(candidate):
        trial=trial_root/candidate['id'];trial.mkdir(exist_ok=False)
        fresh=build_scene(config=copy.deepcopy(cfg),visualize=False)
        print(f"Transport trial {action['id']} {candidate['id']} offset={candidate['offset_world_m']}",flush=True)
        answer=complete_transport_candidate(fresh,world,action,placement,spec,trial,source_frames,pickup_report,candidate['offset_world_m'])
        if posthoc_accept is not None:posthoc_accept(answer,world,action,trial)
        return answer
    selector=select_posture if posture_selector is None else posture_selector
    answer,posture_search=selector(candidates,screen,evaluate,policy,folder/'transport_posture_search.json')
    import shutil
    chosen=trial_root/posture_search['selected']['id']
    for artifact in chosen.iterdir():
        if artifact.is_file():shutil.copy2(artifact,folder/artifact.name)
    frames,expected,report=answer
    report['transport_posture_search']=posture_search
    (folder/'result.json').write_text(json.dumps(report,indent=2))
    return frames,expected,report


def complete_transport_candidate(scene, world, action, placement, spec, folder, source_frames, pickup_report, transport_offset=None):
    from .ordered_visitation import plan_ordered_translation
    source_pose = RigidTransform(np.array(world['poses'][action['object_id']]))
    target = np.asarray(placement)
    cfg = scene.config
    frames = [copy.deepcopy(f) for f in source_frames if f['mode'] in ('arrive', 'reach', 'close', 'lift', 'hold')]
    grasp = next(f for f in reversed(frames) if f['mode'] == 'close')
    set_robot_configuration(scene, grasp['base_q'], grasp['arm_q'], grasp['gripper_q'])
    X_GC = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body).inverse() @ source_pose
    base = np.asarray(action['pickup_base'])
    delta = target-source_pose.translation()
    destination_base = base+np.r_[delta[:2], 0.]
    held_arm, closed = np.asarray(frames[-1]['arm_q']), scene.gripper_closed_positions
    transport_path, transport_report = None, None
    offset = action.get('transport_offset_m',[0.,0.,0.]) if transport_offset is None else transport_offset
    if np.linalg.norm(offset)>1e-10:
        from .transport_posture import plan_transport_posture
        transport_path,transport_report=plan_transport_posture(scene,base,held_arm,X_GC,offset,folder/'transport_posture.json')
        first=len(frames)
        append_motion(frames,np.repeat(base[None,:],len(transport_path),axis=0),transport_path,
            np.repeat(closed[None,:],len(transport_path),axis=0),action['id']+'.transport_retract')
        for f in frames[first:]:f.update(object_id=action['object_id'],mode='hold')
        held_arm=np.asarray(transport_path[-1])
    carry, bridge_report = plan_ordered_translation(scene, base, destination_base, held_arm, closed,
        X_GC, spec['carry'], folder/'payload_bridge.json', action_id=action['id'])
    before = len(frames)
    append_motion(frames, carry, np.repeat(held_arm[None, :], len(carry), axis=0),
                  np.repeat(closed[None, :], len(carry), axis=0), action['id']+'.carry')
    for f in frames[before:]:
        f.update(object_id=action['object_id'], mode='carry')
    if transport_path is not None:
        first=len(frames);restore=transport_path[::-1]
        append_motion(frames,np.repeat(destination_base[None,:],len(restore),axis=0),restore,
            np.repeat(closed[None,:],len(restore),axis=0),action['id']+'.transport_extend')
        for f in frames[first:]:f.update(object_id=action['object_id'],mode='hold')
    frames.extend(translated_destination_suffix(source_frames,delta))
    # Translation changes collision geometry. Accept each translated phase
    # with the current world before it is admitted to the action result.
    from itertools import groupby
    from .accepted_motion import require_endpoint
    released_pose = None
    accepted_frames, phase_audits = [], {}
    for phase, block in groupby(frames, key=lambda f: f['phase']):
        block = list(block)
        for f in block:
            set_robot_configuration(scene, f['base_q'], f['arm_q'], f['gripper_q'])
            poses = copy.deepcopy(world['poses'])
            if f['mode'] in ('lift', 'hold', 'carry', 'lower'):
                released_pose = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body) @ X_GC
                poses[action['object_id']] = released_pose.GetAsMatrix4().tolist()
            elif f['mode'] in ('release', 'retreat'):
                poses[action['object_id']] = released_pose.GetAsMatrix4().tolist()
            f['world_objects'] = poses
            f['world_supports'] = dict(world['supports'])
            if f['mode'] in ('release', 'retreat'):
                f['world_supports'][action['object_id']] = action['destination_support']
            f['payload_mode'] = 'held' if f['mode'] in ('lift', 'hold', 'carry', 'lower') else 'free'
        previous = accepted_frames[-1] if accepted_frames else world['robot']
        require_endpoint(block[0], configuration(previous), phase)
        previous_poses = previous.get('world_objects', world['poses'])
        if any(np.max(np.abs(np.asarray(block[0]['world_objects'][key])-matrix)) > 1e-6
               for key, matrix in previous_poses.items()):
            raise TransportRejected('object_pose_discontinuity_at_phase_boundary', {'phase': phase})
        checked = audit_action(scene, block, X_GC)
        semantic = SemanticPlanEvaluator(scene, semantic_plan(cfg, block)).audit(block)
        phase_audits[phase] = {'geometry': checked, 'semantic': semantic}
        (folder/'phase_acceptance.json').write_text(json.dumps(phase_audits, indent=2))
        if not checked['satisfied'] or not semantic['satisfied']:
            raise TransportRejected('transport_phase_rejected', {'phase': phase, **phase_audits[phase]})
        accepted_frames.extend(block)
    frames = accepted_frames
    # Check the assembled action after the phase-level checks.
    audit = audit_action(scene, frames, X_GC)
    (folder/'reference_audit.json').write_text(json.dumps(audit, indent=2))
    if not audit['satisfied']:
        raise TransportRejected('phase_composition_regression', audit)
    expected = successor(world, action['object_id'], released_pose.GetAsMatrix4(),
                         action['destination_support'], frames[-1])
    report = {'satisfied': True, 'input_world_sha256': digest(world), 'output_world_sha256': digest(expected),
        'source_pose': source_pose.GetAsMatrix4().tolist(), 'placement_target': list(placement),
        'release_pose': released_pose.GetAsMatrix4().tolist(), 'pickup': pickup_report,
        'payload_bridge': bridge_report, 'transport_posture':transport_report, 'phase_acceptance': phase_audits, 'reference_audit': audit}
    (folder/'result.json').write_text(json.dumps(report, indent=2))
    return frames, expected, report


def search(config, spec, output, *, strategy='backtrack', ground=ground_action, entry_planner=None, complete_accept=None):
    if strategy not in ('greedy', 'backtrack'):
        raise ValueError('Unknown grounding search strategy')
    root = Path(output)
    initial = initial_world(config)
    events, branches = [], []
    began, expansions = time.monotonic(), 0
    def visit(depth, world, prefix, selected):
        nonlocal expansions
        if depth == len(spec['actions']):
            if complete_accept is not None:
                try:
                    admitted = complete_accept(prefix, copy.deepcopy(world), copy.deepcopy(selected))
                    if not admitted.get('satisfied', False):
                        raise TransportRejected('complete_task_admission_rejected', admitted)
                except Exception as exc:
                    events.append(dict(event='complete_task_candidate_rejected',
                        order=[r['action'] for r in selected],world_sha256=digest(world),
                        failure_stage=getattr(exc,'stage','whole_task_planner_or_audit_failure'),
                        reason=str(exc),details=getattr(exc,'details',{})))
                    return None
            return prefix, world, selected
        from .task_order_search import available_actions
        available = available_actions(spec, selected, world)
        if spec.get('task_order',{}).get('mode')=='search':
            events.append({'event':'available_task_actions','depth':depth,
                           'completed':[r['action'] for r in selected],
                           'available':[a['id'] for a in available]})
        options = []
        for template in available:
            placement_order = list(range(len(template['placements'])))
            if spec.get('placement_lookahead', False):
                from .placement_lookahead import rank_placements
                # The lookahead sees this prospective order and every remaining
                # task, rather than the old fixed-depth action.
                by_id = {a['id']:a for a in spec['actions']}
                prefix_actions = [by_id[r['action']] for r in selected]
                local_spec = dict(spec, actions=prefix_actions+[template]+[a for a in spec['actions']
                    if a['id'] != template['id'] and a['id'] not in {r['action'] for r in selected}])
                placement_order, predictions = rank_placements(config, world, local_spec, depth)
                events.append({'event':'placement_lookahead_order','depth':depth,
                               'action':template['id'],'order':placement_order,
                               'predictions':predictions,
                               'scope':'Ranking only; no candidates removed'})
            options.extend((template, i, template['placements'][i], j, c) for i in placement_order
                           for j, c in enumerate(pickup_candidates(template, spec)))
        for template, option, placement, candidate_index, candidate in options:
            action = dict(template, pickup_base=candidate['base_q'], grasp_rpy=candidate['grasp_rpy'])
            candidate_world = copy.deepcopy(world)
            if depth == 0 and config.get('entry_navigation'):
                # The manipulation is planned first. Its selected docking pose
                # is the later entry planner's goal, not a physical state reset.
                candidate_world['robot']['base_q'] = copy.deepcopy(candidate['base_q'])
            if expansions >= spec['max_expansions'] or time.monotonic()-began >= spec['time_budget_s']:
                events.append({'event': 'budget_exhausted', 'depth': depth})
                return None
            index, expansions = expansions, expansions+1
            row = {'index': index, 'depth': depth, 'action': action['id'], 'option': option,
                'pickup_candidate_index': candidate_index, 'pickup_candidate': copy.deepcopy(candidate),
                'placement': placement, 'input_world_sha256': digest(candidate_world), 'local_satisfied': False}
            branches.append(row)
            print(f"Ground action={action['id']} option={option} world={digest(world)[:12]}", flush=True)
            try:
                frames, next_world, report = ground(config, candidate_world, action, placement,
                    spec, root/f'branch_{index:03d}')
                if time.monotonic()-began >= spec['time_budget_s']:
                    raise TransportRejected('search_time_budget_exhausted_after_planner', {})
                entry_report = None
                row['manipulation_satisfied'] = True
                if depth == 0 and entry_planner is not None:
                    try:
                        frames, entry_report = entry_planner(frames, root/f'branch_{index:03d}'/'room_entry')
                        if not entry_report['satisfied']:
                            raise RuntimeError('Entry planner returned a rejected report')
                    except Exception as exc:
                        row['entry_satisfied'] = False
                        raise TransportRejected('entry_bridge_rejected', dict(
                            cause_stage=getattr(exc, 'stage', 'entry_planner_or_audit_failure'),
                            cause=str(exc), evidence=getattr(exc, 'details', {}))) from exc
                    row['entry_satisfied'] = True
                    if time.monotonic()-began >= spec['time_budget_s']:
                        raise TransportRejected('search_time_budget_exhausted_after_entry', {})
                if prefix and np.max(abs(configuration(prefix[-1])-configuration(frames[0]))) > 1e-6:
                    raise TransportRejected('complete_configuration_connection_mismatch', {})
                row.update(local_satisfied=True, output_world_sha256=digest(next_world))
                answer = visit(depth+1, next_world, prefix+frames, selected+[
                    {'branch': index, 'action': action['id'], 'object_id': action['object_id'],
                     'pickup_candidate_index': candidate_index, 'pickup_candidate': copy.deepcopy(candidate),
                     'placement': placement, 'report': report, 'entry_report': entry_report}])
                if answer is not None:
                    return answer
                events.append({'event': 'rollback_after_downstream_failure', 'depth': depth,
                    'discard_world_sha256': digest(next_world), 'restore_world_sha256': digest(world)})
                print(f"Rollback action={action['id']} after downstream failure", flush=True)
            except Exception as exc:
                row.update(failure_stage=getattr(exc, 'stage', 'planner_or_audit_failure'),
                           reason=str(exc), details=getattr(exc, 'details', {}))
                print(f"Reject action={action['id']}: {exc}", flush=True)
            if strategy == 'greedy':
                return None
        return None
    answer = visit(0, initial, [], [])
    result = {'satisfied': answer is not None, 'strategy': strategy, 'expansions': expansions,
        'placement_lookahead': spec.get('placement_lookahead', False),
        'task_order_policy': spec.get('task_order', {'mode':'fixed'}),
        'selected_task_order': [r['action'] for r in answer[2]] if answer else None,
        'branches': branches, 'events': events, 'wall_seconds': time.monotonic()-began,
        'initial_world': initial, 'termination': 'complete_reference' if answer else 'bounded_search_exhausted',
        'scope': 'Planning backtracking before execution; bounded candidates; no task infeasibility claim'}
    if answer:
        result.update(frames=answer[0], final_world=answer[1], selected=answer[2])
        if config.get('entry_navigation'):
            result['initial_world']['robot']['base_q'] = copy.deepcopy(answer[2][0]['pickup_candidate']['base_q'])
    return result


def semantic_plan(config, frames):
    plan = {'version': 1, 'predicates': {'inside': {'kind': 'whole_robot_in_room',
        'region': 'room2', 'margin_m': .02}}, 'phases': []}
    reserved_refs = []
    if config.get('robot_keep_clear'):
        plan['predicates']['customer_space'] = {'kind':'robot_keep_clear', 'zones':config['robot_keep_clear']}
        reserved_refs = ['customer_space']
    for phase in dict.fromkeys(f['phase'] for f in frames):
        f = next(f for f in frames if f['phase'] == phase)
        from .room_entry import is_entry
        if is_entry(config,f):
            name = 'entry_tucked_open'
            plan['predicates'][name] = {'kind':'configuration_equals','indices':list(range(3,12)),
                'target':config['pick_task']['navigation_arm_q']+f['gripper_q'],
                'tolerance':1e-6,'execution_tolerance':[.12]*7+[.004]*2}
            plan['phases'].append({'id':phase,'invariants':[name]+reserved_refs,'exit':['inside']})
            continue
        refs = ['inside']+reserved_refs
        if f.get('joint_whole_task'):
            # Every robot configuration was selected by the shared task solve.
            # Contact opening/closure varies fingers, while spatial predicates
            # remain active. Motion modes retain the prescribed grasp state.
            if f['mode'] in ('close','release'):
                plan['phases'].append({'id':phase,'invariants':refs})
                continue
            indices,target,tolerance,execution=[10,11],f['gripper_q'],1e-6,[.004]*2
        elif f['mode'] == 'joint_approach':
            # The joint planner controls the base and all seven arm joints.
            # Only the empty-hand gripper configuration remains invariant.
            indices, target, tolerance, execution = [10, 11], f['gripper_q'], 1e-6, [.004]*2
        elif f['mode'] == 'arrive':
            indices, target, tolerance, execution = list(range(3, 10)), f['arm_q'], 1e-6, .12
        elif f['mode'] == 'carry':
            indices, target, tolerance = list(range(3, 12)), f['arm_q']+f['gripper_q'], 1e-6
            execution = [.12]*7+[.004]*2
        else:
            indices, target, tolerance, execution = [0, 1, 2], f['base_q'], 1e-6, [.012]*3
        name = phase+'_fixed'
        plan['predicates'][name] = {'kind': 'configuration_equals', 'indices': indices,
            'target': target, 'tolerance': tolerance, 'execution_tolerance': execution}
        refs.append(name)
        plan['phases'].append({'id': phase, 'invariants': refs})
    return plan


def summarize_action(observations, action, source_pose, target):
    key = action['object_id']
    active = [r for r in observations if r['phase'].startswith(action['id']+'.')]
    carry = [r['objects'][key] for r in active if r['mode'] == 'carry']
    held = [r['objects'][key] for r in active if r['mode'] in ('hold', 'carry')]
    settled = [r['objects'][key] for r in active if r['mode'] == 'release'][-50:]
    bilateral = lambda r: min(r['contacts_N'][s] for s in ('left_finger', 'right_finger')) > .05
    lifted = any(r['objects'][key]['pose'][2][3] >= np.array(source_pose)[2, 3]+.10 and bilateral(r['objects'][key]) for r in active)
    held_bilateral = bool(held and all(bilateral(r) for r in held))
    carried_clear = bool(carry and all(max((v for s,v in r['contacts_N'].items()
        if s not in ('left_finger','right_finger')),default=0.) < .05 for r in carry))
    slip = max((float(np.linalg.norm(np.asarray(r['relative_translation_m'])-held[0]['relative_translation_m'])) for r in held), default=None)
    stable = bool(len(settled) >= 20 and all(r['contacts_N'][action['destination_support']] > .05
        and max(r['contacts_N'][s] for s in ('left_finger', 'right_finger')) < .05
        and r['speed_m_s'] < .01 and r['angular_speed_rad_s'] < .05 for r in settled))
    final = observations[-1]['objects'][key]
    error = float(np.linalg.norm(np.array(final['pose'])[:3, 3]-target))
    displacement = float(np.linalg.norm(np.array(final['pose'])[:3, 3]-np.array(source_pose)[:3, 3]))
    return {'lifted_with_bilateral_contact': lifted, 'bilateral_throughout_hold_and_carry': held_bilateral,
        'carried_without_table_support': carried_clear, 'hold_carry_slip_m': slip,
        'released_and_settled_on_destination': stable, 'final_position_error_m': error,
        'object_displacement_m': displacement,
        'satisfied': bool(lifted and held_bilateral and carried_clear and stable
                          and slip is not None and slip <= .01 and error <= .02 and displacement >= .2)}


def measured_precondition(start, row, q, expected_world_sha256):
    errors, angles, speeds, supports = {}, {}, {}, {}
    for key, X in start['world_objects'].items():
        actual, reference = np.array(row['objects'][key]['pose']), np.array(X)
        errors[key] = float(np.linalg.norm(actual[:3, 3]-reference[:3, 3]))
        angles[key] = float(np.arccos(np.clip((np.trace(reference[:3, :3].T@actual[:3, :3])-1)/2, -1, 1)))
        speeds[key] = [row['objects'][key]['speed_m_s'], row['objects'][key]['angular_speed_rad_s']]
        supports[key] = row['objects'][key]['contacts_N'][start['world_supports'][key]] > .05
    difference = np.abs(np.asarray(q)-configuration(start))
    return {'action': row['phase'].split('.')[0], 'time_s': row['time_s'],
        'object_position_errors_m': errors, 'object_orientation_errors_rad': angles,
        'object_speeds_m_s_rad_s': speeds, 'expected_support_satisfied': supports,
        'expected_world_sha256': expected_world_sha256, 'maximum_arm_error_rad': float(max(difference[3:10])),
        'maximum_base_error_m_rad': float(max(difference[:3])),
        'satisfied': bool(max(errors.values()) <= .02 and max(angles.values()) <= .08
            and all(v[0] < .02 and v[1] < .1 for v in speeds.values()) and all(supports.values())
            and max(difference[:3]) <= .015 and max(difference[3:10]) <= .12)}


def execute_reference(config, spec, frames, plan, selected, *, visualize=False, grip=True):
    from .execute_delivery_dynamics import execute
    observations, gates = [], []
    observed_actions, room = set(), None
    min_room = float('inf')
    from .robot_keep_clear import RobotKeepClearEvaluator
    keep_clear_monitor, reserved_min = None, None
    from .object_exclusion import ObjectExclusionEvaluator
    exclusion_monitor = None
    exclusion_min = None
    from .object_corridor import ObjectCorridorEvaluator
    corridor_monitor = None
    from .object_relative import ObjectRelativeEvaluator
    relative_monitor, relative_min, relative_samples = None, None, 0
    from .conditional_permission import PermissionExecutionMonitor
    permission_monitor = None
    from .ordered_visitation import OrderedVisitMonitor
    visit_monitor = None
    corridor_min, corridor_samples = None, 0
    from .room_entry import EntryExecutionGate
    entry_gate = EntryExecutionGate(config,frames)
    expected = {row['action']: row['report']['input_world_sha256'] for row in selected}
    # Expected world poses at action start include every earlier placement.
    starts = {a['id']: next(f for f in frames if f['phase'] == a['id']+'.arrive') for a in spec['actions']}
    def observe(scene, row):
        nonlocal room, min_room, exclusion_monitor, exclusion_min, corridor_monitor, corridor_min, corridor_samples, permission_monitor, visit_monitor, keep_clear_monitor, reserved_min, relative_monitor, relative_min, relative_samples
        if room is None:
            room = room_oracle(scene)
            exclusion_monitor = ObjectExclusionEvaluator(scene)
            corridor_monitor = ObjectCorridorEvaluator(scene)
            relative_monitor = ObjectRelativeEvaluator(scene)
            permission_monitor = PermissionExecutionMonitor(scene)
            visit_monitor = OrderedVisitMonitor(scene)
            keep_clear_monitor = RobotKeepClearEvaluator(scene)
        if visualize:
            from .object_relative import update_relative_visuals
            update_relative_visuals(scene, row['time_s'])
        reserved = keep_clear_monitor.current()
        if reserved['minimum_residual_m'] is not None:
            v = reserved['minimum_residual_m']
            reserved_min = v if reserved_min is None else min(reserved_min, v)
        if not reserved['satisfied']:
            raise TransportRejected('online_robot_keep_clear_violation', dict(time_s=row['time_s'], phase=row['phase'], **reserved))
        visits = visit_monitor.observe({key:v['pose'] for key,v in row['objects'].items()}, row['phase'], row['time_s'])
        if not visits['satisfied']:
            raise TransportRejected('online_ordered_visitation_violation', visits)
        permission = permission_monitor.observe(row)
        if not permission['satisfied']:
            raise TransportRejected('online_conditional_permission_violation',
                dict(time_s=row['time_s'], phase=row['phase'], **permission))
        relative = relative_monitor.evaluate({key:v['pose'] for key,v in row['objects'].items()}, row['phase'])
        if relative['minimum_residual_m'] is not None:
            relative_samples += 1
            v = relative['minimum_residual_m']
            relative_min = v if relative_min is None else min(relative_min, v)
        if not relative['satisfied']:
            raise TransportRejected('online_object_relative_violation',
                dict(time_s=row['time_s'], phase=row['phase'], **relative))
        corridor = corridor_monitor.evaluate({key:v['pose'] for key,v in row['objects'].items()}, row['phase'])
        if corridor['minimum_residual_m'] is not None:
            corridor_samples += 1
            v = corridor['minimum_residual_m']
            corridor_min = v if corridor_min is None else min(corridor_min, v)
        if not corridor['satisfied']:
            raise TransportRejected('online_object_corridor_violation', corridor)
        exclusion = exclusion_monitor.evaluate({key:v['pose'] for key,v in row['objects'].items()})
        if exclusion['minimum_residual_m'] is not None:
            v=exclusion['minimum_residual_m']
            exclusion_min=v if exclusion_min is None else min(exclusion_min,v)
        if not exclusion['satisfied']:
            raise TransportRejected('online_object_exclusion_violation', exclusion)
        robot_room = room.query_current()['residual_m']
        q_now = np.concatenate([scene.plant.GetPositions(scene.plant_context,instance)
            for instance in (scene.base_instance,scene.arm_instance,scene.gripper_instance)])
        entering = entry_gate.observe(config,row,q_now,robot_room)
        if not entering:
            min_room = min(min_room,robot_room)
        for key, body in scene.object_bodies.items():
            min_room = min(min_room, body_room_residual(scene, body,
                RigidTransform(np.array(row['objects'][key]['pose'])), room))
        action_id = row['phase'].split('.')[0]
        if not entering and action_id not in observed_actions:
            observed_actions.add(action_id)
            start = starts[action_id]
            q = np.concatenate([scene.plant.GetPositions(scene.plant_context, instance)
                    for instance in (scene.base_instance, scene.arm_instance, scene.gripper_instance)])
            gate = measured_precondition(start, row, q, expected[action_id])
            gates.append(gate)
            if not gate['satisfied']:
                raise TransportRejected('measured_next_action_precondition_failed', gate)
        if min_room < -1e-7:
            raise TransportRejected('online_robot_or_object_room_violation', {'minimum_residual_m': min_room})
        observations.append(row)
    physics, scene = execute(config, frames, time_scale=spec['time_scale'], visualize=visualize,
                             grip=grip, observation_hook=observe)
    semantic = SemanticPlanEvaluator(scene, plan, execution=True).audit(physics['trace'], require_shared_boundaries=False)
    rows = {r['action']: r for r in selected}
    checks = {a['id']: summarize_action(observations, a, rows[a['id']]['report']['source_pose'],
                                      np.asarray(rows[a['id']]['placement'])) for a in spec['actions']}
    result = {'satisfied': all(c['satisfied'] for c in checks.values()) and semantic['satisfied']
        and physics['unexpected_contact_count'] == 0 and physics['actuator_effort_limits_satisfied'] and entry_gate.report()['satisfied']
        and visit_monitor.report()['satisfied'],
        'actions': checks, 'measured_action_preconditions': gates, 'semantic_audit': semantic,
        'online_samples': len(observations), 'minimum_robot_and_objects_room_residual_m': min_room,
        'minimum_object_exclusion_residual_m': exclusion_min,
        'minimum_object_corridor_residual_m': corridor_min, 'object_corridor_samples': corridor_samples,
        'minimum_object_relative_residual_m': relative_min, 'object_relative_samples': relative_samples,
        'minimum_robot_keep_clear_residual_m': reserved_min,
        'room_entry_gate': entry_gate.report(),
        'conditional_permissions': permission_monitor.report(),
        'ordered_visits': visit_monitor.report(),
        'robot_room_scope': 'After room entry handoff; object room constraints remain active throughout' if entry_gate.enabled else 'All phases',
        'unexpected_contacts': physics['unexpected_contact_count'], 'effort_limits': physics['actuator_effort_limits_satisfied'],
        'object_pose_resets_after_initialization': 0, 'cup_attachment': False,
        'scope': 'Single continuous free-body SAP execution; measured boundary gates; ideal planar base; finite checks'}
    return result, physics, scene


def run(output, config_path=PROJECT_ROOT/'configs/stateful_transport_grounding.yaml', *, strategy='backtrack', visualize=False):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    spec = load_scene_config(config_path)
    cfg = make_config(spec)
    (root/'protocol.json').write_text(json.dumps({'spec': spec, 'config': cfg, 'strategy': strategy,
        'planning_order': ['manipulation_and_transport','room_entry','compose','execute'] if cfg.get('entry_navigation') else ['manipulation_and_transport','execute'],
        'source_sha256': {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PROJECT_ROOT/'src').glob('*.py'))}}, indent=2))
    from functools import partial
    planning_cfg = copy.deepcopy(cfg)
    if cfg.get('entry_navigation'):
        planning_cfg['robot']['start'] = cfg['entry_navigation']['handoff_base']
    def accept_entry(operations, folder):
        from .room_entry import plan_entry
        entry_cfg = copy.deepcopy(cfg)
        entry_cfg['entry_navigation']['handoff_base'] = copy.deepcopy(operations[0]['base_q'])
        return plan_entry(entry_cfg, operations, folder)
    search_result = search(planning_cfg, spec, root, strategy=strategy,
                           ground=partial(ground_action, pickup_cache={}),
                           entry_planner=accept_entry if cfg.get('entry_navigation') else None)
    if cfg.get('entry_navigation'):
        search_result['initial_world_scope'] = 'Expected world at room-entry handoff; physical execution starts at the configured Room 1 pose'
    compact = {k: v for k, v in search_result.items() if k != 'frames'}
    (root/'search.json').write_text(json.dumps(compact, indent=2))
    result = {'satisfied': False, 'reference_satisfied': search_result['satisfied']}
    if search_result['satisfied']:
        frames = search_result['frames']
        if cfg.get('entry_navigation'):
            import shutil
            chosen = search_result['selected'][0]
            cfg['entry_navigation']['handoff_base'] = copy.deepcopy(chosen['pickup_candidate']['base_q'])
            result['entry_reference_satisfied'] = chosen['entry_report']['satisfied']
            # Preserve the established artifact location without replanning.
            shutil.copytree(root/f"branch_{chosen['branch']:03d}"/'room_entry', root/'room_entry')
        plan = semantic_plan(cfg, frames)
        audit = SemanticPlanEvaluator(build_scene(config=cfg, visualize=False), plan).audit(frames)
        if not audit['satisfied']:
            raise TransportRejected('assembled_semantic_plan_failed', audit)
        (root/'reference.json').write_text(json.dumps(frames))
        (root/'plan.json').write_text(json.dumps(plan, indent=2))
        (root/'config.json').write_text(json.dumps(cfg, indent=2))
        try:
            result, physics, scene = execute_reference(cfg, spec, frames, plan, search_result['selected'], visualize=visualize)
            result['reference_satisfied'] = True
            (root/'physics.json').write_text(json.dumps(physics))
            if visualize:
                (root/'scene.html').write_text(add_playback_controls(scene.meshcat.StaticHtml()))
        except Exception as exc:
            result.update(reason=str(exc), details=getattr(exc, 'details', {}))
    (root/'result.json').write_text(json.dumps(result, indent=2))
    print({k: result.get(k) for k in ('satisfied', 'reference_satisfied', 'reason')}, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--config', type=Path, default=PROJECT_ROOT/'configs/stateful_transport_grounding.yaml')
    p.add_argument('--strategy', choices=['greedy', 'backtrack'], default='backtrack')
    p.add_argument('--html', action='store_true')
    args = p.parse_args()
    if not run(args.output, args.config, strategy=args.strategy, visualize=args.html)['satisfied']:
        raise SystemExit(1)
