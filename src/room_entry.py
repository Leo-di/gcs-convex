"""Accepted empty-hand room entry before stateful manipulation.

Navigation has tucked/open invariants and a whole-body Room 2 exit gate.
It does not impose Room 2 containment while crossing the doorway. Existing
manipulation constraints remain active after the exact shared boundary.
"""
import copy
import json
from pathlib import Path
import numpy as np
from .semantic_plan import configuration, SemanticPlanEvaluator
from .accepted_motion import require_endpoint

ENTRY_PHASE = 'room_entry.navigate'
ENTRY_MODE = 'enter_room'


def compress_collinear(path, tolerance=1e-10):
    """Remove only subdivisions of the same numerical straight segment."""
    points = np.asarray(path,float)
    if points.ndim != 2 or len(points)<2 or not np.all(np.isfinite(points)):
        raise ValueError('Finite navigation polyline required')
    kept = {0,len(points)-1}; pending = [(0,len(points)-1)]
    while pending:
        a,b = pending.pop()
        if b<=a+1:continue
        delta = points[b]-points[a]; norm = delta@delta
        u = np.zeros(b-a-1) if norm==0 else np.clip((points[a+1:b]-points[a])@delta/norm,0,1)
        errors = np.linalg.norm(points[a+1:b]-(points[a]+u[:,None]*delta),axis=1)
        if max(errors)>tolerance:
            i = a+1+int(np.argmax(errors));kept.add(i);pending.extend([(a,i),(i,b)])
    return points[sorted(kept)]


def validate_entry(spec):
    if 'entry_navigation' not in spec:
        return
    entry = spec['entry_navigation']
    if not isinstance(entry, dict) or set(entry) != {'handoff_base', 'method'}:
        raise ValueError('Entry requires explicit handoff_base and method')
    q = np.asarray(entry['handoff_base'], float)
    if q.shape != (3,) or not np.all(np.isfinite(q)) or entry['method'] != 'rotation_safe_gcs':
        raise ValueError('Finite entry handoff and rotation_safe_gcs method required')
    if any(a['id'] == 'room_entry' for a in spec['actions']):
        raise ValueError('room_entry action ID is reserved')
    if np.max(abs(q-np.asarray(spec['actions'][0]['pickup_base']))) > 1e-8:
        raise ValueError('Current entry adapter hands off at the first pickup base')


def is_entry(config, frame):
    reserved = frame['phase'] == ENTRY_PHASE or frame.get('mode') == ENTRY_MODE
    if reserved and (not config.get('entry_navigation') or frame['phase'] != ENTRY_PHASE or frame.get('mode') != ENTRY_MODE):
        raise ValueError('Entry phase/mode/config mismatch')
    return reserved


def join_entry(prefix, operations):
    if not prefix or not operations or any(f['phase'] != ENTRY_PHASE for f in prefix):
        raise ValueError('Accepted nonempty entry and operation segments required')
    require_endpoint(operations[0], configuration(prefix[-1]), 'Room entry handoff')
    if set(prefix[-1]['world_objects']) != set(operations[0]['world_objects']):
        raise ValueError('Entry handoff object identities differ')
    pose_error = max(float(np.max(abs(np.asarray(X)-np.asarray(operations[0]['world_objects'][key]))))
                     for key, X in prefix[-1]['world_objects'].items())
    if pose_error > 1e-6 or prefix[-1]['world_supports'] != operations[0]['world_supports']:
        raise ValueError('Object pose/support discontinuity at room entry handoff')
    return copy.deepcopy(prefix+operations), {'configuration_error': float(np.max(abs(configuration(prefix[-1])-configuration(operations[0])))),
        'object_pose_error': pose_error, 'supports_match': True}


def plan_entry(config, operations, output):
    from .build_scene import build_scene, set_robot_configuration
    from .automatic_motion_bridge import plan_bridge
    from .whole_robot_room import WholeRobotRoomOracle
    from .green_room_gcs import append_motion
    from .collision_checker import evaluate_collision_status
    from .object_exclusion import ObjectExclusionEvaluator
    from .stateful_transport_grounding import semantic_plan
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    scene = build_scene(config=config, visualize=False)
    start = np.asarray(config['robot']['start'])
    boundary = operations[0]; goal = np.asarray(config['entry_navigation']['handoff_base'])
    arm = np.asarray(config['pick_task']['navigation_arm_q']); opened = scene.gripper_open_positions
    require_endpoint(boundary, np.r_[goal, arm, opened], 'Operation entry target')
    if (set(boundary['world_objects']) != set(scene.object_initial_poses)
            or any(np.max(abs(np.asarray(boundary['world_objects'][key])-X.GetAsMatrix4()))>1e-6
                   for key,X in scene.object_initial_poses.items())
            or any(v!='table' for v in boundary['world_supports'].values())):
        raise ValueError('Entry must preserve the initial object world and supports')
    bounds = config['semantic_regions']['room1']
    room1 = WholeRobotRoomOracle.from_xy_bounds(scene, bounds['x_bounds'], bounds['y_bounds'], margin=.02)
    initial = room1.query(np.r_[start, arm], opened)
    if not initial['satisfied']:
        raise RuntimeError('Entry initial whole robot must be inside Room 1')
    # Fail before asking the navigation planner when the exit is not legal.
    from .green_room_gcs import room_oracle
    endpoint = room_oracle(scene).query(np.r_[goal, arm], opened)
    if not endpoint['satisfied']:
        raise RuntimeError('Entry exit whole robot is not inside Room 2')
    plan, planner_report = plan_bridge(scene, start, goal, arm,
        config['entry_navigation']['method'], root/'bridge.json')
    # Remove numerical collinear subdivisions only, then audit emitted edges.
    path = compress_collinear(plan.configurations)
    frames = []
    append_motion(frames, path, np.repeat(arm[None,:], len(path), axis=0),
                  np.repeat(opened[None,:], len(path), axis=0), ENTRY_PHASE)
    for f in frames:
        f.update(mode=ENTRY_MODE, object_id=boundary['object_id'], payload_mode='free',
            world_objects=copy.deepcopy(boundary['world_objects']), world_supports=copy.deepcopy(boundary['world_supports']))
    require_endpoint(frames[0], np.r_[start, arm, opened], 'Entry initial configuration')
    joined, boundary_report = join_entry(frames, operations)
    semantic = SemanticPlanEvaluator(scene, semantic_plan(config, frames)).audit(frames)
    exclusions = ObjectExclusionEvaluator(scene)
    from .conditional_permission import ConditionalPermissionEvaluator
    permissions = ConditionalPermissionEvaluator(scene)
    minimum, count = float('inf'), 0
    for i, f in enumerate(frames):
        a, b = configuration(frames[max(0,i-1)]), configuration(f)
        n = max(1,int(np.ceil(np.max(abs(b-a)/np.r_[[.005]*3,[.01]*7,[.001]*2]))))
        for u in np.linspace(0,1,n+1):
            q = (1-u)*a+u*b
            status = evaluate_collision_status(scene,q[:3],arm_q=q[3:10],gripper_q=q[10:])
            minimum = min(minimum,status.minimum_signed_distance); count += 1
            if not status.collision_free or not exclusions.current(scene)['satisfied']:
                raise RuntimeError('Room entry emitted edge violates geometry or object exclusion')
            permission = permissions.current()
            if not permission['satisfied']:
                from .stateful_transport_grounding import TransportRejected
                (root/'permission_rejection.json').write_text(json.dumps(permission, indent=2))
                raise TransportRejected('empty_hand_entry_permission_rejected', permission)
    report = {'satisfied': semantic['satisfied'], 'planner': planner_report, 'semantic': semantic,
        'initial_room1': initial, 'exit_room2': endpoint, 'boundary': boundary_report,
        'collision_samples': count, 'minimum_collision_distance_m': minimum,
        'navigation_vertices_before':len(plan.configurations), 'navigation_vertices_after':len(path),
        'scope': 'Tucked/open navigation, full-body Room 2 exit, unchanged free objects; finite emitted-edge checks'}
    (root/'acceptance.json').write_text(json.dumps(report,indent=2))
    if not report['satisfied']:
        raise RuntimeError('Room entry semantic acceptance failed')
    (root/'reference.json').write_text(json.dumps(frames))
    return joined, report


class EntryExecutionGate:
    """Measured phase gate; the first operation must follow actual entry."""
    def __init__(self, config, frames):
        self.enabled = bool(config.get('entry_navigation'))
        self.samples = 0; self.passed = False; self.handoff = None
        self.minimum_room2_during_entry = None
        flags = [is_entry(config,f) for f in frames]
        if self.enabled:
            n = sum(flags)
            if not n or n == len(frames) or flags != [True]*n+[False]*(len(frames)-n):
                raise ValueError('Exactly one entry prefix must precede all operations')
            self.target = configuration(frames[n])
            require_endpoint(frames[n-1],self.target,'Measured entry gate reference')

    def observe(self, config, row, q, room_residual):
        entering = is_entry(config,row)
        if not self.enabled:
            return False
        tolerance = np.r_[[.015]*3,[.12]*7,[.004]*2]
        error = abs(np.asarray(q)-self.target)
        if entering:
            if self.passed:
                raise RuntimeError('Entry phase cannot recur after manipulation starts')
            if np.any(error[3:] > tolerance[3:]):
                raise RuntimeError('Measured entry must keep arm tucked and fingers open')
            self.samples += 1
            self.minimum_room2_during_entry = room_residual if self.minimum_room2_during_entry is None else min(self.minimum_room2_during_entry,room_residual)
        elif not self.passed:
            if not self.samples or room_residual < -1e-7 or np.any(error > tolerance):
                raise RuntimeError('Measured whole-body room entry handoff rejected')
            self.passed = True
            self.handoff = {'time_s':row['time_s'], 'room2_residual_m':room_residual,
                'maximum_configuration_errors':error.tolist(), 'satisfied':True}
        return entering

    def report(self):
        return {'enabled':self.enabled, 'satisfied':not self.enabled or self.passed, 'navigation_samples':self.samples,
            'minimum_room2_residual_during_navigation_m':self.minimum_room2_during_entry, 'handoff':self.handoff}
