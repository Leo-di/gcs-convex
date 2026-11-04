"""Carry-scoped ordered full-object visits, with conservative future-zone exclusion.

A visit requires the complete collision volume inside a box. Until a visit is
complete, every later box remains forbidden to that object. Progress persists
for the named action; it never comes from a robot command or a phase label.
"""
import copy
import json
from pathlib import Path
import numpy as np
from pydrake.math import RigidTransform
from .object_corridor import shape_containment_residual
from .object_exclusion import shape_box_residual


def validate_visits(specs, object_ids, actions=None):
    if not isinstance(specs, list):
        raise ValueError('Ordered visits must be a list')
    ids, action_ids = set(), set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {'id', 'object_id', 'action_id', 'mode', 'regions', 'margin_m'}:
            raise ValueError('Explicit ordered visit id/object/action/mode/regions/margin required')
        if not isinstance(spec['id'], str) or not spec['id'] or spec['id'] in ids:
            raise ValueError('Unique nonempty ordered visit IDs required')
        ids.add(spec['id'])
        action = spec['action_id']
        if (not isinstance(action, str) or not action or any(c in action for c in './\\')
                or action in action_ids or spec['object_id'] not in object_ids or spec['mode'] != 'carry'
                or not np.isfinite(spec['margin_m']) or spec['margin_m'] < 0):
            raise ValueError('One sequence per action, known object, carry mode and nonnegative finite margin required')
        action_ids.add(action)
        if actions is not None and actions.get(action) != spec['object_id']:
            raise ValueError('Ordered visits must name the object carried by their action')
        if not isinstance(spec['regions'], list) or len(spec['regions']) < 2:
            raise ValueError('At least two ordered regions required')
        names = set()
        for region in spec['regions']:
            if not isinstance(region, dict) or set(region) != {'id', 'lower_xyz', 'upper_xyz'}:
                raise ValueError('Ordered region needs id and 3D bounds')
            if not isinstance(region['id'], str) or not region['id'] or region['id'] in names:
                raise ValueError('Unique nonempty region IDs required')
            names.add(region['id'])
            lo, hi = np.asarray(region['lower_xyz'], float), np.asarray(region['upper_xyz'], float)
            if (lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi]))
                    or np.any(hi-lo <= 2*spec['margin_m'])):
                raise ValueError('Finite ordered region bounds wider than twice margin required')


class OrderedVisitMonitor:
    def __init__(self, scene, specs=None):
        self.scene = scene
        self.specs = scene.config.get('ordered_visits', []) if specs is None else specs
        validate_visits(self.specs, scene.object_bodies)
        inspector = scene.scene_graph.model_inspector()
        self.geometry = {key: [(inspector.GetShape(g), inspector.GetPoseInFrame(g))
            for g in scene.plant.GetCollisionGeometriesForBody(scene.object_bodies[key])]
            for key in {s['object_id'] for s in self.specs}}
        if any(not g for g in self.geometry.values()):
            raise ValueError('Ordered visit object requires collision geometry')
        self.progress = {s['id']: 0 for s in self.specs}
        self.started, self.finished = set(), set()
        self.events, self.failures = [], []
        self.samples = 0

    def residual(self, spec, region, pose, *, inside):
        fn = shape_containment_residual if inside else shape_box_residual
        return min(fn(shape, pose@offset, np.asarray(region['lower_xyz']),
            np.asarray(region['upper_xyz']), spec['margin_m'])
            for shape, offset in self.geometry[spec['object_id']])

    def observe(self, poses, phase, time_s=None):
        self.samples += 1
        if self.failures:
            return self.report(require_complete=False)
        for spec in self.specs:
            key = spec['id']
            if phase != spec['action_id']+'.carry':
                if key in self.started and key not in self.finished:
                    self.finished.add(key)
                    if self.progress[key] != len(spec['regions']):
                        self.failures.append(dict(id=key, reason='carry_ended_before_all_visits', time_s=time_s))
                continue
            if key in self.finished:
                self.failures.append(dict(id=key, reason='carry_reentered_after_completion', time_s=time_s))
                continue
            self.started.add(key)
            if spec['object_id'] not in poses:
                raise ValueError('Missing ordered visit object pose')
            value = poses[spec['object_id']]
            pose = value if isinstance(value, RigidTransform) else RigidTransform(np.asarray(value, float))
            index = self.progress[key]
            # Check future regions first: one sample cannot count simultaneous
            # occupancy of A and B as proof that A preceded B.
            for future in spec['regions'][index+1:]:
                residual = self.residual(spec, future, pose, inside=False)
                if residual < -1e-8:
                    self.failures.append(dict(id=key, reason='future_region_entered_before_predecessor',
                        region=future['id'], next_region=spec['regions'][index]['id'],
                        residual_m=residual, time_s=time_s))
            if self.failures:
                continue
            if index < len(spec['regions']):
                region = spec['regions'][index]
                residual = self.residual(spec, region, pose, inside=True)
                if residual >= -1e-8:
                    self.progress[key] += 1
                    self.events.append(dict(id=key, region=region['id'], index=index,
                        time_s=time_s, containment_residual_m=residual,
                        object_xyz=pose.translation().tolist()))
        return self.report(require_complete=False)

    def report(self, *, require_complete=True):
        complete = all(self.progress[s['id']] == len(s['regions']) for s in self.specs)
        return dict(satisfied=not self.failures and (complete or not require_complete),
            complete=complete, progress=dict(self.progress), events=list(self.events),
            failures=list(self.failures), samples=self.samples,
            scope='Carry only; complete object containment counts a visit; future boxes excluded until predecessor completed; sampled observations')


def checkpoint_policy(carry_spec):
    fractions = carry_spec.get('ordered_checkpoint_fractions', [.5, .2, .8])
    budget = carry_spec.get('ordered_max_leg_attempts', 36)
    if (not isinstance(fractions, list) or not fractions or len(fractions) > 7
            or any(type(f) not in (int, float) or not np.isfinite(f) or not 0 < f < 1 for f in fractions)
            or len(set(fractions)) != len(fractions)):
        raise ValueError('Checkpoint fractions must be 1 to 7 unique finite numbers strictly inside (0,1)')
    if type(budget) is not int or not 1 <= budget <= 500:
        raise ValueError('Ordered leg attempt budget must be an integer from 1 to 500')
    rounds = carry_spec.get('ordered_checkpoint_refinement_rounds', 0)
    if type(rounds) is not int or not 0 <= rounds <= 2:
        raise ValueError('Ordered checkpoint refinement rounds must be an integer from 0 to 2')
    return fractions, budget


def checkpoint_refinement_levels(carry_spec):
    fractions, _ = checkpoint_policy(carry_spec)
    levels = [list(fractions)]
    for _ in range(carry_spec.get('ordered_checkpoint_refinement_rounds', 0)):
        knots = [0., *sorted(levels[-1]), 1.]
        mids = [(a+b)/2 for a,b in zip(knots,knots[1:])]
        # Keep earlier candidates first, then boundary gaps, then interior gaps.
        additions = [mids[0], mids[-1], *mids[1:-1]]
        levels.append(list(dict.fromkeys([*levels[-1], *additions])))
    return levels


def checkpoint_candidates(monitor, spec, region, initial_pose, start, carry_spec, *, fractions=None):
    """Erode the box by every oriented collision shape and both margins.

    Return deterministic XY samples in the remaining body-origin rectangle;
    this permits asymmetric shapes/offsets and never shrinks the object.
    """
    from .whole_robot_room import shape_support
    if fractions is None:
        fractions, _ = checkpoint_policy(carry_spec)
    rotation = initial_pose.rotation()
    lower, upper = np.asarray(region['lower_xyz'], float).copy(), np.asarray(region['upper_xyz'], float).copy()
    margin = spec['margin_m']+carry_spec['margin_m']
    for shape, offset in monitor.geometry[spec['object_id']]:
        geometry_pose = RigidTransform(rotation, [0., 0., 0.])@offset
        center, R = geometry_pose.translation(), geometry_pose.rotation().matrix()
        for j, axis in enumerate(np.eye(3)):
            lower[j] = max(lower[j], region['lower_xyz'][j]+margin-center[j]+shape_support(shape, R.T@(-axis)))
            upper[j] = min(upper[j], region['upper_xyz'][j]-margin-center[j]-shape_support(shape, R.T@axis))
    z = initial_pose.translation()[2]
    if np.any(lower > upper) or z < lower[2]-1e-10 or z > upper[2]+1e-10:
        return []
    offset_xy = initial_pose.translation()[:2]-np.asarray(start)[:2]
    rows = []
    for fy in fractions:
        for fx in fractions:
            center = lower[:2]+np.array([fx, fy])*(upper[:2]-lower[:2])
            base = np.r_[center-offset_xy, start[2]]
            pose = RigidTransform(rotation, [*center, z])
            if (monitor.residual(spec, region, pose, inside=True) >= carry_spec['margin_m']-1e-10
                    and not any(np.linalg.norm(base-r) < 1e-10 for r in rows)):
                rows.append(base)
    return rows


def clone_monitor(monitor):
    # Keep the owning scene/borrowed C++ shapes, but isolate temporal history.
    other = copy.copy(monitor)
    other.progress = dict(monitor.progress)
    other.started, other.finished = set(monitor.started), set(monitor.finished)
    other.events, other.failures = list(monitor.events), list(monitor.failures)
    return other


def plan_ordered_translation(scene, start, goal, arm, gripper, X_GC, carry_spec, artifact, *, action_id):
    """Backtrack over finite region-interior candidates and accepted GCS legs."""
    from .payload_translation_bridge import plan_payload_translation, PayloadBridgeRejected
    from .build_scene import set_robot_configuration
    specs = [s for s in scene.config.get('ordered_visits', []) if s['action_id'] == action_id]
    if not specs:
        return plan_payload_translation(scene, start, goal, arm, gripper, X_GC, carry_spec, artifact, action_id=action_id)
    fractions, budget = checkpoint_policy(carry_spec)
    monitor = OrderedVisitMonitor(scene, specs)
    spec = specs[0]
    start, goal = np.asarray(start), np.asarray(goal)
    set_robot_configuration(scene, start, arm, gripper)
    initial_pose = scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body) @ X_GC
    checkpoints = [checkpoint_candidates(monitor, spec, region, initial_pose, start, carry_spec)
                   for region in spec['regions']]
    for region, candidates in zip(spec['regions'], checkpoints):
        if not candidates:
            raise PayloadBridgeRejected('ordered_checkpoint_not_contained', dict(region=region['id'],
                scope='No candidate in full-shape-eroded region at the fixed lift pose; no unrestricted feasibility claim'))
    original = scene.config.get('object_exclusions', [])
    attempts, refinement_levels = [], []
    level = 0
    root = Path(artifact).parent
    def save_attempts():
        (root/'ordered_search.json').write_text(json.dumps(dict(attempts=attempts, budget=budget,
            fractions=fractions, refinement_levels=refinement_levels, candidate_counts=[len(c) for c in checkpoints]), indent=2))
    def visit(index, current, history, paths, reports):
        if index == len(checkpoints)+1:
            return paths, reports, history
        candidates = checkpoints[index] if index < len(checkpoints) else [goal]
        for candidate_index, endpoint in enumerate(candidates):
            if len(attempts) >= budget:
                return None
            number = len(attempts)
            future = spec['regions'][index+1:]
            scene.config['object_exclusions'] = copy.deepcopy(original)+[
                dict(id=f'__ordered_{spec["id"]}_{j}', object_id=spec['object_id'],
                    lower_xyz=r['lower_xyz'], upper_xyz=r['upper_xyz'], margin_m=spec['margin_m'])
                for j, r in enumerate(future)]
            folder = root/f'ordered_attempt_{number:03d}'
            folder.mkdir(exist_ok=False)
            row = dict(attempt=number, refinement_level=level, leg=index, candidate_index=candidate_index,
                start_base=current.tolist(), goal_base=endpoint.tolist(),
                locked_regions=[r['id'] for r in future], local_satisfied=False)
            attempts.append(row)
            trial = clone_monitor(history)
            try:
                path, report = plan_payload_translation(scene, current, endpoint, arm, gripper, X_GC,
                    carry_spec, folder/'payload_bridge.json', action_id=action_id)
                for a, b in zip(path, path[1:]):
                    for u in np.linspace(0, 1, max(2, int(np.ceil(np.linalg.norm(b[:2]-a[:2])/.0025))+1)):
                        q = (1-u)*a+u*b
                        pose = RigidTransform(initial_pose.rotation(), initial_pose.translation()+np.r_[q[:2]-start[:2], 0.])
                        checked = trial.observe({spec['object_id']: pose}, action_id+'.carry')
                        if not checked['satisfied']:
                            raise PayloadBridgeRejected('ordered_leg_acceptance_failed', checked)
                if index < len(checkpoints) and trial.progress[spec['id']] < index+1:
                    raise PayloadBridgeRejected('ordered_checkpoint_visit_missing', trial.report())
                row['local_satisfied'] = True
                selected = dict(leg=index, attempt=number, refinement_level=level, candidate_index=candidate_index,
                    goal_base=endpoint.tolist(), locked_regions=row['locked_regions'], **report)
                answer = visit(index+1, endpoint, trial, paths+[path if not paths else path[1:]], reports+[selected])
                if answer is not None:
                    save_attempts()
                    return answer
                row['downstream_rejected'] = True
            except RuntimeError as exc:
                row.update(reason=str(exc), stage=getattr(exc, 'stage', 'gcs_endpoint_or_path_rejected'),
                    details=getattr(exc, 'details', {}))
            save_attempts()
        return None
    answer = None
    try:
        for level, fractions in enumerate(checkpoint_refinement_levels(carry_spec)):
            if len(attempts) >= budget:
                break
            checkpoints = [checkpoint_candidates(monitor, spec, region, initial_pose, start, carry_spec,
                            fractions=fractions) for region in spec['regions']]
            entry = dict(level=level, fractions=list(fractions), candidate_counts=[len(c) for c in checkpoints],
                         attempt_begin=len(attempts), satisfied=False)
            refinement_levels.append(entry)
            # Restore the input state before trying the next planning branch.
            answer = visit(0, start, monitor, [], [])
            entry.update(attempt_end=len(attempts), satisfied=answer is not None,
                termination='accepted' if answer is not None else
                    ('attempt_budget_exhausted' if len(attempts)>=budget else 'finite_candidates_exhausted'))
            save_attempts()
            if answer is not None:
                break
    finally:
        scene.config['object_exclusions'] = original
    if answer is None:
        details = dict(attempts=attempts, budget=budget, refinement_levels=refinement_levels, candidate_counts=[len(c) for c in checkpoints],
            scope='Finite region candidates and fixed arm/yaw cover exhausted; not general infeasibility')
        Path(artifact).write_text(json.dumps(dict(satisfied=False, **details), indent=2))
        raise PayloadBridgeRejected('ordered_candidate_search_exhausted', details)
    paths, reports, accepted = answer
    audit = accepted.report()
    if not audit['satisfied']:
        raise PayloadBridgeRejected('ordered_translation_incomplete', audit)
    # Keep the existing per-leg artifact paths for viewers and downstream tools.
    import shutil
    for report in reports:
        shutil.copytree(root/f"ordered_attempt_{report['attempt']:03d}", root/f"ordered_leg_{report['leg']:02d}")
    path = np.vstack(paths)
    report = dict(satisfied=True, method='ordered_region_candidate_search_with_stage_local_official_Drake_GCS',
        legs=reports, ordered_visits=audit, checkpoint_search=dict(attempts=attempts, budget=budget,
            fractions=fractions, refinement_levels=refinement_levels, candidate_counts=[len(c) for c in checkpoints]),
        minimum_clearance_m=min(r['minimum_clearance_m'] for r in reports),
        base_path_length_m=float(np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1).sum()),
        scope='Sequential fixed-arm/yaw GCS with finite eroded-region candidates and backtracking; no global optimality or completeness claim')
    Path(artifact).write_text(json.dumps(report, indent=2))
    return path, report
