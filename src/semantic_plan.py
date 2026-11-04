"""Evaluate a supplied semantic plan on grounded reference or measured motion.

Configuration predicates report their evaluation method. Interpolated path
checks are procedural finite sampling, never continuous-time certificates.
Unknown predicates, missing phases and exhausted audit budgets fail closed.
"""
from __future__ import annotations

import copy
import time
import numpy as np

from .whole_robot_room import WholeRobotRoomOracle, shape_support
from .build_scene import set_robot_configuration


def configuration(frame):
    q = np.r_[frame['base_q'], frame['arm_q'], frame['gripper_q']].astype(float)
    if q.shape != (12,) or not np.all(np.isfinite(q)):
        raise ValueError('Finite base[3], arm[7], gripper[2] required')
    return q


class SemanticPlanEvaluator:
    """Analytical room/posture predicates and conservative keep-clear checks.

    A keep-clear zone is a vertical XY prism. Per geometry, axis separation
    is sufficient for nonintersection; rejection may be conservative. It
    includes every robot collision volume and excludes the payload.
    """

    def __init__(self, scene, plan, *, bindings=None, execution=False):
        self.scene = scene
        self.plan = copy.deepcopy(plan)
        self.bindings = bindings or {}
        self.execution = execution
        if plan.get('version') != 1:
            raise ValueError('Supported semantic plan version is 1')
        self.specs = plan.get('predicates', {})
        phases = plan.get('phases', [])
        if not phases or not isinstance(self.specs, dict):
            raise ValueError('Semantic plan needs phases and a predicate mapping')
        self.phases = {}
        self.oracles = {}
        self.special = {}
        for name, spec in self.specs.items():
            kind = spec['kind']
            if kind == 'learned_room':
                from .learned_semantics import GuardedLearnedRoom
                self.special[name] = GuardedLearnedRoom(scene, spec)
            elif kind == 'robot_keep_clear':
                from .robot_keep_clear import RobotKeepClearEvaluator
                self.special[name] = RobotKeepClearEvaluator(scene, spec['zones'])
            elif kind == 'moving_keep_clear':
                from .dynamic_semantics import MovingKeepClear
                self.special[name] = MovingKeepClear(scene, spec)
            elif kind in ('whole_robot_in_room', 'base_in_room', 'keep_clear'):
                region = scene.config['semantic_regions'][spec['region']]
                self.oracles[name] = WholeRobotRoomOracle.from_xy_bounds(
                    scene, region['x_bounds'], region['y_bounds'], margin=float(spec.get('margin_m', 0.)))
            elif kind == 'configuration_equals':
                indices = spec['indices']
                if (not indices or any(type(i) is not int or not 0 <= i < 12 for i in indices)
                        or len(set(indices)) != len(indices)):
                    raise ValueError('Distinct configuration indices in [0,12) required')
                target = self.resolve(spec['target'])
                tolerance = self.tolerance(spec)
                if (target.shape != (len(indices),) or tolerance.shape != target.shape
                        or not np.all(np.isfinite(target)) or not np.all(np.isfinite(tolerance))
                        or np.any(tolerance < 0)):
                    raise ValueError('Finite matching target and nonnegative tolerances required')
            else:
                raise ValueError(f'Unsupported semantic evaluator: {kind}')
        for phase in phases:
            name = phase['id']
            if not isinstance(name, str) or not name or name in self.phases:
                raise ValueError('Distinct nonempty semantic phase IDs required')
            unknown = set(phase) - {'id', 'entry', 'exit', 'invariants'}
            if unknown:
                raise ValueError(f'Unknown phase fields: {unknown}')
            for scope in ('entry', 'exit', 'invariants'):
                refs = phase.get(scope, [])
                if not isinstance(refs, list) or any(ref not in self.specs for ref in refs):
                    raise ValueError(f'Unknown predicate in {name}.{scope}')
            self.phases[name] = phase

    def resolve(self, value):
        if isinstance(value, str):
            value = self.bindings[value]
        return np.asarray(value, float)

    def tolerance(self, spec):
        key = 'execution_tolerance' if self.execution and 'execution_tolerance' in spec else 'tolerance'
        value = np.asarray(spec[key], float)
        return np.broadcast_to(value, (len(spec['indices']),))

    def evaluate(self, name, q, *, time_s=None):
        spec = self.specs[name]
        if spec['kind'] == 'learned_room':
            return self.special[name].evaluate(q)
        if spec['kind'] == 'moving_keep_clear':
            return self.special[name].evaluate(q, time_s)
        if spec['kind'] == 'robot_keep_clear':
            set_robot_configuration(self.scene, q[:3], q[3:10], q[10:])
            result = self.special[name].current()
            return dict(satisfied=result['satisfied'], residual=result['minimum_residual_m']
                        if result['minimum_residual_m'] is not None else float('inf'),
                        method='whole_robot_conservative_3D_axis_separation', worst=result['worst'])
        if spec['kind'] == 'configuration_equals':
            errors = np.abs(q[spec['indices']] - self.resolve(spec['target']))
            residual = float(np.min(self.tolerance(spec) - errors))
            return {'satisfied': residual >= -1e-8, 'residual': residual,
                    'method': 'analytical_configuration', 'maximum_error': float(max(errors))}
        oracle = self.oracles[name]
        if spec['kind'] in ('whole_robot_in_room', 'base_in_room'):
            result = oracle.query(q[:10], q[10:])
            if spec['kind'] == 'base_in_room':
                residual = result['base_residual_m']
                return {'satisfied': residual >= -1e-8, 'residual': residual,
                        'method': 'analytical_base_shape_support',
                        'scope': 'Full base collision volumes; arm, gripper and payload excluded'}
            return {'satisfied': bool(result['satisfied']), 'residual': result['residual_m'],
                    'method': 'analytical_shape_support', 'worst': result['worst']}
        set_robot_configuration(self.scene, q[:3], q[3:10], q[10:])
        # For each robot volume, at least one separating corridor face must
        # exist. Different volumes may use different separating faces.
        residuals = []
        for body, shape, X_BG, _ in oracle.geometry:
            pose = self.scene.plant.EvalBodyPoseInWorld(self.scene.plant_context, body) @ X_BG
            gaps = [float(n @ pose.translation() - shape_support(shape, pose.rotation().matrix().T @ n)
                          - b - oracle.margin) for n, b in zip(oracle.normals, oracle.offsets)]
            residuals.append(max(gaps))
        residual = min(residuals)
        return {'satisfied': residual >= -1e-8, 'residual': residual,
                'method': 'conservative_axis_separation',
                'scope': 'All robot collision volumes outside corridor; payload excluded; may conservatively reject'}

    def audit(self, frames, *, require_shared_boundaries=True, max_samples=200000):
        if not frames:
            raise ValueError('Cannot validate an empty trajectory')
        if type(max_samples) is not int or max_samples < 1:
            raise ValueError('Positive integer audit sample budget required')
        steps = np.asarray(self.plan.get('audit_step', [.01, .01, .01] + [.01]*7 + [.001]*2), float)
        if steps.shape != (12,) or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
            raise ValueError('Twelve positive finite audit spacings required')
        qs = [configuration(f) for f in frames]
        dynamic = any(s['kind'] == 'moving_keep_clear' for s in self.specs.values())
        times = np.array([f.get('time_s', np.nan) for f in frames]) if dynamic else None
        time_step = float(self.plan.get('maximum_audit_time_step_s', .05))
        if dynamic and (not np.all(np.isfinite(times)) or np.any(np.diff(times) < 0)
                        or not np.isfinite(time_step) or time_step <= 0):
            raise ValueError('Moving semantics require finite nondecreasing timestamps and positive time spacing')
        blocks = []
        for i, f in enumerate(frames):
            if not blocks or blocks[-1][0] != f['phase']:
                blocks.append([f['phase'], i, i])
            else:
                blocks[-1][2] = i
        if [b[0] for b in blocks] != list(self.phases):
            raise ValueError('Trajectory phase order/coverage differs from supplied semantic plan')
        report = {'satisfied': True, 'samples': 0, 'violations': 0, 'counterexamples': [],
                  'phases': [], 'evaluation': 'procedural_interpolated_samples',
                  'scope': 'Finite checks of every supplied edge, including phase boundaries; no continuous-time certificate',
                  'source': 'measured_execution' if self.execution else 'planned_reference'}
        def check(refs, q, phase, scope, edge, alpha, phase_report, time_s=None):
            for name in refs:
                result = self.evaluate(name, q, time_s=time_s)
                stats = phase_report['predicates'].setdefault(name, {
                    'method': result['method'], 'checks': 0, 'minimum_residual': result['residual']})
                stats['checks'] += 1
                stats['minimum_residual'] = min(stats['minimum_residual'], result['residual'])
                for key in ('abstained', 'out_of_domain', 'surrogate_disagreed'):
                    if key in result:
                        stats[key] = stats.get(key, 0)+int(result[key])
                if not result['satisfied']:
                    report['violations'] += 1
                    phase_report['satisfied'] = report['satisfied'] = False
                    if len(report['counterexamples']) < 10:
                        report['counterexamples'].append(dict(phase=phase, scope=scope, predicate=name,
                            edge=edge, alpha=float(alpha), configuration=q.tolist(), result=result))
        for name, lo, hi in blocks:
            phase = self.phases[name]
            row = {'id': name, 'satisfied': True, 'predicates': {}}
            report['phases'].append(row)
            if lo and require_shared_boundaries and np.max(np.abs(qs[lo]-qs[lo-1])) > 1e-6:
                raise ValueError(f'Disconnected reference entering {name}; motion bridge required')
            check(phase.get('entry', []), qs[lo], name, 'entry', lo, 1., row, times[lo] if dynamic else None)
            check(phase.get('exit', []), qs[hi], name, 'exit', hi, 1., row, times[hi] if dynamic else None)
            for i in range(lo, hi+1):
                a, b = qs[max(0, i-1)], qs[i]
                count = max(1, int(np.ceil(np.max(np.abs(b-a)/steps))))
                if dynamic:
                    count = max(count, int(np.ceil((times[i]-times[max(0, i-1)])/time_step)))
                if report['samples'] + count + 1 > max_samples:
                    raise RuntimeError('Semantic trajectory audit sample budget exhausted')
                for alpha in np.linspace(0., 1., count+1):
                    report['samples'] += 1
                    time_s = (1-alpha)*times[max(0, i-1)]+alpha*times[i] if dynamic else None
                    check(phase.get('invariants', []), (1-alpha)*a+alpha*b, name, 'invariants', i, alpha, row, time_s)
        return report


def cup_semantic_plan(config):
    """Default supplied plan for the existing navigate/reach/grasp/lift task."""
    room = {'kind': 'whole_robot_in_room', 'region': 'room2',
            'margin_m': config['green_room']['margin_m']}
    phases = [{'id': 'navigate_to_pickup', 'invariants': ['tucked'], 'exit': ['inside']}]
    for name in ('approach_cup', 'close_gripper', 'lift_cup', 'whole_body_carry_to_dropoff'):
        phases.append({'id': name, 'entry': ['inside'], 'invariants': ['inside', 'docked'], 'exit': ['inside']})
    return {'version': 1, 'predicates': {
        'inside': room,
        'tucked': {'kind': 'configuration_equals', 'indices': list(range(3, 10)),
                   'target': 'tucked', 'tolerance': 1e-6, 'execution_tolerance': .12},
        'docked': {'kind': 'configuration_equals', 'indices': [0, 1, 2],
                   'target': 'dock', 'tolerance': 1e-6, 'execution_tolerance': [.01, .01, .01]}},
        'phases': phases}


def audit_cup_plan(scene, frames, dock, tucked, *, execution=False):
    plan = scene.config.get('grounding_semantic_plan')
    if plan is None:
        return None
    if not execution and any(s['kind'] == 'moving_keep_clear' for s in plan['predicates'].values()):
        interval = .05*scene.config.get('execution_dynamics', {}).get('reference_time_scale', 2.)
        frames = [dict(frames[0], time_s=0.)] + [dict(f, time_s=1.+i*interval) for i, f in enumerate(frames)] + [
            dict(frames[-1], time_s=1.+(len(frames)-1)*interval+2.)]
    return SemanticPlanEvaluator(scene, plan, bindings={'dock': dock, 'tucked': tucked},
        execution=execution).audit(frames, require_shared_boundaries=not execution)


def check_cup_goals(scene, dock, tucked, pick):
    """Semantic checks after IK and before either motion-planning problem."""
    evaluator = SemanticPlanEvaluator(scene, scene.config['grounding_semantic_plan'],
        bindings={'dock': dock, 'tucked': tucked})
    opened, closed = scene.gripper_open_positions, scene.gripper_closed_positions
    goals = {
        'navigate_to_pickup': [(tucked, opened)],
        'approach_cup': [(tucked, opened), (pick.pregrasp_q, opened), (pick.grasp_q, opened)],
        'close_gripper': [(pick.grasp_q, opened), (pick.grasp_q, closed)],
        'lift_cup': [(pick.grasp_q, closed), (pick.lift_q, closed)],
        'whole_body_carry_to_dropoff': [(pick.lift_q, closed)]}
    if list(evaluator.phases) != list(goals):
        raise ValueError('Cup adapter requires navigate/reach/close/lift/hold phases in order')
    checks = []
    for name, values in goals.items():
        phase = evaluator.phases[name]
        for scope, pairs in [('entry', values[:1]), ('exit', values[-1:]), ('invariants', values)]:
            for arm, gripper in pairs:
                for ref in phase.get(scope, []):
                    if evaluator.specs[ref]['kind'] == 'moving_keep_clear':
                        checks.append(dict(phase=name, scope=scope, predicate=ref,
                            result={'satisfied': True, 'deferred': True, 'reason': 'Requires scheduled trajectory timestamps'}))
                        continue
                    checks.append(dict(phase=name, scope=scope, predicate=ref,
                        result=evaluator.evaluate(ref, np.r_[dock, arm, gripper])))
    return {'satisfied': all(c['result']['satisfied'] for c in checks), 'checks': checks,
            'scope': 'Sampled IK goals only; complete motion and physical checks still required'}


def ground_sequence(stages, initial, *, connect, validate, max_attempts=100, time_budget_s=300.):
    """Bounded DFS over supplied grounded pieces with audited motion bridges.

    A stage has alternatives of frame sequences. ``connect`` must plan a
    bridge between complete configurations; ``validate`` audits assembled
    prefixes. A failed later stage backtracks to earlier choices. The caller
    supplies collision/mode semantics; no interpolated fallback is invented.
    This composes reference trajectories and makes no physical success claim.
    """
    if (not stages or type(max_attempts) is not int or max_attempts < 1
            or not np.isfinite(time_budget_s) or time_budget_s <= 0):
        raise ValueError('Nonempty stages and a positive attempt budget required')
    configuration(initial)
    attempts, events = 0, []
    began = time.monotonic()
    def visit(depth, prefix):
        nonlocal attempts
        if depth == len(stages):
            return prefix
        for index, piece in enumerate(stages[depth]):
            if attempts >= max_attempts or time.monotonic()-began >= time_budget_s:
                return None
            attempts += 1
            try:
                if not piece:
                    raise ValueError('Empty grounded piece')
                previous = prefix[-1] if prefix else initial
                start = configuration(piece[0])
                if np.max(np.abs(configuration(previous)-start)) > 1e-6:
                    bridge = connect(copy.deepcopy(previous), copy.deepcopy(piece[0]), depth)
                    if (not bridge or np.max(np.abs(configuration(bridge[0])-configuration(previous))) > 1e-6
                            or np.max(np.abs(configuration(bridge[-1])-start)) > 1e-6):
                        raise ValueError('Motion bridge does not preserve complete endpoints')
                else:
                    bridge = []
                candidate = prefix + copy.deepcopy(bridge) + copy.deepcopy(piece)
                if not validate(candidate, depth):
                    raise RuntimeError('Assembled prefix rejected')
                if time.monotonic()-began >= time_budget_s:
                    raise RuntimeError('Sequence wall-time budget exhausted')
                events.append({'depth': depth, 'alternative': index, 'status': 'prefix_passed', 'bridge_frames': len(bridge)})
                result = visit(depth+1, candidate)
                if result is not None:
                    return result
                events.append({'depth': depth, 'alternative': index, 'status': 'backtracked'})
            except Exception as exc:
                events.append({'depth': depth, 'alternative': index, 'status': 'rejected', 'reason': str(exc)})
        return None
    frames = visit(0, [])
    return {'satisfied': frames is not None, 'frames': frames, 'attempts': attempts, 'events': events,
            'termination': 'complete' if frames is not None else 'finite_alternatives_or_budget_exhausted',
            'elapsed_seconds': time.monotonic()-began,
            'scope': 'Reference composition only; soft time budget checked between callbacks; no completeness, infeasibility or physical success claim'}
