"""Scripted moving occupancy regions and bounded wait scheduling.

The obstacle is a moving vertical XY prism, checked conservatively against
every robot collision shape. This is geometric occupancy, not crowd dynamics.
"""
import copy
import numpy as np
from .whole_robot_room import WholeRobotRoomOracle, shape_support
from .build_scene import set_robot_configuration


class MovingKeepClear:
    def __init__(self, scene, spec):
        self.scene = scene
        self.knots = np.asarray(spec['keyframes'], float)
        self.size = np.asarray(spec['size_xy_m'], float)
        self.margin = float(spec.get('margin_m', 0.))
        if (self.knots.ndim != 2 or self.knots.shape[1] != 3 or len(self.knots) < 2
                or not np.all(np.isfinite(self.knots)) or np.any(np.diff(self.knots[:, 0]) <= 0)
                or self.size.shape != (2,) or not np.all(np.isfinite(self.size)) or np.any(self.size <= 0)
                or not np.isfinite(self.margin) or self.margin < 0):
            raise ValueError('Moving occupancy needs increasing [time,x,y] knots, positive XY size and margin')
        self.oracle = WholeRobotRoomOracle.from_xy_bounds(scene, [-1., 1.], [-1., 1.])

    def evaluate(self, q, time_s):
        if time_s is None or not np.isfinite(time_s) or not self.knots[0, 0] <= time_s <= self.knots[-1, 0]:
            raise ValueError('Moving occupancy time outside prediction horizon')
        center = np.array([np.interp(time_s, self.knots[:, 0], self.knots[:, i]) for i in (1, 2)])
        lo, hi = center-self.size/2, center+self.size/2
        offsets = [-lo[0], hi[0], -lo[1], hi[1]]
        set_robot_configuration(self.scene, q[:3], q[3:10], q[10:])
        residuals = []
        for body, shape, X_BG, _ in self.oracle.geometry:
            pose = self.scene.plant.EvalBodyPoseInWorld(self.scene.plant_context, body) @ X_BG
            residuals.append(max(float(n@pose.translation()-shape_support(shape, pose.rotation().matrix().T@n)
                - b-self.margin) for n, b in zip(self.oracle.normals, offsets)))
        residual = min(residuals)
        return {'satisfied': residual >= -1e-8, 'residual': residual,
            'method': 'time_dependent_conservative_axis_separation',
            'obstacle_center_xy_m': center.tolist(), 'time_s': float(time_s)}


def schedule_with_waits(frames, plan, scene, *, candidate_delays_s, frame_period_s=.1,
                        initial_hold_s=1., tail_hold_s=2.):
    """Search bounded start delays. Waits themselves must satisfy invariants."""
    from .semantic_plan import SemanticPlanEvaluator
    if (not frames or not np.isfinite(frame_period_s) or frame_period_s <= 0
            or initial_hold_s < 0 or tail_hold_s < 0):
        raise ValueError('Nonempty reference and positive finite timing required')
    evaluator = SemanticPlanEvaluator(scene, plan)
    attempts = []
    for delay in candidate_delays_s:
        if not np.isfinite(delay) or delay < 0 or abs(delay/frame_period_s-round(delay/frame_period_s)) > 1e-7:
            raise ValueError('Wait delays must be finite nonnegative frame-period multiples')
        prefix = [copy.deepcopy(frames[0]) for _ in range(round(delay/frame_period_s))]
        reference = prefix+copy.deepcopy(frames)
        # Match the force executor: initial warmup, reference, final settle.
        timed = [dict(reference[0], time_s=0.)]
        timed += [dict(f, time_s=initial_hold_s+i*frame_period_s) for i, f in enumerate(reference)]
        timed += [dict(reference[-1], time_s=initial_hold_s+(len(reference)-1)*frame_period_s+tail_hold_s)]
        try:
            audit = evaluator.audit(timed)
            attempts.append({'delay_s': delay, 'satisfied': audit['satisfied'], 'audit': audit})
            if audit['satisfied']:
                return {'satisfied': True, 'delay_s': delay, 'frames': reference, 'timed_frames': timed, 'attempts': attempts}
        except (ValueError, RuntimeError) as exc:
            attempts.append({'delay_s': delay, 'satisfied': False, 'reason': str(exc)})
    return {'satisfied': False, 'frames': None, 'attempts': attempts,
            'termination': 'finite_start_delays_exhausted_not_infeasibility'}
