"""Phase-local acceptance and exact composition for the static cup adapter.

Checks apply to emitted execution frames, including gripper closure and held
payload motion. Accepted snapshots cannot change when callers edit their input.
These finite checks do not certify continuous-time or physical success.
"""
from dataclasses import dataclass
import json
import numpy as np

from .semantic_plan import configuration
from .semantic_policy import accepts, policy_from_config

CUP_PHASES = ('navigate_to_pickup', 'approach_cup', 'close_gripper',
              'lift_cup', 'whole_body_carry_to_dropoff')


def require_endpoint(frame, expected, label):
    expected = np.asarray(expected, float)
    if expected.shape != (12,) or not np.all(np.isfinite(expected)):
        raise ValueError('Expected a finite complete endpoint')
    if np.max(np.abs(configuration(frame)-expected)) > 1e-6:
        raise ValueError(f'{label}: complete endpoint mismatch; explicit bridge required')


@dataclass(frozen=True)
class AcceptedMotion:
    phase: str
    frames_json: str
    report_json: str

    @property
    def frames(self):
        return json.loads(self.frames_json)

    @property
    def report(self):
        return json.loads(self.report_json)


def accept_cup_phase(scene, frames, dock, tucked, X_GC, *, start, end, max_samples=200000):
    from .green_room_gcs import audit_reference
    if not frames or frames[0]['phase'] not in CUP_PHASES:
        raise ValueError('Nonempty recognized cup phase required')
    require_endpoint(frames[0], start, 'Phase entry')
    require_endpoint(frames[-1], end, 'Phase exit')
    phase = frames[0]['phase']
    qs = np.array([configuration(f) for f in frames])
    opened, closed = scene.gripper_open_positions, scene.gripper_closed_positions
    if phase in ('navigate_to_pickup', 'approach_cup'):
        if np.max(np.abs(qs[:, 10:]-opened)) > 1e-8:
            raise ValueError('Free approach/navigation requires open gripper')
    elif phase == 'close_gripper':
        if (np.max(np.abs(qs[:, :10]-qs[0, :10])) > 1e-6
                or np.max(np.abs(qs[0, 10:]-opened)) > 1e-8
                or np.max(np.abs(qs[-1, 10:]-closed)) > 1e-8
                or np.any(qs[:, 10:] < np.minimum(opened, closed)-1e-8)
                or np.any(qs[:, 10:] > np.maximum(opened, closed)+1e-8)):
            raise ValueError('Closure requires fixed robot pose and open-to-closed fingers')
    else:
        if X_GC is None or np.max(np.abs(qs[:, 10:]-closed)) > 1e-8:
            raise ValueError('Lift/hold requires closed fingers and an attachment transform')
        if phase == 'whole_body_carry_to_dropoff' and np.max(np.abs(qs-qs[0])) > 1e-6:
            raise ValueError('Legacy hold phase must be stationary')
    report, _ = audit_reference(scene, frames, dock, tucked, X_GC,
                                phase_local=True, max_samples=max_samples)
    report['policy_satisfied'] = accepts(policy_from_config(scene.config),
        geometry=report['geometry_satisfied'], goals=report['goal_semantics_satisfied'],
        motion=report['motion_semantics_satisfied'], plan=report['supplied_plan_satisfied'])
    if not report['policy_satisfied']:
        raise RuntimeError(f"Phase {frames[0]['phase']} rejected: {json.dumps(report)}")
    from .build_scene import set_robot_configuration
    poses = []
    for q in (qs[0], qs[-1]):
        set_robot_configuration(scene, q[:3], q[3:10], q[10:])
        pose = (scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.gripper_body) @ X_GC
                if phase in ('lift_cup', 'whole_body_carry_to_dropoff') else scene.cup_initial_pose)
        poses.append(pose.GetAsMatrix4().tolist())
    report['cup_boundary_poses'] = poses
    return AcceptedMotion(frames[0]['phase'], json.dumps(frames), json.dumps(report))


def check_cup_boundary(left, right):
    a, b = left.frames[-1], right.frames[0]
    require_endpoint(b, configuration(a), right.phase)
    cup_error = float(np.max(np.abs(np.asarray(right.report['cup_boundary_poses'][0])
                                   - left.report['cup_boundary_poses'][1])))
    if cup_error > 1e-6:
        raise ValueError('Cup pose discontinuity at contact-mode boundary')
    return {'entering': right.phase, 'maximum_configuration_error':
            float(np.max(np.abs(configuration(b)-configuration(a)))),
            'maximum_cup_pose_matrix_error': cup_error}


def compose_cup_motion(segments, *, start, end):
    if (not all(isinstance(s, AcceptedMotion) for s in segments)
            or tuple(s.phase for s in segments) != CUP_PHASES):
        raise ValueError('Composition requires all accepted cup phases in task order')
    frames, boundaries = [], []
    for segment in segments:
        piece = segment.frames
        if not segment.report['policy_satisfied']:
            raise ValueError('Unaccepted segment')
        if frames:
            boundaries.append(check_cup_boundary(previous, segment))
        frames.extend(piece)
        previous = segment
    require_endpoint(frames[0], start, 'Task entry')
    require_endpoint(frames[-1], end, 'Task exit')
    return frames, {'accepted': True, 'phases': [s.phase for s in segments],
                    'boundaries': boundaries, 'acceptance': 'phase_local_then_exact_composition'}
