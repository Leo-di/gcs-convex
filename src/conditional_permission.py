"""State-dependent access to 3D regions, checked over robot collision volumes.

The permission is a current holding condition, not a permanent 'picked once'
flag. Planning uses an audited rigid grasp; execution uses measured bilateral
contact, absence of support contact, and bounded relative translation drift.
"""
import numpy as np
from .object_exclusion import shape_box_residual


def validate_permissions(specs, object_ids):
    if not isinstance(specs, list):
        raise ValueError('Conditional permissions must be a list')
    ids = set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {
                'id', 'requires_holding', 'lower_xyz', 'upper_xyz', 'margin_m'}:
            raise ValueError('Permission needs id, requires_holding, 3D bounds and margin')
        lo, hi = np.asarray(spec['lower_xyz'], float), np.asarray(spec['upper_xyz'], float)
        if (not isinstance(spec['id'], str) or not spec['id'] or spec['id'] in ids
                or spec['requires_holding'] not in object_ids
                or lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi]))
                or np.any(lo >= hi) or not np.isfinite(spec['margin_m']) or spec['margin_m'] < 0):
            raise ValueError('Unique permission, known held object and ordered finite bounds required')
        ids.add(spec['id'])


class ConditionalPermissionEvaluator:
    def __init__(self, scene):
        self.scene = scene
        self.specs = scene.config.get('conditional_permissions', [])
        validate_permissions(self.specs, scene.object_bodies)
        self.geometry = []
        if not self.specs:
            return
        inspector = scene.scene_graph.model_inspector()
        for instance in (scene.base_instance, scene.arm_instance, scene.gripper_instance):
            for index in scene.plant.GetBodyIndices(instance):
                body = scene.plant.get_body(index)
                for gid in scene.plant.GetCollisionGeometriesForBody(body):
                    self.geometry.append((body, inspector.GetShape(gid), inspector.GetPoseInFrame(gid)))
        if not self.geometry:
            raise ValueError('Permission requires robot collision geometry')

    def current(self, held_objects=()):
        s = self.scene
        poses = [(body.name(), shape, s.plant.EvalBodyPoseInWorld(s.plant_context, body) @ X)
                 for body, shape, X in self.geometry]
        rows = []
        for spec in self.specs:
            residual, witness = min((shape_box_residual(shape, pose, np.asarray(spec['lower_xyz']),
                np.asarray(spec['upper_xyz']), spec['margin_m']), name) for name, shape, pose in poses)
            authorized = spec['requires_holding'] in held_objects
            rows.append(dict(id=spec['id'], requires_holding=spec['requires_holding'],
                authorized=authorized, robot_outside_residual_m=residual, closest_body=witness,
                region_active=residual < 0., satisfied=authorized or residual >= -1e-8))
        return dict(satisfied=all(r['satisfied'] for r in rows), constraints=rows,
            minimum_locked_residual_m=min((r['robot_outside_residual_m'] for r in rows
                if not r['authorized']), default=None),
            scope='All robot collision volumes; conservative axis separation; permission requires current holding')


class MeasuredHolding:
    """No permission from phase names, gripper commands, or desired poses."""
    def __init__(self, object_ids, *, minimum_duration_s=.10, maximum_drift_m=.01):
        self.object_ids = set(object_ids)
        self.minimum_duration_s = minimum_duration_s
        self.maximum_drift_m = maximum_drift_m
        self.anchors = {}
        self.last_time = None

    def observe(self, row):
        now = float(row['time_s'])
        if not np.isfinite(now) or (self.last_time is not None and now < self.last_time):
            raise ValueError('Finite monotone observation time required')
        self.last_time = now
        held, evidence = set(), {}
        for key in self.object_ids:
            obj = row['objects'].get(key, {})
            contact = obj.get('contacts_N', {})
            relative = np.asarray(obj.get('relative_translation_m', []), float)
            required = ('left_finger', 'right_finger', 'table', 'dropoff_table', 'dropoff_bin')
            valid = (all(k in contact and np.isfinite(contact[k]) for k in required)
                and relative.shape == (3,) and np.all(np.isfinite(relative))
                and min(contact['left_finger'], contact['right_finger']) > .05
                and max(contact[k] for k in required[2:]) < .05)
            drift, duration = None, 0.
            if not valid:
                self.anchors.pop(key, None)
            else:
                self.anchors.setdefault(key, (now, relative.copy()))
                start, anchor = self.anchors[key]
                drift = float(np.linalg.norm(relative-anchor))
                duration = now-start
                if drift <= self.maximum_drift_m and duration >= self.minimum_duration_s-1e-9:
                    held.add(key)
            evidence[key] = dict(held=key in held, bilateral_and_unsupported=bool(valid),
                stable_duration_s=duration, relative_drift_m=drift)
        return held, evidence


class PermissionExecutionMonitor:
    def __init__(self, scene):
        self.evaluator = ConditionalPermissionEvaluator(scene)
        self.holding = MeasuredHolding({s['requires_holding'] for s in self.evaluator.specs})
        self.samples = 0
        self.authorized_region_samples = 0
        self.events = []
        self.previous = None

    def observe(self, row):
        held, evidence = self.holding.observe(row)
        report = self.evaluator.current(held)
        self.samples += 1
        self.authorized_region_samples += int(any(c['region_active'] and c['authorized']
                                                  for c in report['constraints']))
        state = [(c['id'], c['authorized'], c['region_active']) for c in report['constraints']]
        if state != self.previous or not report['satisfied']:
            self.events.append(dict(time_s=row['time_s'], phase=row['phase'],
                holding_evidence=evidence, **report))
            self.previous = state
        return dict(holding_evidence=evidence, **report)

    def report(self):
        return dict(enabled=bool(self.evaluator.specs), samples=self.samples,
            authorized_region_samples=self.authorized_region_samples, events=self.events,
            holding_rule='Bilateral forces >0.05 N, all table/bin forces <0.05 N, relative drift <=10 mm, sustained >=0.1 s',
            scope='Measured 10 ms observations; revocable permission, no continuous-time safety claim')
