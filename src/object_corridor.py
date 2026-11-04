"""Phase-scoped full-object containment in overlapping 3D boxes.

Each object must fit entirely in at least one cell, including its margin.
This sufficient condition is conservative near cell seams: containment in
the union alone need not imply containment in one cell. The max-of-minimum
face residual is 1-Lipschitz for fixed-orientation translations, permitting
its use in the payload translation cover. No robot geometry is constrained.
"""
import numpy as np
from pydrake.math import RigidTransform
from .whole_robot_room import shape_support


def validate_corridors(specs, object_ids, actions=None):
    if not isinstance(specs, list):
        raise ValueError('Object corridors must be a list')
    ids = set()
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {'id', 'object_id', 'action_id', 'mode', 'cells', 'margin_m'}:
            raise ValueError('Explicit corridor id/object/action/mode/cells/margin required')
        if not isinstance(spec['id'], str) or not spec['id'] or spec['id'] in ids:
            raise ValueError('Unique nonempty corridor IDs required')
        ids.add(spec['id'])
        if (spec['object_id'] not in object_ids or not isinstance(spec['action_id'], str)
                or not spec['action_id'] or any(c in spec['action_id'] for c in './\\')
                or spec['mode'] != 'carry' or not np.isfinite(spec['margin_m']) or spec['margin_m'] < 0):
            raise ValueError('Known object, safe action ID, carry mode and finite nonnegative margin required')
        if actions is not None and actions.get(spec['action_id']) != spec['object_id']:
            raise ValueError('Corridor must name the object carried by its action')
        if not isinstance(spec['cells'], list) or not spec['cells']:
            raise ValueError('Nonempty corridor cells required')
        for cell in spec['cells']:
            if not isinstance(cell, dict) or set(cell) != {'lower_xyz', 'upper_xyz'}:
                raise ValueError('Corridor cells require 3D lower/upper bounds')
            lo, hi = np.asarray(cell['lower_xyz'], float), np.asarray(cell['upper_xyz'], float)
            if (lo.shape != (3,) or hi.shape != (3,) or not np.all(np.isfinite([lo, hi]))
                    or np.any(hi-lo <= 2*spec['margin_m'])):
                raise ValueError('Finite ordered cells wider than twice the margin required')


def shape_containment_residual(shape, pose, lower, upper, margin=0.):
    center, rotation = pose.translation(), pose.rotation().matrix()
    # Evaluate both signs, rather than assuming a symmetric collision shape.
    values = []
    for axis in np.eye(3):
        j = int(np.argmax(axis))
        values += [upper[j]-center[j]-shape_support(shape, rotation.T@axis),
                   center[j]-lower[j]-shape_support(shape, rotation.T@(-axis))]
    return float(min(values)-margin)


class ObjectCorridorEvaluator:
    def __init__(self, scene):
        self.scene = scene  # Own the inspector's borrowed collision shapes.
        self.specs = scene.config.get('object_corridors', [])
        validate_corridors(self.specs, scene.object_bodies)
        inspector = scene.scene_graph.model_inspector()
        self.geometry = {}
        for key in {s['object_id'] for s in self.specs}:
            self.geometry[key] = [(inspector.GetShape(g), inspector.GetPoseInFrame(g))
                for g in scene.plant.GetCollisionGeometriesForBody(scene.object_bodies[key])]
            if not self.geometry[key]:
                raise ValueError('Corridor object needs collision geometry')

    def evaluate(self, poses, phase):
        if not isinstance(phase, str) or not phase:
            raise ValueError('Explicit phase required for corridor evaluation')
        rows = []
        for spec in self.specs:
            if phase != spec['action_id']+'.'+spec['mode']:
                continue
            key = spec['object_id']
            if key not in poses:
                raise ValueError(f'Missing world pose for corridor object {key}')
            pose = poses[key]
            if not isinstance(pose, RigidTransform):
                matrix = np.asarray(pose, float)
                if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
                    raise ValueError('Finite object world transform required')
                pose = RigidTransform(matrix)
            # All shapes must share a containing cell. This also bounds a
            # disconnected multi-shape object conservatively.
            values = [min(shape_containment_residual(shape, pose@X_BG,
                cell['lower_xyz'], cell['upper_xyz'], spec['margin_m'])
                for shape, X_BG in self.geometry[key]) for cell in spec['cells']]
            residual = max(values)
            rows.append({'id': spec['id'], 'object_id': key, 'residual_m': residual,
                'satisfied': residual >= -1e-8, 'best_cell': int(np.argmax(values))})
        return {'satisfied': all(r['satisfied'] for r in rows), 'constraints': rows,
            'minimum_residual_m': min((r['residual_m'] for r in rows), default=None),
            'scope': 'Whole object in one corridor cell at each pose; carry phase only; robot unrestricted'}

    def current(self, scene, phase):
        return self.evaluate({key: scene.plant.EvalBodyPoseInWorld(scene.plant_context, scene.object_bodies[key])
                              for key in self.geometry}, phase)
