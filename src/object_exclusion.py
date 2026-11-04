"""Object-specific exclusion from axis-aligned 3D semantic boxes.

No obstacle is inserted into Drake: robot links may enter these boxes. For
each object collision shape, axis separation is a conservative disjointness
test. Its residual is 1-Lipschitz under fixed-orientation translation, so it
can bound the radius of the existing payload translation cover. Rotating
motions still require their own bounds or sampled acceptance.
"""
import numpy as np
from pydrake.math import RigidTransform
from .whole_robot_room import shape_support


def validate_exclusions(specs, object_ids):
    ids = set()
    if not isinstance(specs, list):
        raise ValueError('Object exclusions must be a list')
    for spec in specs:
        if set(spec) != {'id', 'object_id', 'lower_xyz', 'upper_xyz', 'margin_m'}:
            raise ValueError('Explicit id/object_id/3D bounds/margin required for exclusion')
        if not isinstance(spec['id'], str) or not spec['id'] or spec['id'] in ids:
            raise ValueError('Unique nonempty exclusion IDs required')
        ids.add(spec['id'])
        lo, hi = np.asarray(spec['lower_xyz'],float), np.asarray(spec['upper_xyz'],float)
        if (spec['object_id'] not in object_ids or lo.shape != (3,) or hi.shape != (3,)
                or not np.all(np.isfinite([lo,hi])) or np.any(lo >= hi)
                or not np.isfinite(spec['margin_m']) or spec['margin_m'] < 0):
            raise ValueError('Known object, ordered finite 3D bounds and nonnegative margin required')


def shape_box_residual(shape, pose, lower, upper, margin=0.):
    center, rotation = pose.translation(), pose.rotation().matrix()
    support = np.array([shape_support(shape, rotation.T @ n) for n in np.eye(3)])
    return float(max(np.max(center-support-upper), np.max(lower-center-support))-margin)


class ObjectExclusionEvaluator:
    def __init__(self, scene):
        # Inspector shapes are borrowed C++ objects. Keep their owning scene
        # alive even when callers construct this evaluator from a temporary.
        self.scene = scene
        self.specs = scene.config.get('object_exclusions', [])
        validate_exclusions(self.specs, scene.object_bodies)
        inspector = scene.scene_graph.model_inspector()
        self.geometry = {}
        for key in {s['object_id'] for s in self.specs}:
            self.geometry[key] = [(inspector.GetShape(g), inspector.GetPoseInFrame(g))
                for g in scene.plant.GetCollisionGeometriesForBody(scene.object_bodies[key])]
            if not self.geometry[key]:
                raise ValueError('Excluded object needs collision geometry')

    def evaluate(self, poses):
        rows=[]
        for spec in self.specs:
            key=spec['object_id']
            if key not in poses:
                raise ValueError(f'Missing world pose for excluded object {key}')
            matrix = poses[key]
            if isinstance(matrix, RigidTransform):
                pose = matrix
            else:
                matrix = np.asarray(matrix,float)
                if matrix.shape != (4,4) or not np.all(np.isfinite(matrix)):
                    raise ValueError('Finite object world transform required')
                pose = RigidTransform(matrix)
            residual=min(shape_box_residual(shape, pose @ X_BG, np.asarray(spec['lower_xyz']),
                np.asarray(spec['upper_xyz']), spec['margin_m']) for shape,X_BG in self.geometry[key])
            rows.append({'id':spec['id'], 'object_id':key, 'residual_m':residual, 'satisfied':residual >= -1e-8})
        return {'satisfied':all(r['satisfied'] for r in rows), 'constraints':rows,
            'minimum_residual_m':min((r['residual_m'] for r in rows),default=None),
            'scope':'Object collision volumes only; conservative axis separation from 3D boxes; robot unrestricted by these boxes'}

    def current(self, scene):
        return self.evaluate({key:scene.plant.EvalBodyPoseInWorld(scene.plant_context,scene.object_bodies[key])
                              for key in self.geometry})
