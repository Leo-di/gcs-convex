"""Phase-scoped whole-object regions expressed in another object's frame.

The sequential transport adapter freezes the reference during each carry,
but recomputes its pose from the current world after earlier actions. The
execution evaluator always uses both measured poses, including rotations.
This does not plan synchronized motion of two independently moving objects.
"""
import numpy as np
from pydrake.math import RigidTransform
from .object_corridor import shape_containment_residual, validate_corridors


def validate_relative(specs, object_ids, actions=None):
    if not isinstance(specs, list):
        raise ValueError('Object-relative constraints must be a list')
    corridors = []
    for spec in specs:
        if not isinstance(spec, dict) or set(spec) != {
                'id', 'object_id', 'reference_object_id', 'action_id', 'mode',
                'cells', 'margin_m', 'reference_policy'}:
            raise ValueError('Explicit relative region, reference and policy required')
        if (spec['reference_object_id'] not in object_ids
                or spec['reference_object_id'] == spec['object_id']
                or spec['reference_policy'] != 'stationary_during_action'):
            raise ValueError('Distinct known reference with stationary_during_action policy required')
        corridors.append({k: v for k, v in spec.items()
                          if k not in ('reference_object_id', 'reference_policy')})
    validate_corridors(corridors, object_ids, actions)


def world_pose(value):
    if isinstance(value, RigidTransform):
        return value
    matrix = np.asarray(value, float)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('Finite rigid object transform required')
    return RigidTransform(matrix)


class ObjectRelativeEvaluator:
    def __init__(self, scene):
        self.scene = scene
        self.specs = scene.config.get('object_relative', [])
        validate_relative(self.specs, scene.object_bodies)
        inspector = scene.scene_graph.model_inspector()
        self.geometry = {}
        self.object_ids = set()
        for spec in self.specs:
            key = spec['object_id']
            self.object_ids.update((key, spec['reference_object_id']))
            self.geometry[key] = [(inspector.GetShape(g), inspector.GetPoseInFrame(g))
                for g in scene.plant.GetCollisionGeometriesForBody(scene.object_bodies[key])]
            if not self.geometry[key]:
                raise ValueError('Relative constrained object requires collision geometry')

    def evaluate(self, poses, phase):
        if not isinstance(phase, str) or not phase:
            raise ValueError('Explicit phase required for relative evaluation')
        rows = []
        for spec in self.specs:
            if phase != spec['action_id']+'.'+spec['mode']:
                continue
            key, ref = spec['object_id'], spec['reference_object_id']
            if key not in poses or ref not in poses:
                raise ValueError('Both constrained and reference object poses required')
            X_WR, X_WO = world_pose(poses[ref]), world_pose(poses[key])
            X_RO = X_WR.inverse() @ X_WO
            values = [min(shape_containment_residual(shape, X_RO @ X_BG,
                cell['lower_xyz'], cell['upper_xyz'], spec['margin_m'])
                for shape, X_BG in self.geometry[key]) for cell in spec['cells']]
            residual = max(values)
            rows.append(dict(id=spec['id'], object_id=key, reference_object_id=ref,
                residual_m=residual, satisfied=residual >= -1e-8,
                best_cell=int(np.argmax(values)),
                reference_pose=X_WR.GetAsMatrix4().tolist(),
                relative_pose=X_RO.GetAsMatrix4().tolist()))
        return dict(satisfied=all(r['satisfied'] for r in rows), constraints=rows,
            minimum_residual_m=min((r['residual_m'] for r in rows), default=None),
            scope='Whole object in a cell of the current reference frame; carry only; '
                  'stationary reference per planned action; measured poses at execution')

    def current(self, scene, phase):
        return self.evaluate({key: scene.plant.EvalBodyPoseInWorld(
            scene.plant_context, scene.object_bodies[key]) for key in self.object_ids}, phase)


def initialize_relative_visuals(scene):
    """Visual-only boxes, with transforms recorded from measured references."""
    from pydrake.geometry import Box, Rgba
    for i, spec in enumerate(scene.config.get('object_relative_display', scene.config.get('object_relative', []))):
        for j, cell in enumerate(spec['cells']):
            lo, hi = np.asarray(cell['lower_xyz']), np.asarray(cell['upper_xyz'])
            path = f'/semantic/object_relative_{i}/cell_{j}'
            scene.meshcat.SetObject(path, Box(*(hi-lo)), Rgba(.45, .2, .9, .16))
    update_relative_visuals(scene)


def update_relative_visuals(scene, time_s=None):
    if scene.meshcat is None:
        return
    for i, spec in enumerate(scene.config.get('object_relative_display', scene.config.get('object_relative', []))):
        ref = scene.plant.EvalBodyPoseInWorld(scene.plant_context,
                                             scene.object_bodies[spec['reference_object_id']])
        for j, cell in enumerate(spec['cells']):
            center = (np.asarray(cell['lower_xyz'])+np.asarray(cell['upper_xyz']))/2
            path = f'/semantic/object_relative_{i}/cell_{j}'
            kwargs = {} if time_s is None else {'time_in_recording': time_s}
            scene.meshcat.SetTransform(path, ref @ RigidTransform(center), **kwargs)
