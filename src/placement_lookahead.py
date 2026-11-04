"""Order placements using downstream object geometry, without removing branches.

Predicted release poses are estimates. A blocked estimate changes search order
only; all original candidates and the full motion checks remain available.
"""
import copy
import numpy as np
from pydrake.math import RigidTransform


def rank_placements(config, world, spec, depth):
    from .build_scene import build_scene
    from .stateful_transport_grounding import world_config

    action = spec['actions'][depth]
    future = spec['actions'][depth + 1:]
    if not future:
        return list(range(len(action['placements']))), []
    # Every object is transported once in this adapter. Earlier placements
    # remain fixed, whereas unprocessed source objects must not be treated as
    # permanent blockers of a later destination.
    fixed_ids = [a['object_id'] for a in spec['actions'][:depth + 1]]
    scene = build_scene(config=world_config(config, world, action['object_id']), visualize=False)
    plant = scene.plant
    geometry = {key: plant.GetCollisionGeometriesForBody(body)
                for key, body in scene.object_bodies.items()}
    if any(not ids for ids in geometry.values()):
        raise ValueError('Lookahead requires object collision geometry')
    context = scene.scene_graph.GetMyContextFromRoot(scene.root_context)
    rows = []
    for index, placement in enumerate(action['placements']):
        pose = np.array(world['poses'][action['object_id']], copy=True)
        pose[:3, 3] = placement
        plant.SetFreeBodyPose(scene.plant_context, scene.object_bodies[action['object_id']], RigidTransform(pose))
        future_rows = []
        for later in future:
            key = later['object_id']
            options = []
            for target in later['placements']:
                predicted = np.array(world['poses'][key], copy=True)
                predicted[:3, 3] = target
                plant.SetFreeBodyPose(scene.plant_context, scene.object_bodies[key], RigidTransform(predicted))
                query = scene.scene_graph.get_query_output_port().Eval(context)
                distances = [dict(object_id=other, distance_m=float(
                    query.ComputeSignedDistancePairClosestPoints(gid, oid).distance))
                    for other in fixed_ids for gid in geometry[key] for oid in geometry[other]]
                options.append(dict(placement=copy.deepcopy(target), blocked=any(
                    p['distance_m'] < .005 for p in distances), pairs=distances))
            future_rows.append(dict(action_id=later['id'], options=options,
                                    all_blocked=all(o['blocked'] for o in options)))
        rows.append(dict(placement_index=index, future=future_rows,
                         exhausted_actions=sum(r['all_blocked'] for r in future_rows),
                         blocked_options=sum(o['blocked'] for r in future_rows for o in r['options'])))
    order = sorted(range(len(rows)), key=lambda i: (rows[i]['exhausted_actions'], rows[i]['blocked_options'], i))
    return order, rows
