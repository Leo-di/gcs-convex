"""Query-local, attached-object-aware 10D endpoint growth for semantic GCS.

The caller owns the mode-specific cache. No trajectory or bridge is fabricated
if the grown regions cannot connect it. Region audits are numerical, not proofs.
"""
import copy
import time
import numpy as np
from pydrake.common import RandomGenerator
from pydrake.geometry.optimization import HPolyhedron
from .build_scene import build_scene, set_robot_configuration
from .carry_mode_gcs import CarryOracle
from .grasp_audit import measure_cup_gripper, summarize_grasp_audit
from .collision_checker import evaluate_collision_status
from .iris_region_generator import (
    _base_region_specs, _base_in_region, _cover_domain, _grow_cover_region,
    _make_collision_checker, _make_parameterization, _reduced_position_indices,
    result_to_dict,
)
from .online_iris_attachment import connection_status


def sets(records):
    return [HPolyhedron(np.array(r['A']), np.array(r['b'])) for r in records]


def grow_carry_region(config, anchor, seed, label, index):
    oracle = CarryOracle(config, anchor)
    scene = build_scene(config=config, attached_cup_X_GC=oracle.X_GC, visualize=False)
    if scene.attached_cup_instance is None:
        raise ValueError('Carrying IRIS requires attached-object collision geometry')
    set_robot_configuration(scene, seed[:3], seed[3:], anchor['gripper_q'])
    parameterization = _make_parameterization(
        scene.plant.GetPositions(scene.plant_context).copy(), _reduced_position_indices(scene))
    spec = copy.deepcopy(config['iris_np2_cover'])
    spec.update(particles=2000, ray_particles=1000, max_iterations=6,
                max_separating_iterations=30, max_planes_per_iteration=10,
                epsilon=.0025, mixing_steps=50, holdout_samples_per_region=1000,
                holdout_mixing_steps=40)
    domain = _cover_domain(scene, seed, _base_region_specs(config)[label], spec)
    if not domain.PointInSet(seed):
        raise ValueError('Endpoint outside its semantic domain')
    result = _grow_cover_region(scene=scene, checker=_make_collision_checker(scene),
        parameterization=parameterization, seed=seed, domain=domain, spec=spec,
        delta=.005, index=index, add_hyperplane_if_solve_fails=False)
    record = result_to_dict(result)
    record.update(semantic_region=label, online=True,
                  add_hyperplane_if_nonlinear_solve_fails=False,
                  algorithm='Drake IrisNp2 online carrying endpoint')
    # Audit against the original free-cup model as well. The planning checker
    # filters cup/gripper rigid-body pairs; this restores unfiltered grasp tests.
    bad = []
    rng = RandomGenerator(917000 + index)
    point = result.region.ChebyshevCenter()
    for i in range(1001):
        point = seed.copy() if i == 0 else result.region.UniformSample(rng, point, mixing_steps=40)
        oracle.state(point[:2], point[2:])
        status = evaluate_collision_status(oracle.scene, point[:3], arm_q=point[3:],
                                           gripper_q=anchor['gripper_q'])
        grasp = summarize_grasp_audit([{'attached': True,
                                       'cup_gripper_audit': measure_cup_gripper(oracle.scene)}])
        semantic = _base_in_region(point, _base_region_specs(config)[label])
        if not status.collision_free or not grasp['satisfied'] or not semantic:
            bad.append({'q': point.tolist(), 'collision_residual': status.residual,
                        'grasp': grasp['first_violation'], 'semantic': semantic})
    record['original_scene_audit'] = {'samples': 1001, 'violations': len(bad),
                                      'first_violation': bad[0] if bad else None}
    record['accepted'] = result.holdout_collisions == 0 and not bad
    record['options'] = spec
    return record


def attach_carry_endpoints(config, records, anchor, start, goal, *, seed_base=81000,
                           grow_fn=grow_carry_region):
    """At most one IRIS call per uncovered endpoint, with no mutation of cache."""
    start, goal = np.asarray(start), np.asarray(goal)
    if start.shape != (10,) or goal.shape != (10,):
        raise ValueError('Carrying endpoints must have dimension 10')
    if not anchor.get('attached'):
        raise ValueError('Carrying endpoint requires an attached grasp anchor')
    selected = copy.deepcopy(records)
    if any(not r.get('attached_object_included') or
           r.get('independent_holdout_audit', {}).get('collisions', 0) or
           r.get('accepted') is False for r in selected):
        raise ValueError('Cache contains wrong-mode or rejected regions')
    before = connection_status(sets(selected), start, goal)
    attempts, decisions = [], []
    began = time.perf_counter()
    specs = _base_region_specs(config)
    for i, (name, point) in enumerate([('start', start), ('goal', goal)]):
        if any(r.PointInSet(point) for r in sets(selected)):
            decisions.append({'endpoint': name, 'action': 'already_covered'})
            continue
        labels = [label for label, spec in specs.items() if _base_in_region(point, spec)]
        if not labels:
            decisions.append({'endpoint': name, 'action': 'outside_semantic_domains'})
            continue
        try:
            record = grow_fn(config, anchor, point, labels[0], seed_base+i)
        except RuntimeError as error:
            record = {'accepted': False, 'error': str(error)}
        record = dict(record, endpoint=name)
        attempts.append(record)
        decisions.append({'endpoint': name, 'action': 'iris', 'accepted': record['accepted']})
        if record['accepted']:
            selected.append(record)
    return selected, {'before': before, 'after': connection_status(sets(selected), start, goal),
                      'decisions': decisions, 'attempts': attempts,
                      'wall_seconds': time.perf_counter()-began,
                      'scope': '10D carrying only; fixed grasp; one call per uncovered endpoint; no bridge search or region safety certificate'}
