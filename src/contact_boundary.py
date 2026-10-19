"""Numerical closure of a table-supported payload boundary, not a contact filter.

Only sub-nanometre support roundoff is admitted. Original endpoints are kept
in GCS; interior witnesses are used only to initialize official IRIS.
"""
import itertools
import numpy as np
from pydrake.geometry.optimization import HPolyhedron, VPolytope

SUPPORT_TOLERANCE_M = 1e-10
MAX_CLOSURE_SHIFT_RAD = 1e-9


def contact_domain(scene, checker, mapping, start, end, lower, upper, reach):
    """Full-dimensional pyramid with a supported endpoint as its apex.

    This is an IRIS bounding domain, not an executable line-segment fallback.
    The cap is clearance-scaled; IRIS and independent audits remain required.
    """
    start,end=np.asarray(start),np.asarray(end)
    scene.plant.SetPositions(scene.plant_context,mapping(end))
    context=scene.scene_graph.GetMyContextFromRoot(scene.root_context)
    query=scene.scene_graph.get_query_output_port().Eval(context)
    pairs=query.ComputeSignedDistancePairwiseClosestPoints(.1)
    clearance=min((p.distance for p in pairs),default=.1)
    if clearance <= 0 or not checker.CheckConfigCollisionFree(mapping(end)):
        raise RuntimeError('Contact-domain interior endpoint is not collision free')
    radius=min(float(clearance)/(4*reach*len(end)),.0002,
               float(np.min(end-np.asarray(lower)))*.5,
               float(np.min(np.asarray(upper)-end))*.5)
    if radius <= 1e-10:
        raise RuntimeError('No positive-volume cap for support-contact IRIS domain')
    vertices=np.column_stack([start,*[end+radius*np.array(signs) for signs in itertools.product((-1,1),repeat=len(end))]])
    domain=HPolyhedron(VPolytope(vertices))
    return domain,end,radius


def support_boundary(scene, positions):
    if scene.attached_cup_instance is None:
        return False
    scene.plant.SetPositions(scene.plant_context, positions)
    context=scene.scene_graph.GetMyContextFromRoot(scene.root_context)
    query=scene.scene_graph.get_query_output_port().Eval(context)
    cup=set(scene.plant.GetCollisionGeometriesForBody(scene.plant.GetBodyByName('attached_cup')))
    table=set(scene.plant.GetCollisionGeometriesForBody(scene.plant.GetBodyByName('table')))
    found=False
    for pair in query.ComputeSignedDistancePairwiseClosestPoints(.001):
        support=((pair.id_A in cup and pair.id_B in table) or
                 (pair.id_B in cup and pair.id_A in table))
        if support:
            if pair.distance < -SUPPORT_TOLERANCE_M:
                return False
            found |= abs(pair.distance) <= SUPPORT_TOLERANCE_M
        elif pair.distance < 0:
            return False
    return found


def interior_witness(scene, checker, mapping, point, seed):
    point=np.asarray(point)
    if not support_boundary(scene, mapping(point)):
        if not checker.CheckConfigCollisionFree(mapping(point)):
            raise RuntimeError('IRIS containment endpoint collides outside support roundoff tolerance')
        return point, False
    distance=np.linalg.norm(seed-point)
    if distance == 0:
        raise RuntimeError('Contact witness needs a distinct interior seed')
    for step in (1e-12, 1e-11, 1e-10, 5e-10):
        alpha=min(.5,step/distance)
        witness=point+alpha*(seed-point)
        if np.linalg.norm(witness-point)>MAX_CLOSURE_SHIFT_RAD:
            break
        if not np.array_equal(witness,point) and checker.CheckConfigCollisionFree(mapping(witness)):
            return witness, True
    raise RuntimeError('No interior IRIS witness within numerical contact closure tolerance')


def close_boundary(region, points):
    A,b=region.A(),region.b().copy()
    maximum=0.
    for point in points:
        violation=np.maximum(A@point-b,0.)
        norm=np.linalg.norm(A,axis=1)
        shifts=np.divide(violation,norm,out=np.zeros_like(violation),where=norm>0)
        if np.any((norm==0)&(violation>0)) or np.max(shifts,initial=0.)>MAX_CLOSURE_SHIFT_RAD:
            raise RuntimeError('IRIS boundary closure exceeds roundoff budget; no enlarged-region fallback')
        maximum=max(maximum,float(np.max(shifts,initial=0.)))
        b=np.maximum(b,A@point)
    return HPolyhedron(A,b), maximum
