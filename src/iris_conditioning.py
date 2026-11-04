"""Similarity-only conditioning for numerically failed capped IRIS domains.

q = origin + scale * rotation @ z preserves the Euclidean joint-space metric
up to one scalar. Domain, containment points, initial ellipsoid and requested
joint-space margin are all transformed together. This is a numerical retry of
official IrisNp2 on the same physical problem, never a region/box fallback.
"""
import numpy as np
from pydrake.geometry.optimization import HPolyhedron,Hyperellipsoid
from pydrake.planning import IrisNp2,IrisParameterizationFunction


def solve_capped_iris(checker,ellipse,domain,options,mapping,cap_radius=None,*,solve=None):
    solve=IrisNp2 if solve is None else solve
    try:return solve(checker,ellipse,domain,options),None
    except RuntimeError as error:
        if cap_radius is None or 'maximum inscribed ellipse problem' not in str(error):raise
        original_error=str(error)
    origin=np.asarray(ellipse.center());points=np.asarray(options.sampled_iris_options.containment_points).copy()
    delta=points-origin[:,None];lengths=np.linalg.norm(delta,axis=0);direction=delta[:,int(np.argmax(lengths))]
    length=float(np.max(lengths));scale=float(np.sqrt(float(cap_radius)*max(length,float(cap_radius))))
    if not np.isfinite(scale) or scale<=0:raise RuntimeError('Invalid contact-domain similarity scale')
    rotation=np.linalg.qr(np.column_stack([direction,np.eye(len(origin))]))[0]
    transform=scale*rotation
    local=HPolyhedron(domain.A()@transform,domain.b()-domain.A()@origin)
    seed=Hyperellipsoid(ellipse.A()@transform,np.zeros(len(origin)))
    sampled=options.sampled_iris_options;old_margin=float(sampled.configuration_space_margin);old_parameterization=options.parameterization
    try:
        options.parameterization=IrisParameterizationFunction(lambda z:mapping(origin+transform@z),len(origin))
        sampled.containment_points=rotation.T@(points-origin[:,None])/scale
        sampled.configuration_space_margin=old_margin/scale
        result=solve(checker,seed,local,options)
    finally:
        options.parameterization=old_parameterization
        sampled.containment_points=points
        sampled.configuration_space_margin=old_margin
    A=result.A()@rotation.T/scale;b=result.b()+A@origin
    row_norm=np.linalg.norm(A,axis=1);row_norm=np.where(row_norm>0,row_norm,1.)
    region=HPolyhedron(A/row_norm[:,None],b/row_norm)
    return region,dict(method='similarity_conditioned_official_IrisNp2',original_error=original_error,
        origin=origin.tolist(),rotation=rotation.tolist(),scale=scale,
        joint_margin_rad=old_margin,normalized_margin=old_margin/scale,
        scope='Same capped physical domain, collision model, guide endpoints and Euclidean joint margin; one numerical retry; no enlarged domain or collision-filter fallback')
