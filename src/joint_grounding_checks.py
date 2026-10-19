"""Cheap endpoint filter, not a certificate of an executable manipulation path."""
import numpy as np


def check_pick_endpoints(scene,base,oracle,*,enforce_semantic=True):
    from .grasp_planner import plan_pick
    pick=plan_pick(scene,base)
    goals={name:oracle.query(np.r_[base,q]) for name,q in [('pregrasp',pick.pregrasp_q),('grasp',pick.grasp_q),('lift',pick.lift_q)]}
    if enforce_semantic and not all(g['satisfied'] for g in goals.values()):
        raise RuntimeError('Manipulation endpoint violates whole-robot room semantics')
    return pick,goals
