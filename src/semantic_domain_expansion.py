"""Expand 10D IRIS input domains using the measured semantic slack.

This only certifies the supplied Lipschitz occupation bound; collision
geometry is handled by official IRIS and independent path auditing.
"""

import numpy as np
from .joint_approach_gcs import motion_bound


def expand_domain(a, b, lower, upper, residual, reach, *, fraction=0.7, cap_scale=1.0):
    a, b, lower, upper = map(lambda x: np.asarray(x, float), (a, b, lower, upper))
    if any(
        x.shape != (10,) or not np.all(np.isfinite(x)) for x in (a, b, lower, upper)
    ) or np.any(lower >= upper):
        raise ValueError("Finite ten-dimensional domain limits required")
    if not np.isfinite(residual) or not 0 < fraction < 1 or not 0 < cap_scale <= 1:
        raise ValueError("Finite semantic residual and strict slack fraction required")
    center = (a + b) / 2
    half = np.abs(b - a) / 2
    if np.any(center - half < lower - 1e-10) or np.any(center + half > upper + 1e-10):
        raise ValueError("Guide endpoints outside configuration limits")
    caps = np.minimum(
        np.r_[[0.10, 0.10, 0.04], [0.04] * 7],
        np.minimum(center - half - lower, upper - center - half),
    )
    caps = np.maximum(0, caps) * cap_scale
    initial = motion_bound(half, reach)
    if residual - initial <= 1e-6:
        return None
    target = initial + fraction * (residual - initial - 1e-6)
    low, high = 0.0, 1.0
    for _ in range(45):
        mid = (low + high) / 2
        if motion_bound(half + mid * caps, reach) <= target:
            low = mid
        else:
            high = mid
    radius = half + low * caps
    if np.min(radius) < 1e-9:
        return None
    bound = float(residual - motion_bound(radius, reach))
    return (
        center - radius,
        center + radius,
        dict(
            center_residual_m=float(residual),
            certified_lower_bound_m=bound,
            expansion_scale=low,
            half_width=radius.tolist(),
            guide_half_width=half.tolist(),
            scope="Conservative occupation-motion bound for the current spatial-rule oracle; not an IRIS collision certificate",
        ),
    )
