"""Bounded path simplification; every shortcut needs an explicit edge oracle."""
import numpy as np


def shortcut_path(path, edge_valid, *, max_checks=2048):
    """Try whole spans, subdividing rejected spans at their middle vertex.

    Only subsequences of original vertices are returned, including both exact
    endpoints. A finite budget affects quality, never permits an unchecked edge.
    Original edges must already have been validated by the caller.
    """
    points = np.asarray(path, dtype=float)
    if (points.ndim != 2 or len(points) < 2 or not np.all(np.isfinite(points))
            or type(max_checks) is not int or max_checks < 0):
        raise ValueError('Finite polyline and nonnegative integer check budget required')
    # At most 2*N-3 oracle calls, unlike a farthest-visible scan that can spend
    # its entire budget near the first vertex of a finely subdivided cover.
    indices, checks, rejected = [0], 0, 0
    pending = [(0, len(points)-1)]
    while pending:
        i, j = pending.pop()
        if j == i+1 or checks >= max_checks:
            indices.extend(range(i+1, j+1))
            continue
        checks += 1
        if edge_valid(points[i], points[j]):
            indices.append(j)
            continue
        rejected += 1
        middle = (i+j)//2
        pending.extend([(middle, j), (i, middle)])
    result = points[indices].copy()
    return result, {'input_vertices': len(points), 'output_vertices': len(result),
                    'edge_checks': checks, 'rejected_shortcuts': rejected,
                    'max_checks': max_checks,
                    'input_length_rad': float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                    'output_length_rad': float(np.linalg.norm(np.diff(result, axis=0), axis=1).sum())}


def cover_edge_validator(regions, *, tolerance=1e-7):
    """Clip a line against each polytope, then require full interval coverage.

    Membership of endpoints alone is insufficient for a nonconvex union.
    This retains the existing cover guarantees (including finite collision
    audits); it does not upgrade them to a continuous collision certificate.
    """
    if not regions or not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Nonempty cover and finite nonnegative tolerance required')
    counts = [len(region.b()) for region in regions]
    if len(set(counts)) == 1:
        matrices = np.asarray([r.A() for r in regions])
        bounds = np.asarray([r.b() for r in regions])
    else:
        # Pad variable-size halfspace systems with inactive zero inequalities.
        matrices = np.zeros((len(regions), max(counts), regions[0].ambient_dimension()))
        bounds = np.zeros((len(regions), max(counts)))
        for i, region in enumerate(regions):
            matrices[i, :counts[i]] = region.A()
            bounds[i, :counts[i]] = region.b()

    def valid(a, b):
        slack = bounds+tolerance-np.einsum('ijk,k->ij', matrices, a)
        slope = np.einsum('ijk,k->ij', matrices, b-a)
        parallel = np.abs(slope) < 1e-14
        feasible = np.all(~parallel | (slack >= 0), axis=1)
        ratio = np.divide(slack, slope, out=np.zeros_like(slack), where=~parallel)
        lower = np.maximum(0., np.max(np.where(slope < -1e-14, ratio, -np.inf), axis=1))
        upper = np.minimum(1., np.min(np.where(slope > 1e-14, ratio, np.inf), axis=1))
        intervals = sorted(zip(lower[feasible & (lower <= upper)], upper[feasible & (lower <= upper)]))
        reached = 0.
        for lo, hi in intervals:
            if lo > reached+1e-12:
                return False
            reached = max(reached, hi)
            if reached >= 1.-1e-12:
                return True
        return False
    return valid
