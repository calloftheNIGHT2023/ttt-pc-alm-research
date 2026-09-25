"""400 exact partial posterior primitives; no search or performance claim.

All acceptance arithmetic is rational. A proposal may come from any method.
Caller must establish that the feasible cells cover the posterior, and that
different cells' interiors are disjoint, before adding their certified mass.
No numerical hull volume or Monte Carlo average is accepted as a proof.
"""
from fractions import Fraction as F
from itertools import combinations
from math import factorial
from probe_simplex_readout_intervals import enclosure as existing_four_layer_enclosure


def vector(x):
    return tuple(F(t) for t in x)


def dot(a, b):
    assert len(a) == len(b)
    return sum((x * y for x, y in zip(a, b)), F(0))


def determinant(matrix):
    """Exact Gaussian elimination, including singular matrices."""
    a = [list(map(F, row)) for row in matrix]
    n = len(a)
    assert all(len(row) == n for row in a)
    result = F(1)
    for j in range(n):
        pivot = next((i for i in range(j, n) if a[i][j]), None)
        if pivot is None:
            return F(0)
        if pivot != j:
            a[j], a[pivot] = a[pivot], a[j]
            result = -result
        value = a[j][j]
        result *= value
        for i in range(j + 1, n):
            ratio = a[i][j] / value
            for k in range(j + 1, n):
                a[i][k] -= ratio * a[j][k]
    return result


def simplex_volume(vertices):
    p = tuple(map(vector, vertices))
    d = len(p[0])
    assert len(p) == d + 1 and all(len(v) == d for v in p)
    return abs(determinant([[p[i][j] - p[0][j] for j in range(d)]
                            for i in range(1, d + 1)])) / factorial(d)


def feasible(point, a, rhs):
    assert len(a) == len(rhs)
    return all(dot(row, point) <= b for row, b in zip(a, rhs))


def contract_cloud(vertices, anchor, a, rhs, safety=F(1048575, 1048576)):
    """A common homothety gives exactly feasible vertices, not exact coverage."""
    p, c = tuple(map(vector, vertices)), vector(anchor)
    a, rhs = tuple(map(vector, a)), vector(rhs)
    safety = F(safety)
    assert p and 0 < safety < 1 and all(len(v) == len(c) for v in p)
    slacks = [b - dot(row, c) for row, b in zip(a, rhs)]
    assert all(s > 0 if any(row) else s >= 0 for row, s in zip(a, slacks))
    lam = F(1)
    for row, slack in zip(a, slacks):
        for v in p:
            rise = dot(row, [x - y for x, y in zip(v, c)])
            if rise > 0:
                lam = min(lam, slack / rise)
    lam *= safety
    assert 0 < lam < 1
    inside = tuple(tuple(y + lam * (x - y) for x, y in zip(v, c)) for v in p)
    assert all(feasible(v, a, rhs) for v in inside)
    return inside, lam


def supporting_plane(face, cloud, apex):
    """Verify an exact supporting hyperplane of the entire finite point cloud."""
    d = len(apex)
    assert len(face) == d
    edges = [[face[i][j] - face[0][j] for j in range(d)] for i in range(1, d)]
    normal = tuple((-1)**j * determinant([[v[k] for k in range(d) if k != j]
                                         for v in edges]) for j in range(d))
    if not any(normal):
        return None
    rhs = dot(normal, face[0])
    values = [dot(normal, v) - rhs for v in cloud]
    assert all(dot(normal, v) == rhs for v in face)
    if not (all(t <= 0 for t in values) or all(t >= 0 for t in values)):
        return None
    if dot(normal, apex) == rhs:
        return None
    # Same exact plane has one canonical key irrespective of orientation.
    first = next(t for t in normal if t)
    return tuple(t / first for t in normal) + (rhs / first,)


def certified_cones(vertices, face_indices, anchor, a, rhs):
    """Return non-overlapping interior simplices inside one feasible cell.

Faces are only proposals. Reject non-supporting faces; retain at most one
simplex per supporting plane. Cones from a common interior apex to distinct
support planes overlap only on null sets. This is a LOWER volume bound;
discarding coplanar triangulations can make it loose even for a perfect hull.
"""
    cloud, lam = contract_cloud(vertices, anchor, a, rhs)
    d = len(cloud[0])
    apex = tuple(sum((v[j] for v in cloud), F(0)) / len(cloud) for j in range(d))
    used, cones, reasons = set(), [], dict(degenerate_or_unsupported=0, duplicate_plane=0)
    for indices in face_indices:
        indices = tuple(int(i) for i in indices)
        assert len(indices) == d and len(set(indices)) == d
        assert all(0 <= i < len(cloud) for i in indices)
        face = tuple(cloud[i] for i in indices)
        key = supporting_plane(face, cloud, apex)
        if key is None:
            reasons['degenerate_or_unsupported'] += 1
            continue
        if key in used:
            reasons['duplicate_plane'] += 1
            continue
        simplex = (apex,) + face
        mass = simplex_volume(simplex)
        assert mass > 0 and all(feasible(v, a, rhs) for v in simplex)
        used.add(key)
        cones.append(simplex)
    return dict(simplices=cones, mass=sum((simplex_volume(s) for s in cones), F(0)),
                contraction=lam, rejections=reasons, supporting_planes=sorted(used))


def tent(z):
    z = F(z)
    return max(F(0), 1 - abs(2 * z - 1))


def forward(q, bias):
    h = F(q)
    for b in bias:
        h = tent(h + b)
    return h


def tent_interval(lo, hi):
    lo, hi = F(lo), F(hi)
    assert lo <= hi
    left, right = tent(lo), tent(hi)
    return min(left, right), F(1) if lo <= F(1, 2) <= hi else max(left, right)


def simplex_range(vertices, q):
    """Dependency-losing interval propagation; rigorous but can be loose."""
    p = tuple(map(vector, vertices))
    lo = hi = F(q)
    for j in range(len(p[0])):
        lo, hi = tent_interval(lo + min(v[j] for v in p), hi + max(v[j] for v in p))
    return lo, hi


def affine_mean(vertices, q):
    """Exact integral average if one closed affine branch holds at every layer."""
    p = tuple(map(vector, vertices))
    h = [F(q)] * len(p)
    for j in range(len(p[0])):
        z = [v + b[j] for v, b in zip(h, p)]
        lo, hi = min(z), max(z)
        if not (hi <= 0 or 0 <= lo <= hi <= F(1, 2) or
                F(1, 2) <= lo <= hi <= 1 or lo >= 1):
            return None
        h = list(map(tent, z))
    return sum(h, F(0)) / len(h)


def bisect_simplex(vertices):
    p = tuple(map(vector, vertices))
    pairs = list(combinations(range(len(p)), 2))
    i, j = max(pairs, key=lambda ij: sum((a - b)**2 for a, b in zip(p[ij[0]], p[ij[1]])))
    midpoint = tuple((a + b) / 2 for a, b in zip(p[i], p[j]))
    first, second = list(p), list(p)
    first[i], second[j] = midpoint, midpoint
    return tuple(first), tuple(second)


def moment_bounds(vertices, q, splits=0):
    """Return exact lower/upper integral; splits is max recursion depth."""
    assert isinstance(splits, int) and splits >= 0
    mass = simplex_volume(vertices)
    mean = affine_mean(vertices, q)
    if mean is not None:
        return dict(mass=mass, lower=mass * mean, upper=mass * mean, leaves=1, affine_leaves=1)
    if splits:
        first, second = [moment_bounds(s, q, splits - 1) for s in bisect_simplex(vertices)]
        assert first['mass'] + second['mass'] == mass
        return {key:first[key] + second[key] for key in first}
    lo, hi = simplex_range(vertices, q)
    if len(vertices[0]) == 4:
        # Reuse the already developed support-only rational enclosure. Its
        # propagation retains affine dependence until the first crossing.
        old_lo, old_hi, _ = existing_four_layer_enclosure(F(q), tuple(map(vector, vertices)), 4)
        lo, hi = max(lo, old_lo), min(hi, old_hi)
        assert lo <= hi
    return dict(mass=mass, lower=mass * lo, upper=mass * hi, leaves=1, affine_leaves=0)


def posterior_interval(mass, lower_moment, upper_moment, total_upper):
    """Z <= total_upper; inner mass/moments exact; outputs in [0,1].

Existence of a positive posterior normalizer is a caller precondition. The
upper mass bound MUST cover all feasible cells, including unsearched ones.
"""
    mass, lo, hi, upper = map(F, (mass, lower_moment, upper_moment, total_upper))
    assert upper > 0 and 0 <= lo <= hi <= mass <= upper
    return lo / upper, 1 - (mass - hi) / upper


def intersect_intervals(old, new):
    lo, hi = max(F(old[0]), F(new[0])), min(F(old[1]), F(new[1]))
    assert lo <= hi, 'Inconsistent certificates: never silently repair.'
    return lo, hi


def project(prediction, interval):
    lo, hi = map(F, interval)
    assert lo <= hi
    return min(hi, max(lo, F(prediction)))


def row_parallelotope_upper(a, rhs, indices):
    """Exact volume upper from d paired existing row directions.

Rows must bound the whole cell. Bounds in coordinates y=T theta use only
positive/negative proportional rows already supplied, not numerical LP
optima. Enumerating bases or proposing good directions is external/charged.
"""
    a, rhs = tuple(map(vector, a)), vector(rhs)
    assert a and len(a) == len(rhs)
    d = len(a[0])
    assert len(indices) == d
    matrix = tuple(a[i] for i in indices)
    det = abs(determinant(matrix))
    if not det:
        return None
    widths, intervals = [], []
    for direction in matrix:
        j = next(j for j, t in enumerate(direction) if t)
        low, high = [], []
        for row, b in zip(a, rhs):
            scale = row[j] / direction[j]
            if not scale or any(t != scale * u for t, u in zip(row, direction)):
                continue
            (high if scale > 0 else low).append(b / scale)
        if not low or not high:
            return None
        lo, hi = max(low), min(high)
        if lo > hi:
            return dict(volume=F(0), empty=True, determinant=det)
        intervals.append((lo, hi))
        widths.append(hi - lo)
    volume = F(1)
    for width in widths:
        volume *= width
    return dict(volume=volume / det, empty=False, determinant=det, intervals=intervals)
