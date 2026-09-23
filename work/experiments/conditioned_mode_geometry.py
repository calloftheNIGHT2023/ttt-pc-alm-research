"""Common numerical repair of a failed volume assertion, not a search method.

Successful frozen geometry calls are returned unchanged. On the one declared
volume-consistency exception, recompute halfspace intersections in whitened
coordinates. Never jitter vertices, relax observations, or access query truth.
The failed original geometry work and the repair are both charged by fit().
"""
from contextlib import contextmanager
import hashlib
import math
import time
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import region_posterior_memory as geometry
import matched_budget_suite as suite

ORIGINAL = geometry.polytope


def boundary_residual(facets, origin):
    edges = facets[:, 1:]-facets[:, :1]
    normals = np.stack([(-1)**j*np.linalg.det(np.delete(edges, j, axis=2)) for j in range(4)], axis=1)
    sign = np.where(np.sum(normals*(facets[:, 0]-origin), axis=1) < 0, -1., 1.)
    normals *= sign[:, None]
    return float(np.linalg.norm(normals.sum(0))/max(np.linalg.norm(normals, axis=1).sum(), 1e-300))


def conditioned_polytope(g, rhs):
    """Invertible change of coordinates; all checks stay explicit and strict."""
    d = g.shape[1]
    assert d == 4
    a = np.r_[g, np.eye(d), -np.eye(d)]
    r = np.r_[rhs, np.full(2*d, geometry.PRIOR_BOUND)]
    norm = np.linalg.norm(a, axis=1)
    keep = norm >= 1e-14
    assert np.all(r[~keep] >= -1e-12)
    a, r = a[keep]/norm[keep, None], r[keep]/norm[keep]
    lows, highs = [], []
    for e in np.eye(d):
        lo = linprog(e, A_ub=a, b_ub=r, bounds=[(None, None)]*d,
                     options={'primal_feasibility_tolerance': 1e-9})
        hi = linprog(-e, A_ub=a, b_ub=r, bounds=[(None, None)]*d,
                     options={'primal_feasibility_tolerance': 1e-9})
        assert lo.success and hi.success
        lows.append(e@lo.x)
        highs.append(e@hi.x)
    center = (np.array(lows)+highs)/2
    scale = (np.array(highs)-lows)/2
    assert np.all(scale > 1e-12)
    ag, rr = a*scale[None], r-a@center
    cc = linprog(np.r_[np.zeros(d), -1.],
                 A_ub=np.c_[ag, np.linalg.norm(ag, axis=1)], b_ub=rr,
                 bounds=[(None, None)]*d+[(0, None)],
                 options={'primal_feasibility_tolerance': 1e-9})
    assert cc.success and cc.x[-1] > 1e-10
    old_vertices = HalfspaceIntersection(np.c_[ag, -rr], cc.x[:-1]).intersections
    centroid = old_vertices.mean(0)
    _, singular, vt = np.linalg.svd(old_vertices-centroid, full_matrices=False)
    assert np.isfinite(singular).all() and singular[-1] > 0
    transform = vt.T*(singular/np.sqrt(len(old_vertices)))[None]
    aw, rw = ag@transform, rr-ag@centroid
    assert np.min(rw) > 0
    vertices = HalfspaceIntersection(np.c_[aw, -rw], np.zeros(d)).intersections
    assert np.isfinite(vertices).all()
    hull = ConvexHull(vertices)
    white_facets = vertices[hull.simplices]
    cones = np.abs(np.linalg.det(white_facets))/24
    total = float(cones.sum())
    assert total > 0 and np.isclose(total, hull.volume, rtol=1e-8, atol=1e-20)
    closure = boundary_residual(white_facets, np.zeros(4))
    assert closure < 1e-10, closure
    assert np.max(vertices@aw.T-rw) < 1e-10
    # Restore the existing sample() interface exactly; no new sampling rule.
    facets = centroid+white_facets@transform.T
    jacobian = abs(float(np.linalg.det(transform)))
    restored_total = float(np.abs(np.linalg.det(facets-centroid)).sum()/24)
    assert np.isclose(restored_total, total*jacobian, rtol=1e-8, atol=1e-20)
    assert np.max(facets.reshape(-1, d)@ag.T-rr) < 1e-10
    volume = total*jacobian*float(np.prod(scale))
    poly = dict(center=center, scale=scale, interior=centroid, facets=facets,
                simplex_probs=cones/total, volume=volume, a=a, rhs=r)
    note = dict(reason='positive_volume', volume=volume, vertices=len(vertices), facets=len(facets),
        repair='affine_whitened_halfspace_reintersection', condition=float(singular[0]/singular[-1]),
        cone_hull_relative_gap=float(abs(total-hull.volume)/hull.volume), boundary_relative_residual=closure,
        transform_jacobian=jacobian, no_observation_or_prior_change=True)
    return poly, note


@contextmanager
def geometry_scope(log):
    assert geometry.polytope is ORIGINAL
    def guarded(g, rhs):
        try:
            return ORIGINAL(g, rhs)
        except AssertionError as exc:
            # The frozen polytope has exactly one assertion: two volume scalars.
            pair = exc.args[0] if len(exc.args) == 1 else None
            if not (isinstance(pair, tuple) and len(pair) == 2 and
                    all(isinstance(t, (float, np.floating)) and np.isfinite(t) and t > 0 for t in pair)):
                raise
            record = dict(original_volume_pair=[float(t) for t in pair],
                constraints_sha256=hashlib.sha256(g.tobytes()+rhs.tobytes()).hexdigest(), completed=False)
            log.append(record)
            poly, note = conditioned_polytope(g, rhs)
            record.update(completed=True, repair_note=note)
            return poly, note
    geometry.polytope = guarded
    try:
        yield
    finally:
        geometry.polytope = ORIGINAL


def fit(cfg, x, v, q, seed, loaded):
    start = time.perf_counter()
    log = []
    with geometry_scope(log):
        arrays, meta = suite.guarded_fit(cfg, x, v, q, seed, loaded)
    meta = {**meta, 'geometry_repair_count': len(log), 'geometry_repair_log': log,
            'charged_complete_seconds': time.perf_counter()-start}
    return arrays, meta
