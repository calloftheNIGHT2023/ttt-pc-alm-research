"""Finite projected primal rays from local Lagrangian credit, proposal-only."""
import numpy as np
import credit_activity_ray_proposals_v2 as previous
base = previous.base
codes_from_h = previous.previous.codes_from_h


def direction(x, b, h, a):
    regs = codes_from_h(x, b, h); p = base.SLOPES[regs] * a
    db = p.sum(1) / len(x); dh = np.zeros_like(h); dh[:-1] = p[1:] - a[:-1]
    return db, dh


def ray_patterns(x, b, h, a):
    db, dh = direction(x, b, h, a)
    scale = max(float(np.max(np.abs(db))), float(np.max(np.abs(dh))))
    if scale == 0: return {codes_from_h(x, b, h).tobytes()}
    db = db / scale; dh = dh / scale; d, n = h.shape
    q = np.concatenate([b, h[:-1].ravel()]); step = np.concatenate([db, dh[:-1].ravel()])
    lower = np.concatenate([np.full(d, -.12), np.zeros((d - 1) * n)])
    upper = np.concatenate([np.full(d, .12), np.ones((d - 1) * n)])
    assert np.all(q >= lower - 1e-14) and np.all(q <= upper + 1e-14)
    nz = step != 0
    hits = (np.where(step[nz] > 0, upper[nz], lower[nz]) - q[nz]) / step[nz]
    hits = hits[np.isfinite(hits) & (hits >= 0)]
    end = max(1., float(hits.max()) + max(1., float(hits.max()) * .1)) if len(hits) else 1.
    bounds = np.unique(np.concatenate(([0., end], hits))); events = set(map(float, bounds))
    for lo, hi in zip(bounds[:-1], bounds[1:]):
        mid = lo / 2 + hi / 2
        if not lo < mid < hi: continue
        raw = q + mid * step; active = (raw > lower) & (raw < upper)
        slope = np.where(active, step, 0); intercept = np.where(active, q, np.where(raw <= lower, lower, upper))
        bs, bc = slope[:d], intercept[:d]
        hs = np.vstack([np.zeros(n), slope[d:].reshape(d - 1, n)])
        hc = np.vstack([x, intercept[d:].reshape(d - 1, n)])
        zs = hs + bs[:, None]; zc = hc + bc[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            roots = (base.KNOTS[:, None, None] - zc[None]) / zs[None]
        events.update(map(float, roots[np.isfinite(roots) & (roots > lo) & (roots < hi)]))
    ordered = sorted(events); tests = ordered + [lo / 2 + hi / 2 for lo, hi in zip(ordered[:-1], ordered[1:]) if lo < lo / 2 + hi / 2 < hi]
    result = set()
    for t in tests:
        bb = np.clip(b + t * db, -.12, .12); hh = np.clip(h + t * dh, 0, 1)
        result.add(codes_from_h(x, bb, hh).tobytes())
    return result


def energy(x, b, h, a):
    return float(np.sum(a * (h - base.g(np.vstack([x, h[:-1]]) + b[:, None]))))


def verify():
    rng = np.random.default_rng(817344); errors = []; probes = 0; near = 0; zero = 0
    for _ in range(80):
        x = rng.uniform(0, 1, 4); b = rng.uniform(-.1, .1, 4); h = rng.uniform(.05, .95, (4, 4)); a = rng.normal(size=h.shape)
        db, dh = direction(x, b, h, a); numerical = (energy(x, b + 1e-7 * db, h + 1e-7 * dh, a) - energy(x, b - 1e-7 * db, h - 1e-7 * dh, a)) / 2e-7
        expected = -len(x) * float(np.sum(db ** 2)) - float(np.sum(dh ** 2))
        errors.append(abs(numerical - expected)); assert abs(numerical - expected) < 2e-7
        found = ray_patterns(x, b, h, a); assert found == ray_patterns(x, b, h, a * 2)
        assert ray_patterns(x, b, h, np.zeros_like(a)) == {codes_from_h(x, b, h).tobytes()}; zero += 1
        for t in np.concatenate([np.linspace(0, 10, 101), [20., 100., 1000.]]):
            bb = np.clip(b + t * db, -.12, .12); hh = np.clip(h + t * dh, 0, 1); z = np.vstack([x, hh[:-1]]) + bb[:, None]
            if np.min(np.abs(z[None] - base.KNOTS[:, None, None])) < 1e-10: near += 1; continue
            assert codes_from_h(x, bb, hh).tobytes() in found; probes += 1
    return dict(passed=True, finite_difference_direction_checks=len(errors), maximum_derivative_error=max(errors), nonboundary_dense_probes=probes,
        boundary_probes_excluded_from_completeness=near, zero_direction_cases=zero, positive_scale_cases=zero, proposal_only=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
