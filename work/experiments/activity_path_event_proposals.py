"""Read-only event proposals along observed activity/parameter segments.

Floating arithmetic is a proposal mechanism, never a rejection certificate.
No query, truth, full-mode reference or global BP enters the local path.
"""
import time
from fractions import Fraction
import numpy as np
import split_activity_mode_memory as old
base = old.base


def affine_segment_patterns(z0, z1):
    """Open-interval modes of a straight preactivation segment (proposal-only)."""
    delta = z1 - z0; changed = delta != 0
    roots = ((base.KNOTS[:, None] - z0[changed][None]) / delta[changed][None]).ravel() if np.any(changed) else np.empty(0)
    events = np.unique(np.concatenate(([0., 1.], roots[(roots > 0) & (roots < 1)])))
    result = {}
    for lo, hi in zip(events[:-1], events[1:]):
        mid = lo + (hi - lo) * .5
        if not lo < mid < hi: continue
        codes = np.searchsorted(base.KNOTS, z0 + mid * delta, side='right').astype(np.uint8)
        result.setdefault(codes.tobytes(), float(mid))
    return result


def parameter_segment_patterns(x, b0, b1):
    """Piecewise-affine forward partition in a single parameter path variable."""
    delta = b1 - b0; pieces = [(0., 1., np.zeros_like(x), x.copy())]
    for j in range(len(b0)):
        output = []
        for lo, hi, slope, intercept in pieces:
            zs = slope + delta[j]; zc = intercept + b0[j]; changed = zs != 0
            roots = ((base.KNOTS[:, None] - zc[changed][None]) / zs[changed][None]).ravel() if np.any(changed) else np.empty(0)
            events = np.unique(np.concatenate(([lo, hi], roots[(roots > lo) & (roots < hi)])))
            for left, right in zip(events[:-1], events[1:]):
                mid = left + (right - left) * .5
                if not left < mid < right: continue
                codes = np.searchsorted(base.KNOTS, zs * mid + zc, side='right')
                s = base.SLOPES[codes]; c = base.INTERCEPTS[codes]
                output.append((left, right, s * zs, s * zc + c))
        pieces = output
    result = {}
    for lo, hi, _, _ in pieces:
        mid = lo + (hi - lo) * .5
        # Always emit codes of the actual forward at the proposed point.
        codes = base.pattern(x, b0 + mid * delta).astype(np.uint8)
        result.setdefault(codes.tobytes(), float(mid))
    return result


class Collector:
    def __init__(self, x):
        self.x = x; self.forward = {}; self.split = {}; self.parameter_path = {}; self.activity_path = {}
        self.last_b = None; self.last_z = None; self.parameter_seconds = 0.; self.activity_seconds = 0.
        self.parameter_segments = 0; self.activity_segments = 0; self.max_snapshot_bytes = 0

    def parameter(self, b, step=0):
        start = time.perf_counter()
        for reg in old.old.interface.archived.signatures(self.x, b): self.forward.setdefault(reg.tobytes(), None)
        if self.last_b is not None:
            for restart, (before, after) in enumerate(zip(self.last_b, b)):
                if np.array_equal(before, after): continue
                self.parameter_segments += 1
                for key, t in parameter_segment_patterns(self.x, before, after).items():
                    self.parameter_path.setdefault(key, dict(step=int(step), restart=restart, t=t))
        self.last_b = b.copy(); self.parameter_seconds += time.perf_counter() - start

    def activity(self, b, h, u=None, step=0, phase=''):
        start = time.perf_counter(); r, d = b.shape; prev = np.broadcast_to(self.x, (r, len(self.x))); zz = []
        for j in range(d): zz.append(prev + b[:, j, None]); prev = h[j]
        z = np.array(zz).transpose(1, 0, 2); codes = np.searchsorted(base.KNOTS, z, side='right').astype(np.uint8)
        for reg in codes: self.split.setdefault(reg.tobytes(), None)
        if self.last_z is not None:
            for restart, (before, after) in enumerate(zip(self.last_z, z)):
                if np.array_equal(before, after): continue
                self.activity_segments += 1
                for key, t in affine_segment_patterns(before, after).items():
                    self.activity_path.setdefault(key, dict(step=int(step), restart=restart, t=t, phase=phase))
        self.last_z = z.copy()
        self.max_snapshot_bytes = max(self.max_snapshot_bytes, self.last_z.nbytes + (0 if self.last_b is None else self.last_b.nbytes))
        self.activity_seconds += time.perf_counter() - start


def discover(x, v, cfg):
    pool = old.old.interface.make_pool(4, 'prior256'); anchor = np.zeros(4)
    starts, meta = old.old.interface.select_pool(x, v, anchor, pool, 64); obs = Collector(x)
    with old.old.core.pipeline.discovery_box(.12):
        if cfg['generator'] != 'adam':
            best = old.trace.refine(starts, x, v, anchor, cfg['generator'], cfg['sweeps'], obs)
            expected = old.trace.original(starts, x, v, anchor, cfg['generator'], cfg['sweeps'])
        else:
            evaluate = old.old.core.batched.evaluate
            def wrapped(b, *args, **kwargs):
                obs.parameter(b); obs.activity(b, old.one_pass_relaxation(b, x, v))
                return evaluate(b, *args, **kwargs)
            try:
                old.old.core.batched.evaluate = wrapped
                best, _ = old.old.core.batched.refine(starts, x, v, anchor, solver='adam', steps=cfg['steps'], lr=.003)
            finally: old.old.core.batched.evaluate = evaluate
            expected, _ = old.old.core.batched.refine(starts, x, v, anchor, solver='adam', steps=cfg['steps'], lr=.003)
    assert np.array_equal(best, expected)
    _, regs, _ = old.discover(x, v, dict(**cfg, restarts=64))
    assert set(obs.forward) | set(obs.split) == {r.tobytes() for r in regs}
    return obs, dict(**meta, original_best_bitwise=True, original_endpoint_universe_exact=True)


def exact_affine_patterns(z0, z1):
    before = [Fraction(float(t)) for t in z0.ravel()]; after = [Fraction(float(t)) for t in z1.ravel()]
    knots = [Fraction(0), Fraction(1, 2), Fraction(1)]; events = {Fraction(0), Fraction(1)}
    for left, right in zip(before, after):
        if left == right: continue
        events.update(t for k in knots if 0 < (t := (k - left) / (right - left)) < 1)
    ordered = sorted(events); result = set()
    for lo, hi in zip(ordered[:-1], ordered[1:]):
        mid = (lo + hi) / 2
        code = [sum(k <= left + mid * (right - left) for k in knots) for left, right in zip(before, after)]
        result.add(np.array(code, np.uint8).reshape(z0.shape).tobytes())
    return result


def verify():
    rng = np.random.default_rng(814993); affine_cases = 0; parameter_cases = 0; probes = 0
    for _ in range(80):
        # Dyadic endpoints provide exact independently ordered rational events.
        z0 = rng.integers(-8, 25, (4, 4)) / 16; z1 = rng.integers(-8, 25, (4, 4)) / 16
        exact = exact_affine_patterns(z0, z1); actual = set(affine_segment_patterns(z0, z1))
        assert exact <= actual; affine_cases += 1
    for _ in range(40):
        x = rng.uniform(0, 1, 4); b0 = rng.uniform(-.12, .12, 4); b1 = rng.uniform(-.12, .12, 4)
        found = set(parameter_segment_patterns(x, b0, b1)); found.update([base.pattern(x, b0).astype(np.uint8).tobytes(), base.pattern(x, b1).astype(np.uint8).tobytes()])
        assert all(len(key) == len(x) * len(b0) for key in found)
        for t in np.linspace(0, 1, 501):
            assert base.pattern(x, b0 + t * (b1 - b0)).astype(np.uint8).tobytes() in found; probes += 1
        parameter_cases += 1
    before_jac = base.forward_jacobian; before_refine = old.old.core.batched.refine
    def forbidden(*args, **kwargs): raise AssertionError('global BP entered local proposal path')
    try:
        base.forward_jacobian = forbidden; old.old.core.batched.refine = forbidden
        discover(x, base.forward(x, b1), dict(generator='alm', sweeps=2))
    finally: base.forward_jacobian = before_jac; old.old.core.batched.refine = before_refine
    return dict(passed=True, exact_rational_activity_segment_cases=affine_cases, parameter_segment_cases=parameter_cases,
        dense_forward_probes=probes, local_path_no_global_bp=True, floating_enumeration_not_claimed_complete=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
