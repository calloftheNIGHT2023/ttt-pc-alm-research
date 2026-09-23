"""Support-only local-credit activity-ray proposals; no global-BP candidate."""
from fractions import Fraction as F
import numpy as np
import batched_credit_bank_memory as old
base = old.base


def codes_from_h(x, b, h):
    return np.searchsorted(base.KNOTS, np.vstack([x, h[:-1]]) + b[:, None], side='right').astype(np.uint8)


def ray_patterns(x, b, h, a):
    """Complete real ray in exact arithmetic; float implementation proposes only."""
    scale = float(np.max(np.abs(a[:-1])))
    if scale == 0: return {codes_from_h(x, b, h).tobytes()}
    direction = a / scale; aa = direction[:-1]
    targets = base.KNOTS[:, None, None] - b[None, 1:, None]
    valid = (targets >= 0) & (targets <= 1) & (aa[None] != 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        times = (targets - h[None, :-1]) / aa[None]
    events = np.unique(times[valid & np.isfinite(times)]); result = set()
    low = np.where(direction > 0, 0., np.where(direction < 0, 1., h))
    high = np.where(direction > 0, 1., np.where(direction < 0, 0., h))
    result.update([codes_from_h(x, b, low).tobytes(), codes_from_h(x, b, high).tobytes(), codes_from_h(x, b, h).tobytes()])
    for t in events:
        result.add(codes_from_h(x, b, np.clip(h + t * direction, 0, 1)).tobytes())
    for lo, hi in zip(events[:-1], events[1:]):
        mid = lo / 2 + hi / 2
        if not lo < mid < hi: continue
        moved = np.clip(h + mid * direction, 0, 1)
        result.add(codes_from_h(x, b, moved).tobytes())
    return result


class CapturingCollector(old.Collector):
    instances = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.snapshots = {}; type(self).instances.append(self)

    def activity(self, b, h, u=None, step=0, phase=''):
        before = {k: id(value[1]) for k, value in self.pending.items()}
        super().activity(b, h, u, step, phase)
        changed = [k for k, value in self.pending.items() if before.get(k) != id(value[1])]
        if not changed: return
        r, d = b.shape; prev = np.broadcast_to(self.x, (r, len(self.x))); rr = []; ss = []
        for j in range(d):
            z = prev + b[:, j, None]; ss.append(np.searchsorted(base.KNOTS, z, side='right').astype(np.uint8))
            rr.append(h[j] - base.g(z)); prev = h[j]
        res = np.array(rr).transpose(1, 0, 2); split = np.array(ss).transpose(1, 0, 2)
        credits = dict(residual=res)
        if self.kind == 'local':
            credits.update(raw=u.transpose(1, 0, 2), augmented=u.transpose(1, 0, 2) + res)
        elif self.kind == 'bp':
            credits.update(current_bp=self.current, history_bp=self.history, combined_bp=self.current + self.history)
        for key in changed:
            _, a, label = self.pending[key]; possible = np.flatnonzero(np.all(credits[label] == a, axis=(1, 2)))
            found = False
            for source, regs in [('forward', self.codes), ('split', split)]:
                for i in possible:
                    if regs[i].tobytes() != key: continue
                    if source == 'forward':
                        parent = []; hp = self.x
                        for bias in b[i]: hp = base.g(hp + bias); parent.append(hp)
                        parent = np.array(parent)
                    else: parent = h[:, i].copy()
                    assert codes_from_h(self.x, b[i], parent).tobytes() == key
                    self.snapshots[key] = dict(b=b[i].copy(), h=parent, residual=res[i].copy(), a=a.copy(), label=label,
                        step=int(step), phase=phase, restart=int(i), source=source)
                    found = True; break
                if found: break
            assert found, (label, step, phase)


def capture(x, v, cfg):
    previous = old.Collector; CapturingCollector.instances = []
    try:
        old.Collector = CapturingCollector
        _, _, meta, proofs = old.prepare(x, v, dict(**cfg, bank_size=0, capture_proofs=True))
    finally: old.Collector = previous
    assert len(CapturingCollector.instances) == 1; obs = CapturingCollector.instances[0]
    chosen = {}
    for proof in proofs:
        key = bytes.fromhex(proof['pattern']); snapshot = obs.snapshots[key]
        assert np.array_equal(snapshot['a'], proof['a']); chosen[key] = snapshot
    assert len(chosen) == meta['matching_certified_patterns']
    endpoint = set(obs.forward) | set(obs.split); assert len(endpoint) == meta['original_pattern_count']
    return chosen, endpoint, meta


def exact_ray_patterns(x, b, h, a):
    d, n = h.shape; ff = lambda arr: [[F(float(z)) for z in row] for row in arr]
    hh = ff(h); aa = ff(a); bb = [F(float(z)) for z in b]; xx = [F(float(z)) for z in x]; knots = [F(0), F(1, 2), F(1)]
    events = set()
    for j in range(d - 1):
        for i in range(n):
            if aa[j][i] == 0: continue
            for knot in knots:
                target = knot - bb[j + 1]
                if 0 <= target <= 1: events.add((target - hh[j][i]) / aa[j][i])
    events = sorted(events)
    tests = [F(0)] if not events else [events[0] - 1, events[-1] + 1] + [(l + r) / 2 for l, r in zip(events[:-1], events[1:])]
    result = set()
    for t in tests:
        codes = [[sum(k <= xx[i] + bb[0] for k in knots) for i in range(n)]]
        for j in range(1, d):
            codes.append([sum(k <= min(F(1), max(F(0), hh[j - 1][i] + t * aa[j - 1][i])) + bb[j] for k in knots) for i in range(n)])
        result.add(np.array(codes, np.uint8).tobytes())
    return result


def verify():
    rng = np.random.default_rng(816333); cases = 0; probes = 0; boundary_probes = 0
    for _ in range(80):
        x = rng.integers(0, 17, 4) / 16; b = rng.integers(-2, 3, 4) / 16
        h = rng.integers(0, 17, (4, 4)) / 16; a = rng.integers(-8, 9, (4, 4)) / 16
        actual = ray_patterns(x, b, h, a); exact = exact_ray_patterns(x, b, h, a)
        assert exact <= actual; cases += 1
        for t in np.linspace(-5, 5, 101):
            moved = np.clip(h + t * a, 0, 1)
            z = moved[:-1] + b[1:, None]
            if np.min(np.abs(z[None] - base.KNOTS[:, None, None])) < 1e-11:
                boundary_probes += 1; continue
            assert codes_from_h(x, b, moved).tobytes() in actual; probes += 1
        assert ray_patterns(x, b, h, a * 2) == actual
    return dict(passed=True, exact_rational_ray_cases=cases, nonboundary_dense_ray_probes=probes, boundary_dense_probes_not_used_as_completeness_test=boundary_probes, positive_rescaling_exact_cases=cases,
        finite_float_proposals_not_certificates=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify(), indent=2))
