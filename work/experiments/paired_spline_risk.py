"""Evaluator-only analytical input integrals for fixed 1D tent predictors.

No fitting or query-target-based decisions. Numerical posterior volumes and
posterior Monte Carlo remain approximations; only the q dimension is integrated
segment by segment. Uses the already verified scalar spline compiler as oracle.
"""
import numpy as np
import compiled_piecewise_readout as compiler
base = compiler.base


class Spline:
    def __init__(self, knots, values):
        self.knots = np.asarray(knots, dtype=float)
        self.values = np.asarray(values, dtype=float)
        self.widths = np.diff(self.knots)
        assert self.knots[0] == 0 and self.knots[-1] == 1
        assert np.all(self.widths > 0) and len(self.values) == len(self.knots)
        self.slopes = np.diff(self.values) / self.widths
        w = self.widths; left = self.knots[:-1]; y = self.values[:-1]; s = self.slopes
        i0 = y * w + s * w * w / 2
        i1 = left * i0 + y * w * w / 2 + s * w ** 3 / 3
        self.cum0 = np.r_[0., np.cumsum(i0)]
        self.cum1 = np.r_[0., np.cumsum(i1)]

    def __call__(self, q):
        q = np.asarray(q); assert np.all((q >= 0) & (q <= 1))
        i = np.clip(np.searchsorted(self.knots, q, side='right') - 1, 0, len(self.slopes) - 1)
        return self.values[i] + self.slopes[i] * (q - self.knots[i])

    def primitive(self, q):
        q = np.asarray(q); assert np.all((q >= 0) & (q <= 1))
        i = np.clip(np.searchsorted(self.knots, q, side='right') - 1, 0, len(self.slopes) - 1)
        t = q - self.knots[i]; y = self.values[i]; s = self.slopes[i]
        v0 = y * t + s * t * t / 2
        v1 = self.knots[i] * v0 + y * t * t / 2 + s * t ** 3 / 3
        return self.cum0[i] + v0, self.cum1[i] + v1


def from_bank(bank, weights=None):
    if weights is None: weights = np.full(len(bank), 1 / len(bank))
    cache = compiler.prepare(bank); predict, _ = compiler.compile_weights(cache, np.asarray(weights))
    knots = np.unique(np.r_[0., cache['knots'], 1.])
    return Spline(knots, predict(knots))


def product_integral(left, right):
    knots = np.union1d(left.knots, right.knots); x = left(knots); y = right(knots)
    return float(np.sum(np.diff(knots) * (2*x[:-1]*y[:-1] + x[:-1]*y[1:] + x[1:]*y[:-1] + 2*x[1:]*y[1:]) / 6))


def pair(left, right):
    knots = np.union1d(left.knots, right.knots)
    a = left(knots); b = right(knots)
    delta = Spline(knots, a-b)
    # Direct difference-of-squares product avoids subtracting near-equal risks.
    square_difference = product_integral(delta, Spline(knots, a+b))
    return delta, square_difference


def batch_segments(bank):
    """Vectorized piecewise-affine forward composition, no interval truncation."""
    bank = np.asarray(bank); owners = np.arange(len(bank))
    low = np.zeros(len(bank)); high = np.ones(len(bank)); slope = np.ones(len(bank)); offset = np.zeros(len(bank))
    for layer in range(bank.shape[1]):
        bias = bank[owners, layer]
        roots = np.divide(base.KNOTS[None, :] - offset[:, None] - bias[:, None], slope[:, None],
                          out=np.full((len(owners), 3), np.inf), where=slope[:, None] != 0)
        roots = np.sort(np.clip(roots, low[:, None], high[:, None]), axis=1)
        cuts = np.column_stack([low, roots, high]); ll = cuts[:, :-1].ravel(); hh = cuts[:, 1:].ravel()
        parent = np.repeat(np.arange(len(owners)), 4); keep = hh > ll
        low = ll[keep]; high = hh[keep]; parent = parent[keep]
        region = np.searchsorted(base.KNOTS, slope[parent] * ((low+high)/2) + offset[parent] + bias[parent], side='right')
        offset = base.SLOPES[region] * (offset[parent] + bias[parent]) + base.INTERCEPTS[region]
        slope = base.SLOPES[region] * slope[parent]; owners = owners[parent]
    return owners, low, high, slope, offset


def integrate_bank(delta, segments, count):
    owners, low, high, slope, offset = segments
    l0, l1 = delta.primitive(low); h0, h1 = delta.primitive(high)
    pieces = slope * (h1-l1) + offset * (h0-l0)
    return np.bincount(owners, weights=pieces, minlength=count)


def verify():
    rng = np.random.default_rng(772913); errors = []; segment_errors = []; symmetric_errors = []
    for d in [1, 2, 4, 6]:
        teachers = rng.uniform(-.12, .12, (16, d)); seg = batch_segments(teachers)
        assert np.allclose(np.bincount(seg[0], weights=seg[2]-seg[1], minlength=16), 1, atol=1e-14)
        for i, b in enumerate(teachers):
            k, s, c = compiler.segments(b); mask = seg[0] == i
            assert np.array_equal(seg[1][mask], k[:-1]) and np.array_equal(seg[2][mask], k[1:])
            assert np.array_equal(seg[3][mask], s) and np.array_equal(seg[4][mask], c)
            mid = (k[:-1]+k[1:])/2
            segment_errors.append(float(np.max(np.abs(s*mid+c-base.forward(mid, b)))))
        bank = rng.uniform(-.12, .12, (7, d)); left = from_bank(bank)
        right = from_bank(bank[:3]); delta, c = pair(left, right)
        direct = integrate_bank(delta, seg, 16)
        expected = np.array([product_integral(delta, from_bank(b[None])) for b in teachers])
        errors.append(float(np.max(np.abs(direct-expected))))
        back, cb = pair(right, left)
        symmetric_errors.append(abs(c+cb) + float(np.max(np.abs(integrate_bank(back, seg, 16)+direct))))
        zero, cz = pair(left, left); assert cz == 0 and np.array_equal(integrate_bank(zero, seg, 16), np.zeros(16))
        q = rng.uniform(0, 1, 4096)
        assert np.max(np.abs(left(q)-np.mean([base.forward(q, b) for b in bank], axis=0))) < 1e-11
        # Primitive constant/linear identities, independent of compiler.
    constant = Spline([0, 1], [1, 1]); linear = Spline([0, 1], [0, 1]); q = rng.uniform(0, 1, 100)
    a, b = constant.primitive(q); assert np.allclose(a, q) and np.allclose(b, q*q/2)
    a, b = linear.primitive(q); assert np.allclose(a, q*q/2) and np.allclose(b, q**3/3)
    assert max(errors+segment_errors+symmetric_errors) < 1e-10
    return dict(passed=True, scalar_batch_segment_cases=64, max_bank_integral_error=max(errors),
                max_segment_forward_error=max(segment_errors), max_antisymmetry_error=max(symmetric_errors),
                zero_pair_exact=True, primitive_identities=True)


if __name__ == '__main__':
    import json
    print(json.dumps(verify()))
