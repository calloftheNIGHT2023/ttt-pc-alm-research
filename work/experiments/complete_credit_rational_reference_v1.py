"""Independent scalar Fraction reference for the 308 complete primal sweep.

No symbolic trace, polynomial library, or cumulative event-cost recurrence.
The bias objective is rebuilt directly on every interval between breakpoints.
"""
from fractions import Fraction as F


def tent(z):
    return max(F(0), min(2 * z, 2 - 2 * z))


def clipped(q, lo, hi):
    return min(hi, max(lo, q))


def direct_step(state, alpha):
    alpha = F(alpha)
    b0, h0, u = state['b'], state['h'], state['direction']
    x, v = state['x'], state['v']
    tau, bound, eps = state['trust'], state['bound'], state['eps']
    d, n = len(b0), len(x)
    h = [list(row) for row in h0]
    activity_ties, bias_ties = [], []
    for j in range(d - 1, -1, -1):
        for i in range(n):
            previous = x[i] if j == 0 else h0[j - 1][i]
            a = tent(previous + b0[j]) - alpha * u[j][i]
            before = h0[j][i]
            if j == d - 1:
                h[j][i] = clipped((a + tau * before) / (1 + tau),
                                  max(F(0), v[i] - eps), min(F(1), v[i] + eps))
                continue
            target = h[j + 1][i] + alpha * u[j + 1][i]
            candidates = []
            intervals = [(None, F(0), 0, 0), (F(0), F(1, 2), 2, 0),
                         (F(1, 2), F(1), -2, 2), (F(1), None, 0, 0)]
            for ki, (lower, upper, slope, intercept) in enumerate(intervals):
                lo = F(0) if lower is None else max(F(0), lower - b0[j + 1])
                hi = F(1) if upper is None else min(F(1), upper - b0[j + 1])
                if lo > hi:
                    continue
                c = slope * b0[j + 1] + intercept
                value = clipped((a + slope * (target - c) + tau * before) /
                                (1 + slope * slope + tau), lo, hi)
                cost = (value - a)**2 + (tent(value + b0[j + 1]) - target)**2
                cost += tau * (value - before)**2
                candidates.append((cost, value, ki))
            best_cost = min(row[0] for row in candidates)
            winners = [row for row in candidates if row[0] == best_cost]
            h[j][i] = winners[0][1]
            if len({row[1] for row in winners}) > 1:
                activity_ties.append((j, i, [(row[2], row[1]) for row in winners]))
    b = []
    for j in range(d):
        previous = x if j == 0 else h[j - 1]
        target = [h[j][i] + alpha * u[j][i] for i in range(n)]
        points = {-bound, bound}
        for p in previous:
            for knot in [F(0), F(1, 2), F(1)]:
                if -bound < knot - p < bound:
                    points.add(knot - p)
        points = sorted(points)
        candidates = []
        for lo, hi in zip(points[:-1], points[1:]):
            midpoint = (lo + hi) / 2
            slopes, intercepts = [], []
            for p in previous:
                z = p + midpoint
                if z < 0 or z >= 1:
                    slope, intercept = 0, 0
                elif z < F(1, 2):
                    slope, intercept = 2, 0
                else:
                    slope, intercept = -2, 2
                slopes.append(slope)
                intercepts.append(intercept)
            numerator = sum(s * (t - s * p - c)
                            for s, t, p, c in zip(slopes, target, previous, intercepts)) / n
            denominator = sum(F(s * s) for s in slopes) / n
            value = clipped((numerator + tau * b0[j]) / (denominator + tau), lo, hi)
            cost = sum((tent(p + value) - t)**2 for p, t in zip(previous, target)) / n
            cost += tau * (value - b0[j])**2
            candidates.append((cost, value))
        best_cost = min(row[0] for row in candidates)
        winners = [row[1] for row in candidates if row[0] == best_cost]
        b.append(winners[0])
        if len(set(winners)) > 1:
            bias_ties.append((j, winners))
    forward, codes = list(x), []
    for bias in b:
        row = []
        for i in range(n):
            z = forward[i] + bias
            row.append(sum(z >= k for k in [F(0), F(1, 2), F(1)]))
            forward[i] = tent(z)
        codes.append(row)
    return dict(b=b, h=h, forward=forward, codes=codes,
                mode=''.join(str(k) for row in codes for k in row),
                activity_ties=activity_ties, bias_ties=bias_ties)
