"""308: exact policy cells of one complete [0,1]-tent ALM primal sweep.

This is an event-enumeration prototype, not an efficient online algorithm.
Inputs are support observations and a local state, never query answers.
Rational arithmetic defines the mathematical step; floating agreement is a
separate test. No BP call or BP-derived initialization is used here.
"""
from functools import cmp_to_key
from fractions import Fraction as F
import time

from exact_quadratic_events_v1 import (
    Budget, Poly, WorkLimit, between, compare, roots, sign, sign_at,
)

KNOTS = (F(0), F(1, 2), F(1))
SLOPES = (0, 2, -2, 0)
INTERCEPTS = (0, 0, 2, 0)
ALPHA = Poly((0, 1))


class SampleOnEvent(Exception):
    def __init__(self, point, label):
        self.point, self.label = point, label


class Trace:
    """Record all comparisons needed to preserve this execution policy."""
    def __init__(self, point, budget, strict=True):
        self.point, self.budget, self.strict = point, budget, strict
        self.guards = {}
        self.decisions = []

    def test(self, value, label):
        p = Poly(value)
        result = sign_at(p, self.point, self.budget)
        if p.c[1] or p.c[2]:
            if self.strict and result == 0:
                raise SampleOnEvent(self.point, label)
            # Proportional guards have identical roots. Canonicalize their
            # sign too, so duplicate tests do not inflate retained state.
            first = next(c for c in p.c if c)
            key = tuple(c / first for c in p.c)
            self.guards.setdefault(key, (label, Poly(key), result * sign(first)))
        return result

    def branch(self, value, label):
        k = sum(self.test(value - knot, (label, 'knot', i)) >= 0
                for i, knot in enumerate(KNOTS))
        self.decisions.append((label, 'branch', k))
        return k

    def tent(self, value, label):
        value = Poly(value)
        k = self.branch(value, label)
        return SLOPES[k] * value + INTERCEPTS[k]

    def clip(self, value, low, high, label):
        value, low, high = Poly(value), Poly(low), Poly(high)
        assert self.test(high - low, (label, 'ordered')) >= 0
        ls = self.test(value - low, (label, 'low'))
        hs = self.test(value - high, (label, 'high'))
        status = 'low' if ls < 0 else 'high' if hs > 0 else 'free'
        self.decisions.append((label, 'clip', status))
        return low if ls < 0 else high if hs > 0 else value

    def minimum(self, candidates, label):
        """Candidates are (value, quadratic cost, original index)."""
        winner = 0
        for k in range(1, len(candidates)):
            if self.test(candidates[k][1] - candidates[winner][1],
                         (label, 'cost', candidates[k][2], candidates[winner][2])) < 0:
                winner = k
        value, _, original = candidates[winner]
        self.decisions.append((label, 'winner', original))
        return value.affine()


def rational_state(b, h, direction, x, v, *, bound=.12, trust=.01, eps=.001):
    state = dict(b=[F(q) for q in b], h=[[F(q) for q in row] for row in h],
                 direction=[[F(q) for q in row] for row in direction],
                 x=[F(q) for q in x], v=[F(q) for q in v],
                 bound=F(bound), trust=F(trust), eps=F(eps))
    d, n = len(b), len(x)
    assert d > 0 and n > 0 and len(h) == len(direction) == d and len(v) == n
    assert all(len(row) == n for row in state['h'] + state['direction'])
    assert state['trust'] > 0 and state['bound'] > 0 and state['eps'] >= 0
    assert all(abs(q) <= state['bound'] for q in state['b'])
    assert all(0 <= q <= 1 for row in state['h'] for q in row)
    assert all(0 <= q <= 1 for q in state['x'] + state['v'])
    return state


def bias_block(previous, target, old, trace, label, bound, trust):
    """Exact equivalent of streaming_branch_projection.bias_solve at rho=inf.

    Keep knot-major stable ordering and zero-width intervals. In particular,
    an event at a box endpoint must not be counted as an interior transition.
    """
    n = len(previous)
    aa, bb, cc = F(0), Poly(0), Poly(0)
    for i, (prev, goal) in enumerate(zip(previous, target)):
        k = trace.branch(prev - bound, (label, 'initial', i))
        s, c = SLOPES[k], SLOPES[k] * prev + INTERCEPTS[k]
        aa += s * s
        bb += s * (goal - c)
        cc += (c - goal).square()
    events = []
    for ki, knot in enumerate(KNOTS):
        sb, sa = SLOPES[ki], SLOPES[ki + 1]
        for i, (prev, goal) in enumerate(zip(previous, target)):
            event = knot - prev
            valid_low = trace.test(event + bound, (label, 'valid_low', ki, i)) > 0
            valid_high = trace.test(event - bound, (label, 'valid_high', ki, i)) < 0
            position = trace.clip(event, -bound, bound, (label, 'event', ki, i))
            cb, ca = sb * prev + INTERCEPTS[ki], sa * prev + INTERCEPTS[ki + 1]
            delta = (F(sa * sa - sb * sb), sa * (goal - ca) - sb * (goal - cb),
                     (ca - goal).square() - (cb - goal).square())
            if not (valid_low and valid_high):
                delta = (F(0), Poly(0), Poly(0))
            events.append((position, ki * n + i, delta))
    events.sort(key=cmp_to_key(lambda a, b: trace.test(
        a[0] - b[0], (label, 'sort', a[1], b[1]))))
    trace.decisions.append((label, 'order', tuple(e[1] for e in events)))
    low = Poly(-bound)
    candidates = []
    for k in range(len(events) + 1):
        high = events[k][0] if k < len(events) else Poly(bound)
        assert aa >= 0
        a, b, c = aa / n, bb / n, cc / n
        raw = (b + trust * old) / (a + trust)
        candidate = trace.clip(raw, low, high, (label, 'candidate', k))
        cost = a * candidate.square() - 2 * b * candidate + c + trust * (candidate - old).square()
        candidates.append((candidate, cost, k))
        if k < len(events):
            da, db, dc = events[k][2]
            aa, bb, cc = aa + da, bb + db, cc + dc
            low = high
    return trace.minimum(candidates, label)


def sweep(state, point, budget=None, *, strict=True):
    """One full reverse-activity + shared-bias step, then true forward codes."""
    budget = budget if budget is not None else Budget()
    tr = Trace(point, budget, strict)
    b0, h0, direction = state['b'], state['h'], state['direction']
    x, v, bound, trust, eps = (state[k] for k in ('x', 'v', 'bound', 'trust', 'eps'))
    d, n = len(b0), len(x)
    h = [[Poly(q) for q in row] for row in h0]
    for j in reversed(range(d)):
        for i in range(n):
            label = ('activity', j, i)
            prev = x[i] if j == 0 else h0[j - 1][i]
            a = tr.tent(Poly(prev + b0[j]), (label, 'previous')) - ALPHA * direction[j][i]
            before = h0[j][i]
            if j == d - 1:
                h[j][i] = tr.clip((a + trust * before) / (1 + trust),
                                  max(F(0), v[i] - eps), min(F(1), v[i] + eps), label)
                continue
            target = h[j + 1][i] + ALPHA * direction[j + 1][i]
            nb = b0[j + 1]
            candidates = []
            for k, s in enumerate(SLOPES):
                low = F(0) if k == 0 else max(F(0), KNOTS[k - 1] - nb)
                high = F(1) if k == 3 else min(F(1), KNOTS[k] - nb)
                if low > high:
                    continue
                offset = s * nb + INTERCEPTS[k]
                raw = (a + s * (target - offset) + trust * before) / (1 + s * s + trust)
                candidate = tr.clip(raw, low, high, (label, 'candidate', k))
                # g(candidate+nb) equals this affine expression throughout
                # the candidate's closed legal interval, including knots.
                cost = (candidate - a).square() + (s * candidate + offset - target).square()
                cost += trust * (candidate - before).square()
                candidates.append((candidate, cost, k))
            h[j][i] = tr.minimum(candidates, label)
    b = []
    for j in range(d):
        previous = [Poly(q) for q in x] if j == 0 else h[j - 1]
        target = [h[j][i] + ALPHA * direction[j][i] for i in range(n)]
        b.append(bias_block(previous, target, b0[j], tr, ('bias', j), bound, trust))
    forward = [Poly(q) for q in x]
    codes = []
    for j in range(d):
        row = []
        for i in range(n):
            z = forward[i] + b[j]
            k = tr.branch(z, ('full_forward', j, i))
            row.append(k)
            forward[i] = (SLOPES[k] * z + INTERCEPTS[k]).affine()
        codes.append(row)
    return dict(b=b, h=h, forward=forward, codes=codes,
                mode=''.join(str(k) for row in codes for k in row),
                guards=list(tr.guards.values()), decisions=tr.decisions)


def partition(oracle, *, low=F(-1), high=F(2), max_segments=4096,
              comparison_limit=1000000):
    """Certify open policy cells and evaluate every retained boundary policy.

    Each interval is either a certified cell or explicitly pending. Strict
    tracing at a rational event splits that interval before trying again.
    No floating ordering, finite-difference detection, or dense-grid fallback.
    """
    budget = Budget(comparison_limit)
    low, high = F(low), F(high)
    assert low < high and max_segments > 0
    pending, cells, cache, split_events = [(low, high)], [], {}, []
    current = None
    boundary_values = []
    started = time.perf_counter()
    limit_reason = None
    try:
        while pending:
            if len(cells) >= max_segments:
                raise WorkLimit('retained segment cap')
            current = pending.pop()
            left, right = current
            witness = between(left, right, budget)
            try:
                result = oracle(witness, budget, True)
            except SampleOnEvent as event:
                assert event.point == witness
                split_events.append(dict(point=witness, label=event.label))
                pending.extend([(witness, right), (left, witness)])
                current = None
                continue
            cell_low, cell_high = left, right
            for _, guard, _ in result['guards']:
                for root in roots(guard, cache, budget):
                    side = compare(root, witness, budget)
                    assert side != 0, 'Strict tracing missed a guard zero'
                    if side < 0 and compare(root, cell_low, budget) > 0:
                        cell_low = root
                    elif side > 0 and compare(root, cell_high, budget) < 0:
                        cell_high = root
            # Do all comparisons before mutating coverage bookkeeping.
            left_extra = compare(left, cell_low, budget) < 0
            right_extra = compare(cell_high, right, budget) < 0
            cells.append(dict(low=cell_low, high=cell_high, witness=witness, **result))
            if right_extra:
                pending.append((cell_high, right))
            if left_extra:
                pending.append((left, cell_low))
            current = None
        cells.sort(key=cmp_to_key(lambda a, b: compare(a['low'], b['low'], budget)))
        assert cells and compare(cells[0]['low'], low, budget) == 0
        assert compare(cells[-1]['high'], high, budget) == 0
        for a, b in zip(cells[:-1], cells[1:]):
            assert compare(a['high'], b['low'], budget) == 0
        boundaries = [cells[0]['low']] + [c['high'] for c in cells]
        for point in boundaries:
            boundary_values.append(dict(point=point, **oracle(point, budget, False)))
    except WorkLimit as exc:
        limit_reason = str(exc)
        if current is not None:
            pending.append(current)
    return dict(cells=cells, pending=pending, boundaries=boundary_values,
                split_events=split_events, open_intervals_complete=not pending,
                coverage_complete=limit_reason is None, limit_reason=limit_reason,
                polynomial_root_comparisons=budget.comparisons,
                comparison_limit=comparison_limit, max_segments=max_segments,
                root_cache_entries=len(cache), seconds=time.perf_counter() - started)


def sweep_partition(state, **options):
    return partition(lambda point, budget, strict: sweep(state, point, budget, strict=strict), **options)
