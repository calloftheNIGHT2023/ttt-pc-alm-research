"""334 exact positive common-scale integer implementation of 321.

No approximation, global derivative, LP or query target. Arbitrary-precision
integers replace Fraction normalization within the dynamic program. All credit
channels share this implementation. Output rational strings preserve 321.
"""
from fractions import Fraction
from itertools import product
import heapq
import math
import time

S = (0, 2, -2, 0)
C = (0, 0, 2, 0)


def ratios(values):
    result = []
    for value in values:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Only finite binary64 inputs are supported')
        result.append(value.as_integer_ratio())
    return result


class IntegerProblem:
    def __init__(self, x, v, credit):
        self.n = len(x); self.d = len(credit)
        assert self.n > 0 and self.d > 0 and len(v) == self.n
        assert all(len(row) == self.n for row in credit)
        coords = ratios([*x, *v, .12, .001, .5])
        cr = ratios([a for row in credit for a in row])
        self.X = max(den for _, den in coords)
        self.A = max(den for _, den in cr)
        assert self.X & (self.X-1) == self.A & (self.A-1) == 0
        q = [num*(self.X//den) for num, den in coords]
        self.x, self.v = q[:self.n], q[self.n:2*self.n]
        self.B, self.E, half = q[-3:]
        self.zlo = (-self.B, 0, half, self.X)
        self.zhi = (0, half, self.X, self.X+self.B)
        aa = [num*(self.A//den) for num, den in cr]
        self.aa = [aa[j*self.n:(j+1)*self.n] for j in range(self.d)]
        self.denominator = self.X*self.A

    def layer(self, low, high, previous, credit, row):
        zl = [self.zlo[r] for r in row]; zh = [self.zhi[r] for r in row]
        left = max([-self.B] + [z-h for z, h in zip(zl, high)])
        right = min([self.B] + [z-l for z, l in zip(zh, low)])
        if left > right:
            return None
        sa = [S[r]*a for r, a in zip(row, credit)]
        coef = [p-a for p, a in zip(previous, sa)]
        knots = {left, right}
        for c, z0, z1, l, h in zip(coef, zl, zh, low, high):
            knots.add(max(left, min(right, z0-l if c >= 0 else z1-h)))
        values = []
        for bias in knots:
            hh = [max(l, z-bias) if c >= 0 else min(h, w-bias)
                  for c, l, h, z, w in zip(coef, low, high, zl, zh)]
            values.append(sum(c*h-s*bias-a*C[r]*self.X
                              for c, h, s, a, r in zip(coef, hh, sa, credit, row)))
        return min(values)

    def terminal(self, output_mask):
        lo = [max(0, y-self.E) for y in self.v]
        hi = [min(self.X if output_mask & (1 << i) else 0, y+self.E)
              for i, y in enumerate(self.v)]
        if any(l > h for l, h in zip(lo, hi)):
            return None
        return sum(min(a*l, a*h) for a, l, h in zip(self.aa[-1], lo, hi))

    def fixed(self, pattern):
        total = 0
        for j, row in enumerate(pattern):
            low = self.x if j == 0 else [0]*self.n
            high = low if j == 0 else [self.X if r in (1, 2) else 0 for r in pattern[j-1]]
            previous = [0]*self.n if j == 0 else self.aa[j-1]
            value = self.layer(low, high, previous, self.aa[j], row)
            if value is None:
                return None
            total += value
        last = self.terminal(mask(pattern[-1]))
        return None if last is None else total+last

    def string(self, integer):
        return str(Fraction(integer, self.denominator))


def mask(row):
    return sum(1 << i for i, r in enumerate(row) if r in (1, 2))


def pair_best(prefixes, rows, k):
    heap = [(value+rows[0][0], path+rows[0][1], i, 0) for i, (value, path) in enumerate(prefixes)]
    heapq.heapify(heap); out = []
    while heap and len(out) < k:
        value, path, i, j = heapq.heappop(heap); out.append((value, path))
        if j+1 < len(rows):
            heapq.heappush(heap, (prefixes[i][0]+rows[j+1][0], prefixes[i][1]+rows[j+1][1], i, j+1))
    return out


def integer_shells(problem, original, k):
    assert k > 0 and len(original) == problem.d and all(len(row) == problem.n for row in original)
    assert all(int(r) == r and 0 <= r <= 3 for row in original for r in row)
    n, d = problem.n, problem.d
    rows = [(row, mask(row)) for row in product(range(4), repeat=n)]
    terminal = {m: problem.terminal(m) for _, m in rows}
    dp = {(0, -1): [(0, ())]}
    count = empty = retained = peak_entries = pair_outputs = 0
    table_seconds = selection_seconds = 0.
    for j in range(d):
        start = time.perf_counter(); transitions = {}
        for incoming in sorted({m for _, m in dp}):
            low = problem.x if j == 0 else [0]*n
            high = low if j == 0 else [problem.X if incoming & (1 << i) else 0 for i in range(n)]
            previous = [0]*n if j == 0 else problem.aa[j-1]; grouped = {}
            for row, outgoing in rows:
                last = terminal[outgoing] if j == d-1 else 0
                if last is None:
                    empty += 1; continue
                value = problem.layer(low, high, previous, problem.aa[j], row); count += 1
                if value is None:
                    empty += 1; continue
                delta = sum(a != int(b) for a, b in zip(row, original[j]))
                grouped.setdefault((delta, outgoing), []).append((value+last, row))
            transitions[incoming] = {key: sorted(pool)[:k] for key, pool in grouped.items()}
            retained += sum(len(pool) for pool in transitions[incoming].values())
        table_seconds += time.perf_counter()-start; start = time.perf_counter(); nextdp = {}
        for (distance, incoming), prefixes in dp.items():
            for (delta, outgoing), candidates in transitions[incoming].items():
                merged = pair_best(prefixes, candidates, k); pair_outputs += len(merged)
                key = (distance+delta, outgoing)
                nextdp[key] = sorted(nextdp.get(key, [])+merged)[:k]
        dp = nextdp; peak_entries = max(peak_entries, sum(map(len, dp.values())))
        selection_seconds += time.perf_counter()-start
    result = {}
    for (distance, outgoing), pool in dp.items():
        result.setdefault(distance, []).extend(pool)
    result = {m: sorted(pool)[:k] for m, pool in result.items()}
    return result, dict(layer_rows_evaluated=count, empty_transitions=empty, retained_transition_rows=retained,
        pair_outputs=pair_outputs, max_retained_dp_entries=peak_entries, table_seconds=table_seconds,
        selection_seconds=selection_seconds, state_scope='Exact integer DP counts, not Python/native peak memory')


def propose(x, v, credit, original, k=8):
    begin = time.perf_counter(); problem = IntegerProblem(x, v, credit)
    dp, meta = integer_shells(problem, original, k); current = problem.fixed(original)
    admissible = sorted(m for m, pool in dp.items() if m >= 1 and pool[0][0] <= 0)
    minimum = admissible[0] if admissible else None; proposals = []
    if minimum is not None:
        for m in range(minimum, min(minimum+2, len(x)*len(original))+1):
            for rank, (value, path) in enumerate(dp.get(m, [])):
                if value <= 0:
                    proposals.append(dict(mode=bytes(path).hex(), hamming=m, rank=rank, lower=problem.string(value)))
    meta.update(total_seconds=time.perf_counter()-begin,
                coordinate_scale_bits=problem.X.bit_length(), credit_scale_bits=problem.A.bit_length(),
                exact_common_scale=True)
    return dict(current_lower=None if current is None else problem.string(current),
        current_structurally_infeasible=current is None, current_certified_infeasible=current is None or current > 0,
        minimum_nonexcluded_hamming=minimum, necessary_hamming_lower_bound=minimum if current is None or current > 0 else None,
        shell_minima={str(m): problem.string(pool[0][0]) for m, pool in sorted(dp.items())}, proposals=proposals, meta=meta)
