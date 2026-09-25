"""400 exact independent checks; deliberately gated behind the 397 run chain."""
from fractions import Fraction as F
from itertools import permutations, product
from math import factorial
from pathlib import Path
import hashlib
import json
import random
import time
import traceback

import certified_partial_readout_v1 as core

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/certified_partial_readout/math_preflight_v1'
DESIGN = 'outputs/ttt-pc-alm-research/400_certified_partial_readout_v1.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def det_permutation(a):
    answer = F(0)
    for order in permutations(range(len(a))):
        term = F((-1)**sum(order[i] > order[j] for i in range(len(a)) for j in range(i + 1, len(a))))
        for i, j in enumerate(order):
            term *= a[i][j]
        answer += term
    return answer


def box(d, bound=F(1)):
    a, rhs = [], []
    for j in range(d):
        row = tuple(F(int(k == j)) for k in range(d))
        a += [row, tuple(-t for t in row)]
        rhs += [bound, bound]
    return a, rhs


def cube_faces(vertices):
    """All standard simplex facets of each cubical boundary face."""
    lookup = {v:i for i, v in enumerate(vertices)}
    d = len(vertices[0])
    faces = []
    for fixed in range(d):
        for side in [-1, 1]:
            for order in permutations([j for j in range(d) if j != fixed]):
                p = [-1] * d
                p[fixed] = side
                face = [lookup[tuple(p)]]
                for j in order:
                    p[j] = 1
                    face.append(lookup[tuple(p)])
                faces.append(tuple(face))
    return faces


def rejected(fn):
    try:
        fn()
    except (AssertionError, ValueError, ZeroDivisionError, OverflowError):
        return
    raise AssertionError('Invalid input was accepted')


def risk(values, weights, prediction):
    return sum((w * (v - prediction)**2 for v, w in zip(values, weights)), F(0)) / sum(weights)


def main():
    started = time.perf_counter()
    sources = [Path(__file__), Path(core.__file__), ROOT/DESIGN,
               Path(__file__).with_name('probe_simplex_readout_intervals.py')]
    hashes = {p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    save(OUT/'protocol.json', dict(source_sha256=hashes, random_seed=400731,
        source_scope='400 exact primitives and tests only; 397 unchanged',
        research_query_targets_accessed=False, tests='exact deterministic and rational randomized',
        observations='synthetic mathematical cases only; no research task labels'))
    rng = random.Random(400731)
    counters = dict(determinants=0, cube_dimensions=0, standard_simplices=0,
                    discrete_posterior_cases=0, projections=0, strict_projections=0,
                    bisections=0, invalid_inputs=0)
    for d in range(1, 5):
        for _ in range(24):
            matrix = [[F(rng.randrange(-9, 10), rng.randrange(1, 8)) for _ in range(d)] for _ in range(d)]
            assert core.determinant(matrix) == det_permutation(matrix)
            counters['determinants'] += 1
    cones = []
    lam = F(1048575, 1048576)
    for d in range(2, 5):
        vertices = tuple(product([-1, 1], repeat=d))
        a, rhs = box(d)
        result = core.certified_cones(vertices, cube_faces(vertices), (F(0),) * d, a, rhs)
        # One retained simplex per hyperplane, not an unjustified full hull.
        expected = F(2**d, factorial(d - 1)) * lam**d
        assert result['mass'] == expected
        assert len(result['simplices']) == 2*d
        assert result['rejections']['duplicate_plane'] == 2*d*(factorial(d - 1)-1)
        assert result['rejections']['degenerate_or_unsupported'] == 0
        cones.append(dict(dimension=d, exact_mass=str(result['mass']), whole_box_mass=str(2**d),
                          retained=len(result['simplices']), rejections=result['rejections']))
        counters['cube_dimensions'] += 1
        simplex = ((F(0),)*d,) + tuple(tuple(F(int(j == i)) for j in range(d)) for i in range(d))
        rows = [tuple(-F(int(j == i)) for j in range(d)) for i in range(d)] + [(F(1),)*d]
        bounds = [F(0)]*d + [F(1)]
        anchor = (F(1, d + 1),)*d
        faces = [tuple(j for j in range(d + 1) if j != i) for i in range(d + 1)]
        found = core.certified_cones(simplex, faces, anchor, rows, bounds)
        assert found['mass'] == lam**d / factorial(d)
        assert len(found['simplices']) == d + 1
        counters['standard_simplices'] += 1
    # A diagonal through a square is not a supporting face; duplicates add no mass.
    square = tuple(product([-1, 1], repeat=2))
    a, rhs = box(2)
    bad = core.certified_cones(square, cube_faces(square) + [(0, 3), (0, 1), (1, 0)], (0, 0), a, rhs)
    assert bad['mass'] == 4*lam**2
    assert bad['rejections'] == dict(degenerate_or_unsupported=1, duplicate_plane=2)
    inside, contraction = core.contract_cloud([(-2, -2), (2, 2)], (0, 0), a, rhs)
    assert contraction == lam/2 and all(core.feasible(v, a, rhs) for v in inside)
    assert all(max(map(abs, v)) < 1 for v in inside)
    # Exact affine two-layer average; derived directly as 1/4 + 4*b0 + 2*b1.
    triangle = [(0, 0), (F(1, 16), 0), (0, F(1, 16))]
    affine = core.moment_bounds(triangle, F(1, 16), 0)
    assert affine['mass'] == F(1, 512)
    assert affine['lower'] == affine['upper'] == F(3, 4096)
    assert affine['affine_leaves'] == 1
    # A peak inside an interval disproves bounding by vertex outputs alone.
    interval = [(F(0),), (F(1),)]
    assert core.forward(0, interval[0]) == core.forward(0, interval[1]) == 0
    assert core.forward(0, [F(1, 2)]) == 1
    crossing = core.moment_bounds(interval, 0, 0)
    refined = core.moment_bounds(interval, 0, 1)
    assert crossing['lower'] == 0 and crossing['upper'] == 1
    assert refined['lower'] == refined['upper'] == F(1, 2) and refined['affine_leaves'] == 2
    # Deterministic bisection conserves exact mass and nested moment bounds.
    for d in range(1, 5):
        simplex = ((F(-1, 10),)*d,) + tuple(tuple(F(7, 10) if j == i else F(-1, 10) for j in range(d)) for i in range(d))
        children = core.bisect_simplex(simplex)
        assert all(core.simplex_volume(s) == core.simplex_volume(simplex)/2 for s in children)
        previous = core.moment_bounds(simplex, F(1, 5), 0)
        for splits in range(1, 4):
            current = core.moment_bounds(simplex, F(1, 5), splits)
            assert previous['lower'] <= current['lower'] <= current['upper'] <= previous['upper']
            assert current['mass'] == previous['mass']
            previous = current
            counters['bisections'] += 1
    assert core.posterior_interval(0, 0, 0, 1) == (0, 1)
    assert core.posterior_interval(2, F(3, 2), F(3, 2), 2) == (F(3, 4), F(3, 4))
    for _ in range(256):
        n = rng.randrange(2, 10)
        values = [F(rng.randrange(33), 32) for _ in range(n)]
        weights = [F(rng.randrange(1, 25), 17) for _ in range(n)]
        cut = rng.randrange(n + 1)
        total, mass = sum(weights), sum(weights[:cut], F(0))
        moment = sum((w*v for w, v in zip(weights[:cut], values[:cut])), F(0))
        slack = F(rng.randrange(5), 7)
        lower_moment, upper_moment = max(F(0), moment-slack), min(mass, moment+slack)
        upper = total+F(rng.randrange(5), 11)
        bounds = core.posterior_interval(mass, lower_moment, upper_moment, upper)
        mean = sum((w*v for w, v in zip(weights, values)), F(0))/total
        assert bounds[0] <= mean <= bounds[1]
        assert bounds[1]-bounds[0] == 1-mass/upper+(upper_moment-lower_moment)/upper
        midpoint = sum(bounds)/2
        assert (midpoint-mean)**2 <= ((bounds[1]-bounds[0])/2)**2
        exact_bounds = core.posterior_interval(mass, moment, moment, total)
        intersection = core.intersect_intervals(bounds, exact_bounds)
        assert intersection == exact_bounds
        for baseline in [F(-1, 3), F(rng.randrange(33), 32), F(4, 3)]:
            projected = core.project(baseline, bounds)
            gain = risk(values, weights, baseline)-risk(values, weights, projected)
            assert gain == (baseline-mean)**2-(projected-mean)**2
            assert gain >= (baseline-projected)**2 >= 0
            if baseline != projected:
                assert gain > 0
                counters['strict_projections'] += 1
            counters['projections'] += 1
        counters['discrete_posterior_cases'] += 1
    # Positive conditional benefit coexists with a particular teacher worsening.
    projected = core.project(0, (F(3, 4), F(3, 4)))
    gain = risk([F(0), F(1)], [F(1), F(3)], 0)-risk([F(0), F(1)], [F(1), F(3)], projected)
    assert gain == F(9, 16) and (projected-F(0))**2 > (F(0)-F(0))**2
    # Verify a correlated parallelogram volume independently by a shoelace area.
    a = [(1, 1), (-1, -1), (1, -1), (-1, 1)]
    rhs = [2, 0, 1, 1]
    bound = core.row_parallelotope_upper(a, rhs, [0, 2])
    polygon = [(F(-1, 2), F(1, 2)), (F(1, 2), F(-1, 2)), (F(3, 2), F(1, 2)), (F(1, 2), F(3, 2))]
    area = abs(sum(polygon[i][0]*polygon[(i+1)%4][1]-polygon[(i+1)%4][0]*polygon[i][1] for i in range(4)))/2
    assert bound['volume'] == area == 2 and not bound['empty']
    assert core.row_parallelotope_upper(a+[(2, 2)], rhs+[2], [0, 2])['volume'] == 1
    assert core.row_parallelotope_upper(a, rhs, [0, 1]) is None
    assert core.row_parallelotope_upper(a, [-1, 0, 1, 1], [0, 2])['empty']
    invalid = [lambda:core.posterior_interval(1, 0, 2, 1),
               lambda:core.posterior_interval(2, 0, 1, 1),
               lambda:core.posterior_interval(0, 0, 0, 0),
               lambda:core.intersect_intervals((0, F(1, 4)), (F(1, 2), 1)),
               lambda:core.project(0, (1, 0)),
               lambda:core.posterior_interval(1, 0, 1, float('nan')),
               lambda:core.posterior_interval(1, 0, 1, float('inf')),
               lambda:core.contract_cloud(square, (1, 1), *box(2))]
    for fn in invalid:
        rejected(fn)
        counters['invalid_inputs'] += 1
    evidence = dict(cube_cones=cones, affine_moment=str(affine['lower']),
                    crossing_peak=dict(unrefined=[str(crossing[k]) for k in ['lower', 'upper']],
                                       refined=[str(refined[k]) for k in ['lower', 'upper']]),
                    strict_conditional_gain=str(gain), realized_teacher_counterexample=True,
                    parallelogram_upper=str(bound['volume']))
    save(OUT/'evidence.json', evidence)
    assert all(sha(ROOT/p) == digest for p, digest in hashes.items())
    summary = dict(passed=True, counters=counters, seconds=time.perf_counter()-started,
                   source_sha256=hashes, outputs_sha256={f:sha(OUT/f) for f in ['protocol.json', 'evidence.json']},
                   research_query_targets_accessed=False, online_search_integrated=False,
                   empirical_pc_alm_advantage_established=False, exact_fraction_arithmetic=True)
    save(OUT/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    gate = ROOT/'results/prefix_deadline_pilot/report_v1/qa_numeric.json'
    assert gate.is_file() and json.loads(gate.read_text(encoding='utf-8'))['passed'], 'Wait for original 397 chain; do not compete with timed calls.'
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except Exception:
        save(OUT/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
