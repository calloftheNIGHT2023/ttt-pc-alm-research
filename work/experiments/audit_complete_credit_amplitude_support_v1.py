"""308 post-seal range audit, independently excluding interior guard roots.

The audit checks endpoint signs and the rational vertex of each quadratic;
it does not call the candidate's root-finding function for real-state audits.
It also checks exact tiling, scalar reference values and binary64 interval
representability. No LP, posterior or query target is evaluated.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
from functools import cmp_to_key
import gzip
import json
import math
from pathlib import Path
import random
import time
import traceback

from exact_quadratic_events_v1 import Poly, Root, compare, sign_at
from complete_credit_rational_reference_v1 import direct_step
from evaluate_complete_credit_mode_geometry_v1 import load_support, read, save, sha


def point(value):
    if isinstance(value, str):
        return F(value)
    a, b, c = value['polynomial']
    assert math.gcd(math.gcd(abs(a), abs(b)), abs(c)) == 1
    return Root((a, b, c), value['root_index'])


def has_interior_zero(poly, left, right):
    """Independent root exclusion via monotonicity and the vertex.

    On either side of the vertex a nonconstant quadratic is strictly monotone.
    A root is strictly inside such a subinterval iff endpoint signs differ.
    A zero exactly at an interior vertex is handled explicitly (a double root).
    """
    assert compare(left, right) < 0
    p = Poly(poly)
    c, b, a = p.c
    if not a:
        return c == 0 if not b else sign_at(p, left) * sign_at(p, right) < 0
    vertex = -b / (2 * a)
    lsign, rsign = sign_at(p, left), sign_at(p, right)
    if compare(left, vertex) < 0 and compare(vertex, right) < 0:
        vsign = sign_at(p, vertex)
        return vsign == 0 or lsign * vsign < 0 or rsign * vsign < 0
    return lsign * rsign < 0


def binary64_inside(left, right, witness):
    candidate = float(witness)
    if compare(F(candidate), left) <= 0:
        candidate = math.nextafter(candidate, math.inf)
    elif compare(F(candidate), right) >= 0:
        candidate = math.nextafter(candidate, -math.inf)
    if compare(left, F(candidate)) < 0 and compare(F(candidate), right) < 0:
        return candidate
    # A rounded interior rational and its adjacent float straddle this interval.
    # There is no other binary64 number between consecutive finite floats.
    return None


def evaluate(encoded, q, depth):
    if depth:
        return [evaluate(v, q, depth - 1) for v in encoded]
    return Poly(encoded).at(q)


def decoded_state(data):
    result = {k: [F(q) for q in data[k]] for k in ['b', 'x', 'v']}
    result.update({k: [[F(q) for q in row] for row in data[k]] for k in ['h', 'direction']})
    result.update({k: F(data[k]) for k in ['bound', 'trust', 'eps']})
    return result


def reference_check(cell, state, q):
    expected = direct_step(state, q)
    assert evaluate(cell['b'], q, 1) == expected['b']
    assert evaluate(cell['h'], q, 2) == expected['h']
    assert evaluate(cell['forward'], q, 1) == expected['forward']
    assert cell['codes'] == expected['codes'] and cell['mode'] == expected['mode']


def self_test():
    # Root solving is used ONLY as a cross-algorithm oracle in synthetic tests.
    from exact_quadratic_events_v1 import roots
    rng = random.Random(308719)
    checks = 0
    for _ in range(300):
        coefficients = [F(rng.randint(-8, 8), rng.randint(1, 9)) for _ in range(3)]
        if not any(coefficients):
            continue
        p = Poly(coefficients)
        rr = roots(p)
        for _ in range(12):
            a, b = sorted(rng.sample(range(-32, 33), 2))
            left, right = F(a, 8), F(b, 8)
            expected = any(compare(left, r) < 0 and compare(r, right) < 0 for r in rr)
            assert has_interior_zero(p, left, right) == expected, (p, left, right)
            checks += 1
    sqrt2 = roots(Poly((-2, 0, 1)))[1]
    assert not has_interior_zero(Poly((-2, 0, 1)), F(0), sqrt2)
    assert has_interior_zero(Poly((-2, 0, 1)), F(0), F(2))
    assert not has_interior_zero(Poly((1, 0, 1)), F(-2), F(2))
    assert has_interior_zero(Poly((0, 0, 1)), F(-1), F(1))
    assert not has_interior_zero(Poly((0, 0, 1)), F(0), F(1))
    a, b = F(1) + F(1, 2**80), F(1) + F(1, 2**79)
    assert binary64_inside(a, b, (a + b) / 2) is None
    assert binary64_inside(F(1), F(1) + F(1, 2**50), F(1) + F(1, 2**51)) is not None
    return dict(passed=True, rational_interval_cross_algorithm_checks=checks,
                boundary_double_and_no_root_cases=5, float_interval_cases=2,
                real_support_accessed=False)


def audit_payload(data):
    start = time.perf_counter()
    partition, state = data['partition'], decoded_state(data['state'])
    counts = Counter()
    nonrepresentable = []
    intervals = []
    for idx, cell in enumerate(partition['cells']):
        left, right, witness = point(cell['low']), point(cell['high']), F(cell['witness'])
        assert compare(left, witness) < 0 and compare(witness, right) < 0
        intervals.append((left, right))
        for _, coefficients, expected in cell['guards']:
            p = Poly(coefficients)
            assert p.c[1] or p.c[2]
            assert sign_at(p, witness) == expected != 0
            assert not has_interior_zero(p, left, right), (idx, coefficients)
            counts['uniform_guard_root_exclusions'] += 1
        reference_check(cell, state, witness)
        counts['independent_scalar_witness_checks'] += 1
        for encoded in cell['b']:
            p = Poly(encoded)
            assert p.c[2] == 0
            for end in [left, right]:
                assert sign_at(p + state['bound'], end) >= 0
                assert sign_at(state['bound'] - p, end) >= 0
            counts['uniform_bias_box_checks'] += 1
        for encoded in [v for row in cell['h'] for v in row] + cell['forward']:
            p = Poly(encoded)
            assert p.c[2] == 0
            for end in [left, right]:
                assert sign_at(p, end) >= 0 and sign_at(1 - p, end) >= 0
            counts['uniform_activity_and_output_box_checks'] += 1
        fl = binary64_inside(left, right, witness)
        if fl is None:
            nonrepresentable.append(idx)
        else:
            # This checks the exact model at a binary64-representable alpha;
            # it is not a replay of every intermediate floating operation.
            reference_check(cell, state, F(fl))
            counts['binary64_argument_exact_reference_checks'] += 1
    for left, right in partition['pending']:
        left, right = point(left), point(right)
        assert compare(left, right) < 0
        intervals.append((left, right))
    intervals.sort(key=cmp_to_key(lambda a, b: compare(a[0], b[0])))
    assert intervals and compare(intervals[0][0], F(-1)) == 0 and compare(intervals[-1][1], F(2)) == 0
    for a, b in zip(intervals[:-1], intervals[1:]):
        assert compare(a[1], b[0]) == 0
        counts['exact_tiling_adjacencies'] += 1
    endpoints = {end for pair in intervals for end in pair}
    seen = set()
    for boundary in partition['boundaries']:
        q = point(boundary['point'])
        assert q in endpoints and q not in seen
        seen.add(q)
        for _, coefficients, expected in boundary['guards']:
            assert sign_at(Poly(coefficients), q) == expected
            counts['boundary_guard_signs'] += 1
        if isinstance(q, F):
            reference_check(boundary, state, q)
            counts['independent_scalar_boundary_checks'] += 1
        else:
            counts['algebraic_boundaries_not_scalar_checked'] += 1
    if partition['coverage_complete']:
        assert not partition['pending'] and seen == endpoints
        assert len(partition['boundaries']) == len(partition['cells']) + 1
    assert partition['open_intervals_complete'] == (not partition['pending'])
    assert partition['polynomial_root_comparisons'] <= partition['comparison_limit']
    assert len(partition['cells']) <= partition['max_segments']
    return dict(passed=True, counts=dict(counts), cells_without_interior_binary64=nonrepresentable,
                coverage_complete=partition['coverage_complete'], seconds=time.perf_counter() - start)


def run(root, out, only_self_test):
    names = [Path(__file__).name, 'exact_quadratic_events_v1.py', 'complete_credit_rational_reference_v1.py',
             'evaluate_complete_credit_mode_geometry_v1.py']
    save(out / 'protocol.json', dict(source_sha256={n: sha(root / 'work/experiments' / n) for n in names},
         self_test_only=only_self_test, range_proof='endpoint signs and vertex, not candidate root solving',
         query_targets_accessed=False, geometry_evaluated=False))
    checks = self_test()
    save(out / 'self_test.json', checks)
    if only_self_test:
        save(out / 'summary.json', checks)
        print(checks, flush=True)
        return
    folder, summary, protocol, groups = load_support(root)
    counts, records = Counter(), []
    start = time.perf_counter()
    for index, (key, grouped) in enumerate(sorted(groups.items())):
        for family, row in grouped.items():
            with gzip.open(folder / row['file'], 'rt', encoding='utf-8') as stream:
                data = json.load(stream)
            result = audit_payload(data)
            counts.update(result['counts'])
            records.append(dict(seed=key[0], location=list(key[1:]), family=family,
                                source_sha256=row['sha256'], **result))
        if (index + 1) % 8 == 0:
            print(dict(audited_states=index + 1, directions=len(records), guard_exclusions=counts['uniform_guard_root_exclusions']), flush=True)
    assert len(records) == 393
    save(out / 'rows.json', records)
    result = dict(passed=True, states=131, directions=393, counts=dict(counts),
        nonrepresentable_cells=sum(len(r['cells_without_interior_binary64']) for r in records),
        support_summary_sha256=sha(folder / 'summary.json'), geometry_evaluated=False, query_targets_accessed=False,
        seconds=time.perf_counter() - start,
        outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    save(out / 'summary.json', result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, args.out, args.self_test)
    except Exception:
        save(args.out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
