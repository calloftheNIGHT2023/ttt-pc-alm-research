"""308 event arithmetic, full-step exact consistency and floating checks.

Each run writes an exclusive attempt directory and hashes its source files.
Synthetic fixtures validate the implementation, not ML performance.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time
import traceback

from exact_quadratic_events_v1 import (
    Budget, Poly, Root, between, compare, encode, roots, sign_at, sorted_points,
)
from complete_credit_amplitude_events_v2 import Trace, partition, rational_state, sweep, sweep_partition
from complete_credit_rational_reference_v1 import direct_step, tent


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(encode(value), stream, ensure_ascii=False, indent=2)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arithmetic_tests():
    checks = Counter()
    cache = {}
    assert roots(Poly(7)) == [] and roots(Poly(0)) == []
    assert roots(Poly((3, 2))) == [F(-3, 2)]
    assert roots(Poly((1, 0, 1))) == []
    assert roots(Poly((1, -2, 1))) == [F(1)]
    assert roots(Poly((-2, 1, 1))) == [F(-2), F(1)]
    checks['degree_and_discriminant_cases'] = 6
    irr = roots(Poly((-2, 0, 1)), cache)
    assert roots(Poly((10, 0, -5)), cache) == irr
    assert compare(irr[0], F(-3, 2)) > 0 and compare(irr[0], F(-7, 5)) < 0
    assert compare(irr[1], F(7, 5)) > 0 and compare(irr[1], F(3, 2)) < 0
    for r in irr:
        assert sign_at(Poly((-2, 0, 1)), r) == 0
        assert sign_at(Poly((-2, 1, 1)), r) == (-1 if r.index == 0 else 1)
        assert sign_at(Poly((3, 0, -1)), r) == 1
        assert compare(r, r) == 0
        checks['algebraic_root_signs'] += 4
    eps = F(1, 2**100)
    shifted = roots(Poly((eps * eps - 2, -2 * eps, 1)), cache)
    assert compare(irr[0], shifted[0]) < 0 and compare(irr[1], shifted[1]) < 0
    q = between(irr[1], shifted[1])
    assert compare(irr[1], q) < 0 and compare(q, shifted[1]) < 0
    assert sorted_points([irr[1], shifted[0], irr[0], shifted[1], F(0), irr[1]]) == [
        irr[0], shifted[0], F(0), irr[1], shifted[1]]
    checks['close_roots_and_exact_order'] = 4
    # A thin *arithmetic* interval. This is deliberately not an ML fixture.
    p = Poly((1, -1)) * Poly((1 + F(1, 2**80), -1))
    assert roots(p) == [F(1), F(1) + F(1, 2**80)]
    checks['narrow_rational_roots'] = 1
    return dict(checks=dict(checks), narrow_interval_width=F(1, 2**80),
                close_irrational_root_separation=eps)


def generic_tests():
    def oracle_for(poly):
        def oracle(point, budget, strict):
            tr = Trace(point, budget, strict)
            result = tr.test(poly, 'synthetic_polynomial')
            return dict(mode=str(result), guards=list(tr.guards.values()))
        return oracle
    narrow_poly = Poly((1, -1)) * Poly((1 + F(1, 2**80), -1))
    narrow = partition(oracle_for(narrow_poly), low=F(0), high=F(2))
    assert narrow['coverage_complete'] and len(narrow['cells']) == 3
    inside = [c for c in narrow['cells'] if c['mode'] == '-1']
    assert len(inside) == 1 and inside[0]['high'] - inside[0]['low'] == F(1, 2**80)
    irrational = partition(oracle_for(Poly((-2, 0, 1))), low=F(-2), high=F(2))
    assert irrational['coverage_complete'] and len(irrational['cells']) == 3
    assert [b['mode'] for b in irrational['boundaries']] == ['1', '0', '0', '1']
    repeated = partition(oracle_for(Poly((0, 0, 1))), low=F(-1), high=F(1))
    assert repeated['coverage_complete'] and len(repeated['cells']) == 2
    assert [b['mode'] for b in repeated['boundaries']] == ['1', '0', '1']
    zero = partition(oracle_for(Poly(0)))
    assert zero['coverage_complete'] and len(zero['cells']) == 1
    capped = partition(oracle_for(narrow_poly), low=F(0), high=F(2), max_segments=1)
    assert not capped['coverage_complete'] and capped['pending'] and len(capped['cells']) == 1
    low_budget = partition(oracle_for(narrow_poly), low=F(0), high=F(2), comparison_limit=3)
    assert not low_budget['coverage_complete'] and low_budget['pending']
    assert low_budget['polynomial_root_comparisons'] == 3
    return dict(narrow=narrow, irrational=irrational, repeated=repeated, zero=zero,
                capped=capped, low_budget=low_budget)


def fixtures():
    out = []
    out.append(('one_layer', rational_state([.02], [[.3]], [[.2]], [.1], [.4])))
    out.append(('two_layer', rational_state([.02, -.03], [[.25, .68], [.4, .7]],
                                         [[.2, -.1], [.15, .3]], [.1, .3], [.44, .52])))
    out.append(('four_layer_coupled', rational_state(
        [.02, -.03, .015, -.025], [[.25, .72], [.3, .58], [.52, .83], [.65, .34]],
        [[.3, -.2], [.17, .21], [-.2, .1], [.1, -.3]], [.1, .3], [.41, .59])))
    x = [.12, .34, .56, .78]
    b = [.01, -.02, .03, -.04]
    h, previous = [], [F(q) for q in x]
    for bias in b:
        previous = [tent(q + F(bias)) for q in previous]
        h.append(previous)
    out.append(('zero_direction_four_layer', rational_state(b, h, [[0] * 4 for _ in b], x, [.4] * 4)))
    # Exact knot coincidences and permanent ties, not approximate epsilon ties.
    out.append(('knots_and_ties', rational_state([0., 0.], [[0., .5], [0., 1.]],
                                              [[0., 0.], [0., 0.]], [0., .25], [0., 1.])))
    out.append(('negative_noisy_observation', rational_state([0.], [[0.]], [[.2]], [0.], [-.0005])))
    out.append(('above_one_noisy_observation', rational_state([0.], [[1.]], [[.2]], [.5], [1.0005])))
    for impossible in [-.002, 1.002]:
        try:
            rational_state([0.], [[0.]], [[0.]], [0.], [impossible])
        except AssertionError:
            pass
        else:
            raise AssertionError('An impossible observation band was accepted')
    return out


def eval_polys(values, point):
    if isinstance(values, list):
        return [eval_polys(v, point) for v in values]
    return values.at(point)


def compare_reference(result, state, point):
    direct = direct_step(state, point)
    for key in ['b', 'h', 'forward']:
        assert eval_polys(result[key], point) == direct[key], (key, point)
    assert result['codes'] == direct['codes'] and result['mode'] == direct['mode']
    return direct


def check_fixture(name, state, out, *, float_checks=True):
    print(dict(fixture=name, phase='partition'), flush=True)
    result = sweep_partition(state)
    save(out / f'{name}_partition.json', result)
    assert result['coverage_complete'], (name, result['limit_reason'], len(result['cells']))
    checks = Counter()
    points = []
    for cell in result['cells']:
        a = between(cell['low'], cell['witness'])
        b = between(cell['witness'], cell['high'])
        for point in [a, b]:
            direct = compare_reference(cell, state, point)
            checks['exact_open_cell_points'] += 1
            checks['distinct_value_activity_ties'] += len(direct['activity_ties'])
            checks['distinct_value_bias_ties'] += len(direct['bias_ties'])
            for _, guard, expected in cell['guards']:
                assert sign_at(guard, point) == expected != 0
                # Independent cell certificate: no guard root strictly inside.
                for root in roots(guard):
                    assert not (compare(cell['low'], root) < 0 and compare(root, cell['high']) < 0)
                checks['guard_root_exclusions'] += 1
            points.append((point, cell))
    for boundary in result['boundaries']:
        if isinstance(boundary['point'], F):
            compare_reference(boundary, state, boundary['point'])
            checks['exact_rational_boundary_points'] += 1
        else:
            for _, guard, expected in boundary['guards']:
                assert sign_at(guard, boundary['point']) == expected
            checks['exact_algebraic_boundary_guard_checks'] += len(boundary['guards'])
    float_rows, float_summary = [], {}
    if float_checks:
        import numpy as np
        import cold_stagnation_switch as cold
        from posterior_confirmation_pipeline import discovery_box
        from dual_amplitude_local_audit_v1 import scalar_step
        b0, h0, u, x, v = [np.array(state[k], dtype=float) for k in ['b', 'h', 'direction', 'x', 'v']]
        with discovery_box(.12):
            for point, cell in points:
                alpha = float(point)
                # First check whether binary64 rounding left the exact cell.
                rounded_inside = compare(cell['low'], F(alpha)) < 0 and compare(F(alpha), cell['high']) < 0
                exact_at_float = sweep(state, F(alpha), strict=False)
                expected = {k: np.array(eval_polys(exact_at_float[k], F(alpha)), dtype=float) for k in ['b', 'h']}
                production = cold.Local(b0[None], x, v, 'alm')
                production.h = h0[:, None].copy()
                production.u = (alpha * u)[:, None].copy()
                production.step()
                scalar = scalar_step(b0, h0, u, x, v, alpha)
                gap_prod = max(float(np.max(abs(production.b[0] - expected['b']))),
                               float(np.max(abs(production.h[:, 0] - expected['h']))))
                gap_scalar = max(float(np.max(abs(scalar[k] - expected[k]))) for k in ['b', 'h'])
                row = dict(alpha=alpha, exact_point=str(point), rounded_inside_cell=rounded_inside,
                                       production_gap=gap_prod, scalar_gap=gap_scalar,
                                       exceeds_fixed_1e_10_check=max(gap_prod, gap_scalar) > 1e-10)
                if row['exceeds_fixed_1e_10_check']:
                    row.update(expected={k: expected[k].tolist() for k in ['b', 'h']},
                               production_b=production.b[0].tolist(), production_h=production.h[:, 0].tolist(),
                               scalar={k: scalar[k].tolist() for k in ['b', 'h']})
                float_rows.append(row)
        save(out / f'{name}_floating_checks.json', float_rows)
        # A discrepancy is preserved as evidence, never hidden by increasing tolerance.
        float_summary = dict(points=len(float_rows), outside_after_rounding=sum(not r['rounded_inside_cell'] for r in float_rows),
                             discrepancies=sum(r['exceeds_fixed_1e_10_check'] for r in float_rows),
                             max_production_gap=max(r['production_gap'] for r in float_rows),
                             max_scalar_gap=max(r['scalar_gap'] for r in float_rows))
    summary = dict(fixture=name, cells=len(result['cells']), boundaries=len(result['boundaries']),
                   irrational_boundaries=sum(isinstance(r['point'], Root) for r in result['boundaries']),
                   comparisons=result['polynomial_root_comparisons'], partition_seconds=result['seconds'],
                   checks=dict(checks), floating=float_summary,
                   exact_coverage_passed=True, floating_agreement_passed=float_summary.get('discrepancies', 0) == 0)
    save(out / f'{name}_summary.json', summary)
    print(summary, flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--fixtures', nargs='*', default=None)
    parser.add_argument('--no-float', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    src = Path(__file__).parent
    names = ['exact_quadratic_events_v1.py', 'complete_credit_amplitude_events_v1.py', 'complete_credit_amplitude_events_v2.py',
             'complete_credit_rational_reference_v1.py', Path(__file__).name,
             'cold_stagnation_switch.py', 'streaming_branch_projection.py',
             'dual_amplitude_local_audit_v1.py', 'posterior_confirmation_pipeline.py']
    save(args.out / 'protocol.json', dict(source_sha256={n: sha(src / n) for n in names},
         scope='synthetic implementation validation, not task-effectiveness evidence',
         query_answers_accessed=False, comparison_limit=1000000, max_segments=4096,
         requested_fixtures=args.fixtures, float_checks=not args.no_float, fixed_float_check=1e-10))
    try:
        arithmetic = arithmetic_tests()
        save(args.out / 'arithmetic.json', arithmetic)
        generic = generic_tests()
        save(args.out / 'generic_partitions.json', generic)
        rows = [check_fixture(n, s, args.out, float_checks=not args.no_float)
                for n, s in fixtures() if args.fixtures is None or n in args.fixtures]
        assert rows
        save(args.out / 'summary.json', dict(exact_tests_passed=True,
             floating_checks_passed=all(r['floating_agreement_passed'] for r in rows),
             arithmetic=arithmetic, fixtures=rows, seconds=time.perf_counter() - started,
             outputs_sha256={p.name: sha(p) for p in args.out.iterdir() if p.is_file()}))
    except Exception:
        save(args.out / 'failure.json', dict(traceback=traceback.format_exc(), seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    main()
