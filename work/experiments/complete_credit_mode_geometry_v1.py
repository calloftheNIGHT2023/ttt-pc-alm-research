"""308 offline common mode classification with explicit rational certificates.

This evaluator is not an online local-credit component. A numerical LP proposes
points or dual weights; exact arithmetic alone accepts positive/interior,
infeasible, or zero-measure certificates. Unresolved cases remain unresolved.
"""
from fractions import Fraction as F
import time

import numpy as np
from scipy.optimize import linprog
import enumerate_support_modes as reference
import conditioned_mode_geometry as conditioned

geometry = conditioned.geometry
BOUND = F(.12)
EPS = F(.001)
RATIONALIZATION_DENOMINATORS = [2**12, 2**20]


def dot(a, b):
    return sum((x * y for x, y in zip(a, b)), F(0))


def box_rows(depth, bound=BOUND):
    rows, rhs = [], []
    for j in range(depth):
        row = [F(int(k == j)) for k in range(depth)]
        rows.extend([row, [-t for t in row]])
        rhs.extend([bound, bound])
    return rows, rhs


def matrices(x, v, key):
    """Exact observation-wise rows plus the frozen geometry's float row order."""
    pattern = np.frombuffer(bytes.fromhex(key), dtype=np.uint8).reshape(4, len(x))
    assert np.all(pattern < 4) and len(x) == len(v)
    assert geometry.base.EPS == float(EPS)
    exact_a, exact_r = reference.constraints(x, v, pattern, exact=True)
    ba, br = box_rows(4)
    exact_a, exact_r = exact_a + ba, exact_r + br
    # Reproduce branch_polytope arithmetic/order with an explicit pattern.
    # Do not infer a pattern from a rounded substitute representative.
    p, c = np.zeros((len(x), 4)), x.copy()
    rows, rhs = [], []
    low = [None, 0., .5, 1.]
    high = [0., .5, 1., None]
    for j in range(4):
        z = p.copy()
        z[:, j] += 1
        reg = pattern[j]
        for i, k in enumerate(reg):
            if high[k] is not None:
                rows.append(z[i].copy())
                rhs.append(high[k] - c[i])
            if low[k] is not None:
                rows.append(-z[i])
                rhs.append(c[i] - low[k])
        p = geometry.base.SLOPES[reg, None] * z
        c = geometry.base.SLOPES[reg] * c + geometry.base.INTERCEPTS[reg]
    for i in range(len(x)):
        rows.extend([p[i].copy(), -p[i]])
        rhs.extend([v[i] + float(EPS) - c[i], -v[i] + float(EPS) + c[i]])
    return pattern, exact_a, exact_r, np.array(rows), np.array(rhs)


def point_certificate(a, rhs, point):
    point = [F(float(p)) for p in point]
    slacks = [r - dot(row, point) for row, r in zip(a, rhs)]
    norms = [sum(map(abs, row), F(0)) for row in a]
    feasible = all(s >= 0 for s in slacks)
    strict = feasible and all(s > 0 for s, n in zip(slacks, norms) if n)
    radius = min(s / n for s, n in zip(slacks, norms) if n)
    # Half the admissible radius gives a cube strictly inside nonconstant rows.
    return dict(type='exact_interior_cube' if strict else 'exact_point_check',
                point=point, closed_region_feasible=feasible, strict_interior=strict,
                radius=radius / 2 if strict else None,
                minimum_slack=min(slacks), nonconstant_rows=sum(n > 0 for n in norms))


def weighted_certificate(a, rhs, weights, bound=BOUND, source='binary64_lp_weights'):
    weights = [F(w) for w in weights]
    assert all(w >= 0 for w in weights)
    coeff = [sum((w * row[j] for w, row in zip(weights, a)), F(0)) for j in range(len(a[0]))]
    constant = dot(weights, rhs)
    lower = -constant - bound * sum(map(abs, coeff), F(0))
    nonconstant_active = any(w > 0 and any(row) for w, row in zip(weights, a))
    conclusion = 'infeasible' if lower > 0 else (
        'zero_volume_or_empty' if lower == 0 and (any(coeff) or nonconstant_active) else 'unresolved')
    return dict(type='exact_box_separation', source=source, conclusion=conclusion,
                weights=[[i, w] for i, w in enumerate(weights) if w],
                weighted_coefficients=coeff, weighted_rhs=constant, lower=lower,
                nonconstant_active_row=nonconstant_active)


def classify_constraints(a, rhs, *, numeric_a=None, numeric_rhs=None, bound=BOUND):
    """All supplied exact inequalities include the explicitly verified box."""
    begin = time.perf_counter()
    a, rhs = [[F(t) for t in row] for row in a], [F(r) for r in rhs]
    depth = len(a[0])
    assert depth == 4 and len(a) == len(rhs) and geometry.PRIOR_BOUND == float(bound)
    pairs = {(tuple(row), r) for row, r in zip(a, rhs)}
    ba, br = box_rows(depth, bound)
    assert all((tuple(row), r) in pairs for row, r in zip(ba, br))
    answer = dict(classification='unresolved', positive_volume_certified=False,
                  closed_region_feasible=None, volume=None, numerical_volume_available=False,
                  lp_calls=0, certificates=[], geometry_note=None, geometry_repairs=[])
    bad = next((i for i, (row, r) in enumerate(zip(a, rhs)) if not any(row) and r < 0), None)
    if bad is not None:
        answer.update(classification='infeasible', closed_region_feasible=False,
                      certificates=[dict(type='negative_constant_row', row_index=bad, rhs=rhs[bad])],
                      seconds=time.perf_counter() - begin)
        return answer
    matrix, rr = np.array(a, dtype=float), np.array(rhs, dtype=float)
    norms = np.abs(matrix).sum(axis=1)
    result = linprog(np.r_[np.zeros(depth), -1.], A_ub=np.c_[matrix, norms], b_ub=rr,
                     bounds=[(None, None)] * (depth + 1),
                     options={'primal_feasibility_tolerance': 1e-9, 'dual_feasibility_tolerance': 1e-9})
    answer.update(lp_calls=1, lp_status=int(result.status), lp_success=bool(result.success))
    if not result.success:
        answer.update(seconds=time.perf_counter() - begin, reason='LP did not return a checkable proposal')
        return answer
    proof = point_certificate(a, rhs, result.x[:-1])
    answer['certificates'].append(proof)
    if proof['closed_region_feasible']:
        answer['closed_region_feasible'] = True
    if proof['strict_interior']:
        answer.update(classification='positive_volume', positive_volume_certified=True)
    else:
        raw_weights = [F(float(w)) for w in np.maximum(-result.ineqlin.marginals, 0.)]
        candidates = [weighted_certificate(a, rhs, raw_weights, bound)]
        for denominator in RATIONALIZATION_DENOMINATORS:
            candidates.append(weighted_certificate(a, rhs, [w.limit_denominator(denominator) for w in raw_weights],
                              bound, source=f'LP weights rationalized denominator<={denominator}'))
        accepted = next((c for c in candidates if c['conclusion'] == 'infeasible'), None)
        if accepted is None:
            accepted = next((c for c in candidates if c['conclusion'] == 'zero_volume_or_empty'), None)
        answer['certificates'].extend(candidates)
        if accepted is not None:
            answer['classification'] = accepted['conclusion']
            if accepted['conclusion'] == 'infeasible':
                assert answer['closed_region_feasible'] is not True
                answer['closed_region_feasible'] = False
    if answer['positive_volume_certified']:
        repairs = []
        na = matrix if numeric_a is None else numeric_a
        nr = rr if numeric_rhs is None else numeric_rhs
        with conditioned.geometry_scope(repairs):
            poly, note = geometry.polytope(na, nr)
        answer.update(geometry_note=note, geometry_repairs=repairs)
        if poly is not None:
            geometry_proof = point_certificate(a, rhs, poly['center'] + poly['scale'] * poly['interior'])
            answer['certificates'].append(geometry_proof)
            if geometry_proof['strict_interior']:
                answer.update(volume=float(poly['volume']), numerical_volume_available=True,
                    numeric_geometry_array_bytes_subtotal=sum(v.nbytes for v in poly.values() if isinstance(v, np.ndarray)),
                    geometry_state_scope='named temporary geometry arrays, not online fit or peak memory')
            else:
                answer['geometry_note'] = dict(original=note, reason='exact geometry-interior check failed; numerical volume withheld')
    answer['seconds'] = time.perf_counter() - begin
    return answer


def classify_mode(x, v, key):
    pattern, a, r, na, nr = matrices(x, v, key)
    answer = classify_constraints(a, r, numeric_a=na, numeric_rhs=nr)
    answer.update(mode=key, observations=len(x), depth=pattern.shape[0])
    return answer


def verify_certificates(a, rhs, result, bound=BOUND):
    """Scalar recheck usable by a later archive audit; no new LP/geometry."""
    count = 0
    for cert in result['certificates']:
        if cert['type'] == 'negative_constant_row':
            i = cert['row_index']
            assert not any(a[i]) and rhs[i] == F(cert['rhs']) < 0
        elif cert['type'] in ['exact_interior_cube', 'exact_point_check']:
            point = [F(p) for p in cert['point']]
            slacks = [F(r) - dot(row, point) for row, r in zip(a, rhs)]
            assert cert['closed_region_feasible'] == all(s >= 0 for s in slacks)
            if cert['strict_interior']:
                radius = F(cert['radius'])
                assert radius > 0
                assert all(s - radius * sum(map(abs, row), F(0)) > 0 if any(row) else s >= 0
                           for row, s in zip(a, slacks))
        else:
            assert cert['type'] == 'exact_box_separation'
            weights = [F(0)] * len(a)
            for i, w in cert['weights']:
                assert weights[i] == 0 and F(w) >= 0
                weights[i] = F(w)
            coeff = [sum((weights[i] * a[i][j] for i in range(len(a))), F(0)) for j in range(len(a[0]))]
            weighted_rhs = sum((weights[i] * rhs[i] for i in range(len(a))), F(0))
            lower = -weighted_rhs - bound * sum(map(abs, coeff), F(0))
            assert coeff == [F(q) for q in cert['weighted_coefficients']]
            assert weighted_rhs == F(cert['weighted_rhs']) and lower == F(cert['lower'])
            if cert['conclusion'] == 'infeasible':
                assert lower > 0
            elif cert['conclusion'] == 'zero_volume_or_empty':
                assert lower == 0 and (any(coeff) or any(w > 0 and any(row) for w, row in zip(weights, a)))
        count += 1
    return count
