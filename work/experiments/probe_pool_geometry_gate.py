"""Certify components structurally, enforce the unchanged final-pool budget.

For means g_i,h_i in [0,1], |g_i-h_i|<=e_i and probability vectors p,r,
|sum p_i*g_i - sum r_i*h_i| <= TV(p,r) + sum p_i*e_i.
The removed per-component e_i<=B condition was sufficient, not necessary for
this final guarantee. It remains a recorded diagnostic, never a passed claim.
"""
from fractions import Fraction as F
import numpy as np
from probe_exact_polytope_certificate import certify
from probe_exact_geometry_gate import validate, READOUT_BUDGET


def checked_component(poly, rows, rhs):
    validate(poly, rows, rhs)
    result, witness = certify(poly, rows, rhs)
    assert np.isclose(result['reference_volume'], float(poly['volume']), rtol=1e-8, atol=1e-22), 'True volume mismatch'
    bound = F(result['exact_nominal_readout_expectation_error_bound'])
    assert bound >= 0
    result.update(input_validation_passed=True, existing_volume_tolerance_retained=True,
        original_component_budget_diagnostic_passed=bound <= READOUT_BUDGET,
        component_only_not_final_pool_acceptance=True,
        final_pool_budget=str(READOUT_BUDGET))
    return result, witness


def exact_pool_bound(stored, truth, errors):
    stored, truth, errors = [[F(t) for t in xs] for xs in [stored, truth, errors]]
    assert len(stored) == len(truth) == len(errors) > 0
    assert sum(stored) == sum(truth) == 1
    assert all(t >= 0 for t in stored + truth + errors)
    tv = sum(abs(a-b) for a,b in zip(stored, truth)) / 2
    weighted = sum(p*e for p,e in zip(stored, errors))
    bound = tv + weighted
    return dict(exact_mode_weight_tv=str(tv), mode_weight_tv=float(tv),
        exact_weighted_component_bound=str(weighted), weighted_component_bound=float(weighted),
        exact_nominal_readout_error_bound=str(bound), nominal_readout_error_bound=float(bound),
        squared_risk_of_mean_difference_bound=float(2*bound), budget=float(READOUT_BUDGET),
        original_component_threshold_exceedances=sum(e > READOUT_BUDGET for e in errors),
        full_pool_bound_within_original_budget=bound <= READOUT_BUDGET)


def independent_pool_bound_check(stored, truth, errors, record):
    # Positive and negative weight changes must match. This independently
    # checks the algebra, rather than calling exact_pool_bound.
    p, r, e = [[F(t) for t in xs] for xs in [stored, truth, errors]]
    assert len(p) == len(r) == len(e) > 0 and sum(p) == sum(r) == 1
    assert min(p+r+e) >= 0
    added = sum(max(F(0), a-b) for a,b in zip(p, r))
    removed = sum(max(F(0), b-a) for a,b in zip(p, r))
    assert added == removed
    weighted = F(0)
    for index in range(len(p)):
        weighted += p[index]*e[index]
    bound = added + weighted
    assert F(record['exact_mode_weight_tv']) == added
    assert F(record['exact_weighted_component_bound']) == weighted
    assert F(record['exact_nominal_readout_error_bound']) == bound
    assert record['mode_weight_tv'] == float(added)
    assert record['weighted_component_bound'] == float(weighted)
    assert record['nominal_readout_error_bound'] == float(bound)
    assert record['squared_risk_of_mean_difference_bound'] == float(2*bound)
    assert record['budget'] == float(READOUT_BUDGET)
    assert record['original_component_threshold_exceedances'] == sum(t > READOUT_BUDGET for t in e)
    assert record['full_pool_bound_within_original_budget'] == (bound <= READOUT_BUDGET)
    return dict(passed=True, components=len(p), bound_within_budget=bound <= READOUT_BUDGET)


def pool_coupling(keys, cache, certificates):
    assert keys and len(set(keys)) == len(keys)
    raw = np.array([cache[key]['volume'] for key in keys])
    assert raw.shape == (len(keys),) and np.isfinite(raw).all() and np.all(raw > 0)
    raw /= raw.sum()
    stored = [F(float(t)) for t in raw]
    total = sum(stored)
    stored = [t/total for t in stored]
    truth = [F(certificates[key]['exact_reference_volume']) for key in keys]
    assert all(t > 0 for t in truth)
    total = sum(truth)
    truth = [t/total for t in truth]
    errors = [F(certificates[key]['exact_nominal_readout_expectation_error_bound']) for key in keys]
    result = exact_pool_bound(stored, truth, errors)
    result['independent_bound_check'] = independent_pool_bound_check(stored, truth, errors, result)
    assert result['full_pool_bound_within_original_budget'], ('Full-pool nominal coupling exceeds unchanged 1e-12 budget', result['nominal_readout_error_bound'])
    return result
