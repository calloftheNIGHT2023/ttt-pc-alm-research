"""Strict input checks and unchanged numerical budgets for offline geometry.

This wrapper adds no online computation and never accesses a query teacher.
The 1e-12 coupling budget is the already frozen mode-forward audit tolerance,
not a task-quality or significance threshold. Preserve the first certificate.
"""
from fractions import Fraction as F
import numpy as np
from probe_exact_polytope_certificate import certify


READOUT_BUDGET = F(1, 10**12)


def validate(poly, rows, rhs):
    assert len(rows) == len(rhs) and len(rows) >= 8
    assert all(len(r) == 4 and all(t == int(t) for t in r) for r in rows), 'Integer normals required'
    for key in ['center', 'scale', 'interior']:
        assert np.asarray(poly[key]).shape == (4,), ('Bad affine shape', key)
    facets = np.asarray(poly['facets']); weights = np.asarray(poly['simplex_probs'])
    assert facets.ndim == 3 and facets.shape[1:] == (4, 4) and len(facets) >= 5, 'Bad facet shape'
    assert weights.shape == (len(facets),), 'Probability count differs from facet count'
    assert np.asarray(poly['volume']).ndim == 0, 'Non-scalar volume'
    for key in ['center', 'scale', 'interior', 'facets', 'simplex_probs', 'volume']:
        assert np.isfinite(poly[key]).all(), ('Nonfinite geometry', key)
    assert np.all(np.asarray(poly['scale']) > 0)
    assert np.all(weights >= 0) and weights.sum() > 0, 'Invalid probabilities'
    assert abs(float(weights.sum()) - 1.) <= 32*np.finfo(float).eps, 'Probabilities not normalized'
    assert float(poly['volume']) > 0


def checked_certificate(poly, rows, rhs):
    validate(poly, rows, rhs)
    result, witness = certify(poly, rows, rhs)
    assert np.isclose(result['reference_volume'], float(poly['volume']), rtol=1e-8, atol=1e-22), 'True volume mismatch'
    assert F(result['exact_nominal_readout_expectation_error_bound']) <= READOUT_BUDGET, 'Nominal readout coupling exceeds existing 1e-12 forward budget'
    result['input_validation_passed'] = True
    result['existing_volume_tolerance_retained'] = True
    result['readout_coupling_budget'] = float(READOUT_BUDGET)
    return result, witness
