"""Same true-H and final-pool guarantees, tighter mean-position coupling."""
from fractions import Fraction as F
import numpy as np
from probe_pool_geometry_gate_v2 import checked_component as previous_component
from probe_pool_geometry_gate_v2 import exact_pool_bound, independent_pool_bound_check
from probe_barycentric_position_bound import position_bound, independently_check
from probe_exact_geometry_gate import READOUT_BUDGET


def checked_component(poly, rows, rhs):
    result, witness = previous_component(poly, rows, rhs)
    position = position_bound(poly, witness)
    checked = independently_check(poly, witness, position)
    worst = sum(F(2**(4-j))*F(t) for j,t in enumerate(result['exact_maximum_coordinate_displacements']))
    assert worst == F(position['exact_worst_point_position_bound'])
    refined = F(result['exact_weight_total_variation']) + F(position['exact_barycentric_position_bound'])
    assert 0 <= refined <= F(result['exact_nominal_readout_expectation_error_bound'])
    result.update(barycentric_position_certificate=position, independent_barycentric_check=checked,
        exact_barycentric_nominal_readout_error_bound=str(refined),
        barycentric_nominal_readout_error_bound=float(refined),
        original_worst_point_certificate_retained=True)
    return result, witness


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
    errors = [F(certificates[key]['exact_barycentric_nominal_readout_error_bound']) for key in keys]
    old_errors = [F(certificates[key]['exact_nominal_readout_expectation_error_bound']) for key in keys]
    assert all(0 <= new <= old for new,old in zip(errors,old_errors))
    result = exact_pool_bound(stored, truth, errors)
    result['independent_bound_check'] = independent_pool_bound_check(stored, truth, errors, result)
    previous = exact_pool_bound(stored, truth, old_errors)
    assert F(result['exact_nominal_readout_error_bound']) <= F(previous['exact_nominal_readout_error_bound'])
    result.update(original_worst_point_pool_bound=previous,
        bound_revision='Exact mean barycentric displacement, unchanged final 1e-12',
        nominal_mean_not_pathwise_particle_bound=True)
    assert result['full_pool_bound_within_original_budget'], ('Full-pool nominal coupling exceeds unchanged 1e-12 budget', result['nominal_readout_error_bound'])
    return result
