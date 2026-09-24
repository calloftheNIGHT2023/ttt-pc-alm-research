"""Exact mean-displacement bound, conditional on a valid true-H certificate.

Couple saved and mapped cones by the same Dirichlet(1,1,1,1,1) barycentrics.
Their cone origins are identical. E[beta_k]=1/5, so mean displacement is the
saved-probability-weighted average of four facet-vertex distances divided by5.
This replaces a worst-point displacement by a tighter bound on the SAME mean;
it does not certify finite RNG or floating-point execution error.
"""
from fractions import Fraction as F
import numpy as np


def position_bound(poly, witness):
    facets = np.asarray(poly['facets'])
    assert facets.ndim == 3 and facets.shape[1:] == (4, 4)
    assert np.isfinite(facets).all()
    raw, indices = np.unique(facets.reshape(-1, 4), axis=0, return_inverse=True)
    matching = witness['matching']
    assert len(matching) == len(raw)
    exact = [[F(t) for t in row] for row in witness['vertices']]
    assert all(len(row) == 4 for row in exact)
    assert all(type(i) is int and 0 <= i < len(exact) for i in matching)
    center, scale = [[F(float(t)) for t in poly[name]] for name in ['center', 'scale']]
    assert len(center) == len(scale) == 4 and min(scale) > 0
    origin = [c+s*F(float(t)) for c,s,t in zip(center,scale,poly['interior'])]
    assert origin == [F(t) for t in witness['origin']], 'Origin displacement must not be omitted'
    errors = [[abs(c+s*F(float(t))-v) for c,s,t,v in zip(center,scale,point,exact[index])]
              for point,index in zip(raw,matching)]
    distances = [sum(F(2**(4-j))*d for j,d in enumerate(row)) for row in errors]
    cone_costs = [sum(distances[int(i)] for i in face)/5 for face in indices.reshape(-1,4)]
    weights = [F(float(t)) for t in poly['simplex_probs']]
    assert len(weights) == len(cone_costs) and min(weights) >= 0 and sum(weights) > 0
    total = sum(weights)
    weights = [t/total for t in weights]
    upper = sum(p*c for p,c in zip(weights,cone_costs))
    maximum = sum(F(2**(4-j))*max(row[j] for row in errors) for j in range(4))
    assert 0 <= upper <= F(4,5)*maximum
    return dict(exact_barycentric_position_bound=str(upper), barycentric_position_bound=float(upper),
        exact_worst_point_position_bound=str(maximum), worst_point_position_bound=float(maximum),
        exact_cone_position_bounds=[str(t) for t in cone_costs],
        origin_displacement_exactly_zero=True, exact_barycentric_coordinate_mean='1/5',
        scope='Conditional on certified mapped fan; nominal mean only, not finite-particle pathwise error')


def independently_check(poly, witness, record):
    # Deliberately reconstruct sorted vertex identities with Python tuples and
    # accumulate unnormalized masses before dividing, not via position_bound.
    points = sorted(set(tuple(float(t) for t in vertex) for face in poly['facets'] for vertex in face))
    assert len(points) == len(witness['matching'])
    exact = [tuple(F(t) for t in row) for row in witness['vertices']]
    lookup = {point: exact[witness['matching'][i]] for i,point in enumerate(points)}
    origin = tuple(F(float(poly['center'][j]))+F(float(poly['scale'][j]))*F(float(poly['interior'][j])) for j in range(4))
    assert origin == tuple(F(t) for t in witness['origin'])
    total_weight = F(0)
    numerator = F(0)
    maxima = [F(0)]*4
    assert len(poly['facets']) == len(poly['simplex_probs']) == len(record['exact_cone_position_bounds'])
    for index, face in enumerate(poly['facets']):
        cost = F(0)
        for vertex in face:
            target = lookup[tuple(float(t) for t in vertex)]
            for j in range(4):
                displacement = abs(F(float(poly['center'][j]))+F(float(poly['scale'][j]))*F(float(vertex[j]))-target[j])
                maxima[j] = max(maxima[j], displacement)
                cost += displacement * (1 << (4-j))
        cost /= 5
        assert cost == F(record['exact_cone_position_bounds'][index])
        weight = F(float(poly['simplex_probs'][index]))
        assert weight >= 0
        total_weight += weight
        numerator += weight * cost
    assert total_weight > 0
    mean = numerator / total_weight
    maximum = sum(maxima[j]*(1 << (4-j)) for j in range(4))
    assert mean == F(record['exact_barycentric_position_bound'])
    assert record['barycentric_position_bound'] == float(mean)
    assert maximum == F(record['exact_worst_point_position_bound'])
    assert record['worst_point_position_bound'] == float(maximum)
    assert record['origin_displacement_exactly_zero'] is True and record['exact_barycentric_coordinate_mean'] == '1/5'
    assert 0 <= mean <= maximum*4/5
    return dict(passed=True, cones=len(poly['facets']), exact_position_bound=str(mean), independent_arithmetic=True)
