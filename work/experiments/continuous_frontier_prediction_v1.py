"""417 fresh support -> search -> anytime certified 257-query predictions.

No stored frontier, query targets or teacher parameters enter this interface.
Exact Fraction enclosures are outward-rounded before any delivered projection.
"""
from contextlib import contextmanager
from fractions import Fraction as F
import time
import numpy as np
import retired_region_join_v1 as search
import frontier_range_certificate_v1 as ranges
import candidate_set_readout_v1 as reader
import independent_hybrid_memory as regression

METHODS = ('regional_active512', 'regional_passive512', 'regional_instant512',
           'pdhg_cold512', 'pdhg_box512', 'none', 'cheap_only')


@contextmanager
def registered():
    """Expose an already implemented ablation without changing frozen sources."""
    before = search.CONFIGS
    assert not any(name == 'regional_instant512' for name, _, _ in before)
    search.CONFIGS = before+[('regional_instant512', 'regional_instant', 512)]
    try:
        yield
    finally:
        search.CONFIGS = before


def registry_identity():
    return (id(search.CONFIGS), id(search.original.CONFIGS), id(search.original.regional.solve))


def outward(intervals):
    result = np.empty((len(intervals), 2), np.float64)
    for i, (a, b) in enumerate(intervals):
        a, b = F(a), F(b)
        assert 0 <= a <= b <= 1
        low, high = float(a), float(b)
        if F(low) > a:
            low = float(np.nextafter(low, -np.inf))
        if F(high) < b:
            high = float(np.nextafter(high, np.inf))
        assert F(low) <= a <= b <= F(high)
        result[i] = low, high
    return result


def query_order(q):
    n = len(q)
    assert n > 0
    bits = max(1, (n-1).bit_length())
    ranks = sorted(range(n), key=lambda i: int(format(i, f'0{bits}b')[::-1], 2))
    return np.argsort(q, kind='stable')[ranks]


def stream_ranges(sa, sm, q, cheap, cheap_prediction, *, batch_queries=32, emit=None):
    """One paid cover construction; independent exact range per query batch."""
    begin = time.perf_counter()
    assert type(batch_queries) is int and batch_queries > 0
    # Empty query list executes only the frozen coordinate-box preparation.
    prepared = ranges.fit(sa, sm, np.empty(0, np.float64))
    preparation_seconds = time.perf_counter()-begin
    identity = tuple(tuple(F(int(i == j)) for j in range(4)) for i in range(4))
    prior_box = [(-ranges.core.BOUND, ranges.core.BOUND)]*4
    order = query_order(q)
    prediction = cheap_prediction.copy()
    raw = [None]*len(q); combined = [None]*len(q); cells = [None]*len(q)
    lower_upper = np.empty((len(q), 2)); checkpoints = []; crossings = 0
    for start in range(0, len(q), batch_queries):
        ids = order[start:start+batch_queries]
        for index in ids:
            i = int(index); point = F(float(q[i])); bounds = []
            for box in prepared['coordinate_boxes']:
                value, count = ranges.enclose(point, box['inverse'], box['intervals'])
                bounds.append(value); crossings += count
            prior, _ = ranges.enclose(point, identity, prior_box)
            union = min(b[0] for b in bounds), max(b[1] for b in bounds)
            raw[i] = ranges.core.mathcore.intersect_intervals(union, prior)
            combined[i] = ranges.core.mathcore.intersect_intervals(raw[i], cheap[i])
            cells[i] = bounds
        endpoints = outward([combined[int(i)] for i in ids])
        lower_upper[ids] = endpoints
        prediction[ids] = np.clip(cheap_prediction[ids], endpoints[:, 0], endpoints[:, 1])
        if emit is not None:
            emit('fallback', prediction.copy())
        checkpoints.append(dict(processed=min(start+batch_queries, len(q)), indices=ids.tolist(),
                                seconds=time.perf_counter()-begin))
    metadata = dict(upper=prepared['upper'], coordinate_boxes=prepared['coordinate_boxes'],
        empty_cells=prepared['empty_cells'], frontier_intervals=raw, cheap_intervals=cheap,
        intersected_intervals=combined, cell_query_ranges=[list(c) for c in zip(*cells)],
        crossing_operations=crossings, prepare_seconds=preparation_seconds,
        reading_seconds=time.perf_counter()-begin-preparation_seconds,
        checkpoints=checkpoints, total_seconds=time.perf_counter()-begin,
        exact_rational_bounds=True, outward_rounded_delivery=True, archived_search_used=False,
        global_bp_used=False, new_global_lp_used=False, query_targets_accessed=False,
        posterior_mass_required=False, full_modes_required=False)
    return prediction, lower_upper, order, metadata


def fit(x, v, q, *, seed, method, search_seconds=.25, max_expanded=65536,
        batch_queries=32, emit=None):
    begin = time.perf_counter()
    assert method in METHODS and x.shape == v.shape and x.ndim == q.ndim == 1
    assert len(x) > 0 and len(q) > 0 and search_seconds >= 0 and max_expanded > 0
    assert all(np.isfinite(a).all() for a in [x, v, q])
    state_before = registry_identity()
    with reader.local.no_bp(True):
        fallback, _, fm = regression.regression(x, v, 'prior4096_ridge')
        fallback_prediction = np.clip(fallback(q), 0., 1.)
        if emit is not None:
            emit('fallback', fallback_prediction.copy())
        fallback_seconds = time.perf_counter()-begin
        tick = time.perf_counter()
        cheap = ranges.core.lipschitz_intervals(x, v, q)
        cheap_bounds = outward(cheap)
        cheap_prediction = np.clip(fallback_prediction, cheap_bounds[:, 0], cheap_bounds[:, 1])
        if emit is not None:
            emit('fallback', cheap_prediction.copy())
        cheap_seconds = time.perf_counter()-tick
        arrays = dict(x=x.copy(), v=v.copy(), q=q.copy(), fallback_prediction=fallback_prediction,
                      cheap_prediction=cheap_prediction, cheap_bounds=cheap_bounds)
        metadata = dict(method=method, fallback_seconds=fallback_seconds, fallback_metadata=fm,
            cheap_seconds=cheap_seconds, cheap_intervals=cheap, search=None, readout=None, range_readout=None,
            branch='cheap_only', query_targets_accessed=False, archived_search_used=False,
            global_bp_used=False, geometry_global_lp_used=False, cross_method_cache_used=False,
            support_prefix=len(x), queries=len(q), batch_queries=batch_queries,
            search_cooperative_seconds=search_seconds, max_expanded=max_expanded,
            global_wall_time_budget_enforced=False, warm_runtime_setup_charged_by_caller=True,
            exact_objects_state_included_in_numeric_bytes=False,
            memory_scope='Named returned arrays are not peak RSS or exact rational/Python/native state')
        prediction = cheap_prediction.copy()
        if method != 'cheap_only':
            with registered():
                sa, sm = search.join(x, v, name=method, schedule='dfs', order_name='farthest_x',
                                     max_seconds=search_seconds, max_expanded=max_expanded)
            arrays.update({'search_'+k: a for k, a in sa.items()})
            metadata['search'] = sm
            full = False
            if sm['completed']:
                ra, rm = reader.fit(x, v, q, sa['regions'], seed=seed, search_completed=True)
                arrays.update({'readout_'+k: a for k, a in ra.items()})
                metadata['readout'] = rm
                metadata['geometry_global_lp_used'] = rm['global_geometry_lp_used']
                full = rm['readout_available'] and rm['full_candidate_set_resolved']
                if full:
                    prediction = np.clip(ra['prediction'], cheap_bounds[:, 0], cheap_bounds[:, 1])
                    metadata['branch'] = 'completed_posterior'
            if not full:
                prediction, bound, order, rm = stream_ranges(sa, sm, q, cheap, cheap_prediction,
                    batch_queries=batch_queries, emit=emit)
                arrays.update(range_bounds=bound, query_order=order)
                metadata.update(branch='frontier', range_readout=rm)
        arrays['prediction'] = prediction
    assert registry_identity() == state_before
    assert prediction.shape == q.shape and np.isfinite(prediction).all()
    metadata['returned_array_bytes'] = sum(a.nbytes for a in arrays.values())
    metadata['contiguous_seconds'] = time.perf_counter()-begin
    return arrays, metadata
