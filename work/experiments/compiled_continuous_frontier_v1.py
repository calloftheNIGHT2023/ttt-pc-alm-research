"""425 shared exact range compilation inside the paid continuous pipeline."""
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction as F
import pickle
import time
import numpy as np
import continuous_frontier_prediction_v1 as original
import piecewise_frontier_enclosure_v1 as compiler


def stream_ranges(sa, sm, q, cheap, cheap_prediction, *, batch_queries=32, emit=None):
    begin = time.perf_counter()
    assert type(batch_queries) is int and batch_queries > 0
    prepared = original.ranges.fit(sa, sm, np.empty(0, np.float64))
    preparation_seconds = time.perf_counter()-begin
    identity = tuple(tuple(F(int(i == j)) for j in range(4)) for i in range(4))
    boxes = prepared['coordinate_boxes']
    tick = time.perf_counter()
    programs = [compiler.compile_box(b['inverse'], b['intervals']) for b in boxes]
    programs.append(compiler.compile_box(identity, [(-original.ranges.core.BOUND, original.ranges.core.BOUND)]*4))
    compile_seconds = time.perf_counter()-tick
    # Charged online state accounting; this is serialized size, not peak RSS.
    program_pickle_bytes = len(pickle.dumps(programs, protocol=5))
    state_accounting_seconds = time.perf_counter()-tick-compile_seconds
    order = original.query_order(q); prediction = cheap_prediction.copy()
    raw = [None]*len(q); combined = [None]*len(q); cells = [None]*len(q)
    lower_upper = np.empty((len(q), 2)); checkpoints = []; crossings = 0; routes = Counter()
    read_start = time.perf_counter()
    for start in range(0, len(q), batch_queries):
        ids = order[start:start+batch_queries]
        for index in ids:
            i = int(index); point = F(float(q[i])); bounds = []
            for program in programs[:-1]:
                bound, count, route = compiler.evaluate(program, point)
                bounds.append(bound); crossings += count; routes[route] += 1
            prior, _, route = compiler.evaluate(programs[-1], point); routes[route] += 1
            union = min(b[0] for b in bounds), max(b[1] for b in bounds)
            raw[i] = original.ranges.core.mathcore.intersect_intervals(union, prior)
            combined[i] = original.ranges.core.mathcore.intersect_intervals(raw[i], cheap[i])
            cells[i] = bounds
        endpoints = original.outward([combined[int(i)] for i in ids]); lower_upper[ids] = endpoints
        prediction[ids] = np.clip(cheap_prediction[ids], endpoints[:, 0], endpoints[:, 1])
        if emit is not None:
            emit('fallback', prediction.copy())
        checkpoints.append(dict(processed=min(start+batch_queries, len(q)), indices=ids.tolist(),
                                seconds=time.perf_counter()-begin))
    reading_seconds = time.perf_counter()-read_start
    metadata = dict(upper=prepared['upper'], coordinate_boxes=boxes, empty_cells=prepared['empty_cells'],
        frontier_intervals=raw, cheap_intervals=cheap, intersected_intervals=combined,
        cell_query_ranges=[list(c) for c in zip(*cells)], crossing_operations=crossings,
        prepare_seconds=preparation_seconds, compile_seconds=compile_seconds,
        state_accounting_seconds=state_accounting_seconds, reading_seconds=reading_seconds,
        checkpoints=checkpoints, total_seconds=time.perf_counter()-begin,
        exact_rational_bounds=True, outward_rounded_delivery=True, archived_search_used=False,
        global_bp_used=False, new_global_lp_used=False, query_targets_accessed=False,
        posterior_mass_required=False, full_modes_required=False,
        compiled_programs=programs, compiled_routes=dict(routes), program_pickle_bytes=program_pickle_bytes,
        program_bytes_are_not_peak_RSS=True, compiled_inside_paid_call=True,
        original_reference_at_boundaries=True, whole_box_fallback_on_segment_cap=True)
    return prediction, lower_upper, order, metadata


@contextmanager
def installed():
    before = original.stream_ranges
    original.stream_ranges = stream_ranges
    try:
        yield
    finally:
        original.stream_ranges = before


def fit(*args, **kwargs):
    before = original.stream_ranges
    with installed():
        result = original.fit(*args, **kwargs)
    assert original.stream_ranges is before
    return result
