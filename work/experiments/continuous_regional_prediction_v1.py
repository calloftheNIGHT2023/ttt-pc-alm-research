"""384 prototype: fresh support -> search -> common geometry -> prediction.

No stage archives, offline modes, query targets or cross-call solver cache are
inputs. This wrapper is not a new optimizer; it removes staged timing reuse.
"""
import time
import numpy as np
import retired_region_join_v1 as search
import candidate_set_readout_v1 as reader


def fit(x, v, q, *, seed, name, schedule, order_name='observed', max_seconds=8., max_expanded=65536):
    begin = time.perf_counter()
    assert x.shape == v.shape and x.ndim == q.ndim == 1
    sa, sm = search.join(x, v, name=name, schedule=schedule, order_name=order_name,
                         max_seconds=max_seconds, max_expanded=max_expanded)
    ra, rm = reader.fit(x, v, q, sa['regions'], seed=seed, search_completed=bool(sm['completed']))
    arrays = {**{'search_'+k: value for k, value in sa.items()},
              **{'readout_'+k: value for k, value in ra.items()}}
    meta = dict(method=name, schedule=schedule, order_name=order_name, search=sm, readout=rm,
        search_seconds=sm['total_seconds'], readout_seconds=rm['charged_readout_seconds'],
        query_targets_accessed=False, archived_search_used=False, global_bp_used=False,
        geometry_global_lp_used=rm['global_geometry_lp_used'],
        cross_method_cache_used=False, completed=sm['completed'],
        full_candidate_set_resolved=rm['full_candidate_set_resolved'], readout_available=rm['readout_available'],
        returned_array_bytes=sum(a.nbytes for a in arrays.values()),
        search_named_bytes_lower_bound=sm['live_named_array_bytes_lower_bound'],
        geometry_numeric_bytes_subtotal=rm['geometry_numeric_bytes_subtotal'],
        search_archive_retained_during_readout=True,
        timing_scope='Contiguous warm-runtime support-to-prediction, including both wrappers; no disk I/O or independent audit',
        memory_scope='Named state only, not RSS peak or complete native geometry transient workspace',
        global_wall_time_budget_enforced=False,
        budget_scope='Search cooperative boundary only; geometry/readout are currently uncapped')
    meta['contiguous_seconds'] = time.perf_counter()-begin
    assert meta['contiguous_seconds'] >= meta['search_seconds']+meta['readout_seconds']
    return arrays, meta
