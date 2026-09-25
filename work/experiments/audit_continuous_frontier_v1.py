"""417 exact search/cover replay, outward rounding and every delivered packet."""
from collections import Counter
from fractions import Fraction as F
import numpy as np
import continuous_frontier_prediction_v1 as model
import audit_retired_region_join_v1 as search_audit
from audit_new_task_deadline_risk_v1 import readout_check
from test_deadline_risk_session_v1 import kernel_reference


def check_bounds(bounds, exact):
    assert bounds.shape == (len(exact), 2)
    for (lo, hi), (a, b) in zip(bounds, exact):
        lo, hi, a, b = float(lo), float(hi), F(a), F(b)
        assert F(lo) <= a <= b <= F(hi)
        assert F(float(np.nextafter(lo, np.inf))) > a
        assert F(float(np.nextafter(hi, -np.inf))) < b


def audit_call(arrays, metadata, events):
    a = arrays; m = model.reader.serializable(metadata); counts = Counter()
    x, v, q = a['x'], a['v'], a['q']
    assert len(q) == 257 and not m['query_targets_accessed'] and not m['archived_search_used']
    assert not m['global_bp_used'] and m['returned_array_bytes'] == sum(z.nbytes for z in a.values())
    np.testing.assert_allclose(a['fallback_prediction'], kernel_reference(x, v, q, 4096), rtol=1e-10, atol=1e-10)
    counts['independent_prior_kernel_replays'] += 1
    cheap = model.ranges.core.lipschitz_intervals(x, v, q)
    assert model.reader.serializable(cheap) == m['cheap_intervals']
    check_bounds(a['cheap_bounds'], cheap); counts['outward_intervals'] += len(q)
    cp = np.clip(a['fallback_prediction'], a['cheap_bounds'][:, 0], a['cheap_bounds'][:, 1])
    np.testing.assert_array_equal(a['cheap_prediction'], cp)
    expected = [('fallback', a['fallback_prediction']), ('fallback', cp)]
    if m['method'] != 'cheap_only':
        sa = {k[7:]: z for k, z in a.items() if k.startswith('search_')}
        assert np.array_equal(sa['original_x'], x) and np.array_equal(sa['original_v'], v)
        before = model.registry_identity()
        with model.registered():
            counts.update(search_audit.audit_case(sa, m['search'], [], exact=True))
        assert model.registry_identity() == before
        if m['readout'] is not None:
            ra = {k[8:]: z for k, z in a.items() if k.startswith('readout_')}
            counts.update(readout_check(ra, m['readout']))
        if m['branch'] == 'completed_posterior':
            assert m['search']['completed'] and m['readout']['full_candidate_set_resolved']
            final = np.clip(a['readout_prediction'], a['cheap_bounds'][:, 0], a['cheap_bounds'][:, 1])
            counts['completed_posterior_calls'] += 1
        else:
            assert m['branch'] == 'frontier'
            # Frozen whole-query reference is separate from the new streaming loop.
            reference = model.ranges.fit(sa, m['search'], q)
            for key in ['coordinate_boxes', 'empty_cells', 'cell_query_ranges', 'frontier_intervals',
                        'cheap_intervals', 'intersected_intervals', 'crossing_operations']:
                assert model.reader.serializable(reference[key]) == m['range_readout'][key], key
            counts['exact_full_query_range_replays'] += len(q)
            check_bounds(a['range_bounds'], reference['intersected_intervals'])
            counts['outward_intervals'] += len(q)
            order = a['query_order']; assert sorted(order.tolist()) == list(range(len(q)))
            np.testing.assert_array_equal(order, model.query_order(q))
            current = cp.copy(); seen = []
            for checkpoint in m['range_readout']['checkpoints']:
                ids = np.asarray(checkpoint['indices'], dtype=int)
                seen.extend(ids.tolist()); assert seen == order[:len(seen)].tolist()
                assert checkpoint['processed'] == len(seen) and len(ids) <= m['batch_queries']
                current[ids] = np.clip(cp[ids], a['range_bounds'][ids, 0], a['range_bounds'][ids, 1])
                expected.append(('fallback', current.copy()))
            assert len(seen) == len(q)
            final = current; counts['frontier_calls'] += 1
    else:
        assert m['search'] is None and m['readout'] is None and m['range_readout'] is None
        final = cp; counts['cheap_only_calls'] += 1
    np.testing.assert_array_equal(a['prediction'], final)
    if events and events[-1]['kind'] == 'final':
        expected.append(('final', final))
    assert len(events) == len(expected), (len(events), len(expected))
    for event, (kind, pred) in zip(events, expected):
        assert event['kind'] == kind and event['prediction'].shape == (257,)
        assert np.isfinite(event['prediction']).all()
        np.testing.assert_array_equal(event['prediction'], pred)
        counts['full_vector_packet_replays'] += 1
    counts['audited_continuous_calls'] += 1
    return counts
