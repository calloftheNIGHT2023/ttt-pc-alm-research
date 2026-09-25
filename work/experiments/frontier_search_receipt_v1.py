"""418 replayable cut receipts; no scientific search or readout modification."""
from collections import Counter
from contextlib import contextmanager
import hashlib
import time
import numpy as np
import continuous_frontier_prediction_v1 as model


def arrays_digest(arrays):
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        assert isinstance(value, np.ndarray) and value.dtype != object
        digest.update(key.encode()+b'\0'+value.dtype.str.encode()+b'\0')
        digest.update(str(value.shape).encode()+b'\0'+value.tobytes(order='C'))
    return digest.hexdigest()


class Connection:
    def __init__(self, original):
        self.original = original
        self.receipt = None
        self.processed = 0

    def send(self, packet):
        kind = packet.get('kind')
        if kind in ['fallback', 'final'] and self.receipt is not None:
            if packet['kind'] == 'fallback':
                self.processed = min(self.receipt['queries'], self.processed+self.receipt['batch_queries'])
            else:
                assert self.processed == self.receipt['queries']
            packet = dict(packet, frontier_receipt=dict(self.receipt, processed=self.processed))
        self.original.send(packet)
        if kind == 'ready':
            self.receipt = None
            self.processed = 0

    def recv(self):
        return self.original.recv()

    def close(self):
        return self.original.close()


@contextmanager
def observe(connection):
    previous = model.stream_ranges
    def stream(sa, sm, q, cheap, prediction, *, batch_queries=32, emit=None):
        tick = time.perf_counter()
        receipt = dict(version=1, method=sm['method'], schedule=sm['schedule'], ordering=sm['order_name'],
            expanded=int(sm['expanded']), original_max_expanded=int(sm['max_expanded']),
            max_states=int(sm['max_states']), completed=bool(sm['completed']),
            arrays_sha256=arrays_digest(sa), queries=len(q), batch_queries=batch_queries)
        receipt['online_hash_seconds'] = time.perf_counter()-tick
        assert connection.receipt is None
        connection.receipt = receipt
        result = previous(sa, sm, q, cheap, prediction, batch_queries=batch_queries, emit=emit)
        result[3]['search_replay_receipt'] = receipt
        return result
    model.stream_ranges = stream
    try:
        yield
    finally:
        model.stream_ranges = previous


def replay_search(x, v, receipt):
    assert receipt['version'] == 1 and receipt['method'] in model.METHODS[:-1]
    assert receipt['schedule'] == 'dfs' and receipt['ordering'] == 'farthest_x'
    assert 0 <= receipt['expanded'] <= receipt['original_max_expanded']
    with model.registered():
        sa, sm = model.search.join(x, v, name=receipt['method'], schedule='dfs', order_name='farthest_x',
            max_expanded=receipt['expanded'], max_states=receipt['max_states'], max_seconds=3600.)
    assert sm['expanded'] == receipt['expanded'] and sm['completed'] == receipt['completed']
    assert arrays_digest(sa) == receipt['arrays_sha256'], 'Cut replay is not the delivered search state'
    return sa, sm


def audit_events(x, v, q, events):
    import audit_retired_region_join_v1 as checker
    from audit_continuous_frontier_v1 import check_bounds
    from test_deadline_risk_session_v1 import kernel_reference
    counts = Counter(); receipts = [e['frontier_receipt'] for e in events if 'frontier_receipt' in e]
    assert receipts, 'No frontier receipt delivered; do not claim partial-proof audit'
    assert len(q) == 257
    base = receipts[0]
    for receipt in receipts:
        assert {k: v for k, v in receipt.items() if k != 'processed'} == {k: v for k, v in base.items() if k != 'processed'}
        assert receipt['queries'] == len(q) and 0 < receipt['processed'] <= len(q)
    sa, sm = replay_search(x, v, base)
    with model.registered():
        counts.update(checker.audit_case(sa, sm, [], exact=True))
    counts['cut_array_hash_replays'] += 1
    ref = model.ranges.fit(sa, sm, q)
    counts['exact_query_intervals_replayed'] += len(q)
    cheap_bounds = model.outward(ref['cheap_intervals'])
    bounds = model.outward(ref['intersected_intervals'])
    check_bounds(cheap_bounds, ref['cheap_intervals']); check_bounds(bounds, ref['intersected_intervals'])
    first = events[0]; assert first['kind'] == 'fallback' and 'frontier_receipt' not in first
    np.testing.assert_allclose(first['prediction'], kernel_reference(x, v, q, 4096), rtol=1e-10, atol=1e-10)
    cheap = np.clip(first['prediction'], cheap_bounds[:, 0], cheap_bounds[:, 1])
    assert events[1]['kind'] == 'fallback' and 'frontier_receipt' not in events[1]
    np.testing.assert_array_equal(events[1]['prediction'], cheap)
    order = model.query_order(q); previous = 0
    for event in events[2:]:
        if event['kind'] == 'error':
            raise AssertionError(event)
        rr = event['frontier_receipt']; count = rr['processed']
        if event['kind'] == 'fallback':
            assert count == min(len(q), previous+base['batch_queries'])
        else:
            assert event['kind'] == 'final' and count == previous == len(q)
        p = cheap.copy(); ids = order[:count]
        p[ids] = np.clip(cheap[ids], bounds[ids, 0], bounds[ids, 1])
        np.testing.assert_array_equal(event['prediction'], p)
        previous = count; counts['receipt_packet_predictions_replayed'] += 1
    return counts
