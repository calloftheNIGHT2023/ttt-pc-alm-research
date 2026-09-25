"""418 actual interrupted workers with auditable pre-deadline predictions."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import os
import time
import traceback
import numpy as np
import frontier_deadline_worker_v2 as worker
import frontier_search_receipt_v1 as receipt
import audit_continuous_frontier_v1 as audit
import deadline_risk_io_v1 as io
from deadline_prediction_worker_v1 import choose

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/continuous_frontier/receipt_preflight_v1'
DESIGN = 'outputs/ttt-pc-alm-research/418_frontier_receipt_protocol_v1.md'


def main():
    begin = time.perf_counter(); counts = Counter(); rows = []; sessions = []
    prior = io.read(ROOT/'results/continuous_frontier/preflight_v1/summary.json')
    assert prior['passed']; io.verify_hashes(ROOT, prior['source_sha256'])
    sources = {p.relative_to(ROOT).as_posix(): io.sha(p) for p in
               sorted((ROOT/'work/experiments').glob('*.py'))+[ROOT/DESIGN]}
    source = ROOT/'results/runtime_matched_prefix/development_v1/inputs/5920000.npz'
    manifest = io.read(source.parent.parent/'inputs_manifest.json')
    assert io.sha(source) == manifest[source.name]
    d = io.load_arrays(source); assert set(d) == {'x', 'v', 'q'}
    x, v, q = d['x'], d['v'], d['q']
    io.save(OUT/'protocol.json', dict(source_sha256=sources, input_sha256=io.sha(source), seed=5920000,
        methods=receipt.model.METHODS[:-1], queries=257, support=24, query_targets_accessed=False,
        cutoff_calibration_for_functional_test_only=True, performance_comparison=False))
    tampered = False
    for method in receipt.model.METHODS[:-1]:
        cfg = dict(kind='frontier_receipt', name='receipt_'+method, method=method, search_seconds=.25)
        session = worker.Session(cfg, ROOT)
        try:
            result = session.run(x, v, q, seed=5920000, budget=20.)
            pred, m, events, archive = result
            rows.append(io.save_call(ROOT, OUT/method, *result))
            assert m['selected'] == 'final' and m['error'] is None and archive is not None, m
            counts.update(audit.audit_call(archive['arrays'], archive['metadata'], events))
            if any('frontier_receipt' in e for e in events):
                one = receipt.audit_events(x, v, q, events); counts.update(one)
                io.save(OUT/method/'receipt_audit.json', dict(one))
                if not tampered:
                    bad = dict(next(e['frontier_receipt'] for e in events if 'frontier_receipt' in e), arrays_sha256='0'*64)
                    try:
                        receipt.replay_search(x, v, bad)
                    except AssertionError:
                        counts['tampered_search_hash_rejected'] += 1
                    else:
                        raise AssertionError('Tampered search hash was accepted')
                    altered = deepcopy(events)
                    event = next(e for e in altered if 'frontier_receipt' in e)
                    event['prediction'][0] = np.nextafter(event['prediction'][0], np.inf)
                    try:
                        receipt.audit_events(x, v, q, altered)
                    except AssertionError:
                        counts['tampered_prediction_rejected'] += 1
                    else:
                        raise AssertionError('Tampered intermediate prediction was accepted')
                    tampered = True
            else:
                assert archive['metadata']['branch'] == 'completed_posterior'
                counts['completed_without_frontier'] += 1
        finally:
            sessions.append(dict(method=method, **session.close()))
        print(dict(stage='receipt_replayed', method=method), flush=True)
    assert tampered and counts['cut_array_hash_replays'] > 0
    # Fixed-work call gives a long enough frontier interval to exercise an actual
    # controller cutoff, with no performance/risk inference from calibrated times.
    cfg = dict(kind='frontier_receipt', name='receipt_interruption_test', method='regional_active512',
               search_seconds=20., max_expanded=128)
    session = worker.Session(cfg, ROOT)
    try:
        full = session.run(x, v, q, seed=5920000, budget=20.)
        rows.append(io.save_call(ROOT, OUT/'cutoff_calibration', *full))
        assert full[1]['selected'] == 'final' and full[1]['error'] is None
        first = next(e['received_seconds'] for e in full[2] if 'frontier_receipt' in e)
        final = full[2][-1]['received_seconds']; assert final > first
        counts.update(receipt.audit_events(x, v, q, full[2]))
        budgets = [first+t*(final-first) for t in [.5, .7, .3]]
        io.save(OUT/'functional_cutoffs.json', dict(first_range_seconds=first, final_seconds=final,
            tested_budget_candidates=budgets, not_a_performance_comparison=True))
        interrupted = False
        for i, budget in enumerate(budgets):
            result = session.run(x, v, q, seed=5920000, budget=budget)
            pred, m, events, archive = result
            rows.append(io.save_call(ROOT, OUT/f'interrupted_{i}', *result))
            assert m['error'] is None, m
            if m['terminated_owned_worker'] and any('frontier_receipt' in e and e['received_seconds'] <= budget for e in events):
                assert archive is None and not m['received_final'] and m['selected'] == 'fallback'
                counts.update(receipt.audit_events(x, v, q, events))
                expected, selected = choose(events, budget, q)
                np.testing.assert_array_equal(pred, expected); assert selected == m['selected']
                counts['interrupted_frontier_rebuilt_without_archive'] += 1
                interrupted = True
                print(dict(stage='interrupted_frontier_audited', budget=budget,
                    delivered_queries=events[-1].get('frontier_receipt', {}).get('processed'),
                    archive_available=False), flush=True)
                break
        assert interrupted, 'Functional interruption not exercised; do not claim this test passed'
        again = session.run(x, v, q, seed=5920000, budget=20.)
        rows.append(io.save_call(ROOT, OUT/'recreated', *again))
        assert again[1]['selected'] == 'final' and again[1]['generation'] == 0
        np.testing.assert_array_equal(full[0], again[0])
        counts.update(receipt.audit_events(x, v, q, again[2]))
        counts['post_interruption_recreated_prediction'] += 1
    finally:
        sessions.append(dict(method=cfg['name'], **session.close()))
    assert worker.Session.run is worker.old.Session.run
    io.verify_hashes(ROOT, sources); io.save(OUT/'rows.json', rows); io.save(OUT/'sessions.json', sessions)
    files = {p.relative_to(OUT).as_posix(): io.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    summary = dict(passed=True, checks=dict(counts), seconds=time.perf_counter()-begin,
        source_sha256=sources, outputs_sha256=files, query_targets_accessed=False,
        performance_comparison=False, independent_confirmation=False, core_research_goal_complete=False)
    io.save(OUT/'summary.json', summary)
    print({k: v for k, v in summary.items() if k not in ['source_sha256', 'outputs_sha256']}, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except Exception:
        io.save(OUT/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
