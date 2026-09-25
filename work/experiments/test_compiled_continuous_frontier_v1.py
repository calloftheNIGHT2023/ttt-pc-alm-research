"""425 continuous exact equivalence, independent replay and actual lifecycles."""
from collections import Counter
from pathlib import Path
import os
import time
import traceback
import numpy as np
import compiled_continuous_frontier_v1 as model
import frontier_deadline_worker_v3 as worker
import frontier_search_receipt_v1 as receipt
import audit_continuous_frontier_v1 as auditor
import deadline_risk_io_v1 as io
from deadline_prediction_worker_v1 import choose

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/compiled_continuous_frontier/preflight_v1'
DESIGN = 'outputs/ttt-pc-alm-research/425_compiled_continuous_frontier_protocol_v1.md'


def direct(folder, function, data, seed, n, method, **kwargs):
    folder.mkdir(); events = []
    def emit(kind, prediction):
        events.append(dict(kind=kind, prediction=prediction.copy()))
    a, m = function(data['x'][:n], data['v'][:n], data['q'], seed=seed, method=method, emit=emit, **kwargs)
    np.savez_compressed(folder/'arrays.npz', **a)
    io.save(folder/'metadata.json', model.original.reader.serializable(m))
    io.save(folder/'events.json', model.original.reader.serializable(events))
    return a, m, events


def main():
    begin = time.perf_counter(); checks = Counter(); sessions = []; worker_rows = []; direct_rows = []
    for stage in ['preflight_v1', 'component_v1']:
        folder = ROOT/'results/piecewise_frontier'/stage; gate = io.read(folder/'summary.json')
        assert gate['passed'] and not (folder/'failure.json').exists()
        if 'source_sha256' in gate:
            io.verify_hashes(ROOT, gate['source_sha256'])
        for name, digest in gate['outputs_sha256'].items():
            assert io.sha(folder/name) == digest
    sources = {p.relative_to(ROOT).as_posix(): io.sha(p) for p in
               sorted((ROOT/'work/experiments').glob('*.py'))+[ROOT/DESIGN]}
    inputs = ROOT/'results/frontier_deadline/development_v1/inputs'
    manifest = io.read(inputs.parent/'inputs_manifest.json'); data = {}
    for seed in [5920000, 5920001]:
        name = f'{seed}.npz'; assert io.sha(inputs/name) == manifest[name]
        data[seed] = io.load_arrays(inputs/name); assert set(data[seed]) == {'x', 'v', 'q'}
    deterministic = dict(search_seconds=20., max_expanded=128)
    io.save(OUT/'protocol.json', dict(source_sha256=sources, input_sha256=manifest,
        methods=model.original.METHODS, stages=[4, 8, 16, 24], fixed_work=deterministic,
        query_targets_accessed=False, functional_cutoff_not_performance_comparison=True))
    before = model.original.stream_ranges; registry = model.original.registry_identity()
    try:
        with model.installed():
            assert model.original.stream_ranges is model.stream_ranges
            raise RuntimeError('scope restoration sentinel')
    except RuntimeError as error:
        assert str(error) == 'scope restoration sentinel'
    assert model.original.stream_ranges is before; checks['scope_restored_after_exception'] += 1
    for method in model.original.METHODS:
        for n in [4, 8, 16, 24]:
            results = []
            for label, function in [('original', model.original.fit), ('compiled', model.fit)]:
                results.append(direct(OUT/f'{method}_n{n}_{label}', function, data[5920000], 5920000, n, method, **deterministic))
            left, right = results
            assert left[0].keys() == right[0].keys()
            for key, value in left[0].items():
                np.testing.assert_array_equal(value, right[0][key]); checks['identical_continuous_arrays'] += 1
            assert len(left[2]) == len(right[2])
            for a, b in zip(left[2], right[2]):
                assert a['kind'] == b['kind']; np.testing.assert_array_equal(a['prediction'], b['prediction'])
                checks['identical_continuous_packets'] += 1
            checks.update(auditor.audit_call(*right))
            direct_rows.append(dict(method=method, n=n, branch=right[1]['branch']))
        print(dict(stage='compiled_direct_checked', method=method), flush=True)
    assert checks['frontier_calls'] > 0 and checks['cheap_only_calls'] == 4
    # Exercise the completed-search branch independently of the fixed small work cap.
    complete = direct(OUT/'completed_branch', model.fit, data[5920000], 5920000, 4,
                      'regional_active512', search_seconds=20., max_expanded=65536)
    assert complete[1]['branch'] == 'completed_posterior'
    checks.update(auditor.audit_call(*complete))
    for method in model.original.METHODS:
        cfg = dict(name='compiled_'+method, kind='compiled_frontier_receipt', method=method, **deterministic)
        session = worker.Session(cfg, ROOT); repeat = []
        jobs = [(5920000, 20.)]
        if method == 'regional_active512':
            jobs = [(5920000, 20.), (5920001, 20.), (5920000, 20.), (5920000, 1e-6), (5920000, 20.)]
        try:
            for j, (seed, budget) in enumerate(jobs):
                d = data[seed]; result = session.run(d['x'], d['v'], d['q'], seed=seed, budget=budget)
                pred, meta, events, archive = result
                worker_rows.append(io.save_call(ROOT, OUT/f'worker_{method}_{j}', *result))
                assert meta['error'] is None, meta
                if budget < 1:
                    assert meta['selected'] == 'constant' and meta['terminated_owned_worker'] and archive is None
                    checks['tiny_deadline_owned_termination'] += 1
                else:
                    assert meta['selected'] == 'final' and archive is not None
                    checks.update(auditor.audit_call(archive['arrays'], archive['metadata'], events))
                    if any('frontier_receipt' in e for e in events):
                        checks.update(receipt.audit_events(d['x'], d['v'], d['q'], events))
                    if seed == 5920000:
                        repeat.append(archive['arrays'])
                checks['actual_new_worker_calls'] += 1
        finally:
            lifecycle = session.close(); sessions.append(dict(method=method, **lifecycle))
        if method == 'regional_active512':
            assert len(lifecycle['setups']) == len(lifecycle['closures']) == 2
            for a in repeat[1:]:
                for key, value in repeat[0].items():
                    np.testing.assert_array_equal(value, a[key]); checks['worker_a_b_a_recreation_arrays'] += 1
        print(dict(stage='compiled_worker_checked', method=method), flush=True)
    # Batch-one delivery lengthens the receipt window for a functional hard kill.
    # Calibration never selects a model or a performance-comparison budget.
    cfg = dict(name='compiled_cutoff', kind='compiled_frontier_receipt', method='regional_active512',
               batch_queries=1, **deterministic)
    session = worker.Session(cfg, ROOT); d = data[5920000]
    try:
        full = session.run(d['x'], d['v'], d['q'], seed=5920000, budget=20.)
        worker_rows.append(io.save_call(ROOT, OUT/'cutoff_calibration', *full))
        assert full[1]['selected'] == 'final' and full[1]['error'] is None
        checks.update(receipt.audit_events(d['x'], d['v'], d['q'], full[2]))
        first = next(e['received_seconds'] for e in full[2] if 'frontier_receipt' in e)
        final = full[2][-1]['received_seconds']; assert final > first
        budgets = [first+t*(final-first) for t in [.5, .7, .3]]
        io.save(OUT/'functional_cutoffs.json', dict(first=first, final=final, budgets=budgets, performance_comparison=False))
        interrupted = False
        for j, budget in enumerate(budgets):
            result = session.run(d['x'], d['v'], d['q'], seed=5920000, budget=budget)
            pred, meta, events, archive = result
            worker_rows.append(io.save_call(ROOT, OUT/f'cutoff_{j}', *result)); assert meta['error'] is None
            if meta['terminated_owned_worker'] and any('frontier_receipt' in e and e['received_seconds'] <= budget for e in events):
                assert archive is None and not meta['received_final'] and meta['selected'] == 'fallback'
                checks.update(receipt.audit_events(d['x'], d['v'], d['q'], events))
                expected, selected = choose(events, budget, d['q'])
                np.testing.assert_array_equal(pred, expected); assert selected == meta['selected']
                checks['interrupted_receipts_replayed_without_archive'] += 1; interrupted = True
                print(dict(stage='compiled_cutoff_replayed', processed=events[-1]['frontier_receipt']['processed'], budget=budget), flush=True)
                break
        assert interrupted, 'Actual interrupted receipt path was not covered'
        again = session.run(d['x'], d['v'], d['q'], seed=5920000, budget=20.)
        worker_rows.append(io.save_call(ROOT, OUT/'cutoff_recreated', *again))
        assert again[1]['selected'] == 'final' and again[1]['generation'] == 0
        np.testing.assert_array_equal(full[0], again[0]); checks['cutoff_recreated_prediction'] += 1
    finally:
        sessions.append(dict(method=cfg['name'], **session.close()))
    import prefix_matched_controls_v1 as controls
    configs = [dict(name='prior4096_ridge', kind='regression', method='prior4096_ridge'),
        dict(name='adam_gn64_256_anytime', kind='portfolio'),
        next(c for c in controls.cohort_configs(ROOT) if c['name'] == 'official_ttt_native_prior256_32_p4')]
    for cfg in configs:
        results = []
        for label, cls in [('old418', worker.old.Session), ('compiled425', worker.Session)]:
            session = cls(cfg, ROOT)
            try:
                result = session.run(d['x'][:4], d['v'][:4], d['q'], seed=5920000, budget=20.)
                worker_rows.append(io.save_call(ROOT, OUT/f"route_{cfg['name']}_{label}", *result))
                assert result[1]['selected'] == 'final' and result[1]['error'] is None
                results.append(result)
            finally:
                sessions.append(dict(method=cfg['name'], route=label, **session.close()))
        left, right = results; assert left[3]['arrays'].keys() == right[3]['arrays'].keys()
        for key, value in left[3]['arrays'].items():
            np.testing.assert_array_equal(value, right[3]['arrays'][key]); checks['unchanged_baseline_arrays'] += 1
        assert len(left[2]) == len(right[2])
        for a, b in zip(left[2], right[2]):
            assert a['kind'] == b['kind']; np.testing.assert_array_equal(a['prediction'], b['prediction'])
            checks['unchanged_baseline_packets'] += 1
        checks['unchanged_baseline_routes'] += 1
        print(dict(stage='compiled_baseline_route_checked', method=cfg['name']), flush=True)
    assert worker.Session.run is worker.old.Session.run
    assert model.original.stream_ranges is before and model.original.registry_identity() == registry
    io.verify_hashes(ROOT, sources)
    io.save(OUT/'sessions.json', sessions); io.save(OUT/'direct_rows.json', direct_rows); io.save(OUT/'worker_rows.json', worker_rows)
    files = {p.relative_to(OUT).as_posix(): io.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    summary = dict(passed=True, checks=dict(checks), source_sha256=sources, outputs_sha256=files,
        seconds=time.perf_counter()-begin, query_targets_accessed=False, performance_comparison=False,
        independent_confirmation=False, core_research_goal_complete=False)
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
