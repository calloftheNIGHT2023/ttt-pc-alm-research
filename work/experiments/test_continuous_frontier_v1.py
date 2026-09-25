"""417 support-only preflight: rational guarantees, live workers and old routing."""
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import os
import time
import traceback
import numpy as np
import continuous_frontier_prediction_v1 as model
import frontier_deadline_worker_v1 as worker
import audit_continuous_frontier_v1 as auditor
import deadline_risk_io_v1 as io

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'results/continuous_frontier'
OUT = BASE/'preflight_v1'
DESIGN = 'outputs/ttt-pc-alm-research/417_continuous_frontier_protocol_v1.md'


def save(path, value):
    io.save(path, model.reader.serializable(value))


def direct(directory, x, v, q, *, seed, method, **kwargs):
    events = []
    a, m = model.fit(x, v, q, seed=seed, method=method,
        emit=lambda kind, p: events.append(dict(kind=kind, prediction=p.copy())), **kwargs)
    directory.mkdir()
    np.savez_compressed(directory/'arrays.npz', **a)
    save(directory/'metadata.json', m)
    np.savez_compressed(directory/'packets.npz', **{f'p{i}': e['prediction'] for i, e in enumerate(events)})
    save(directory/'events.json', [dict(kind=e['kind']) for e in events])
    return a, m, events


def main():
    begin = time.perf_counter(); checks = Counter(); rng = np.random.default_rng(417731)
    assert io.read(ROOT/'results/frontier_range_certificate/preflight_v1/summary.json')['passed']
    sources = {p.relative_to(ROOT).as_posix(): io.sha(p) for p in
               sorted((ROOT/'work/experiments').glob('*.py'))+[ROOT/DESIGN]}
    old = ROOT/'results/runtime_matched_prefix/development_v1'
    manifest = io.read(old/'inputs_manifest.json'); data = {}
    for seed in [5920000, 5920001]:
        path = old/'inputs'/f'{seed}.npz'; assert io.sha(path) == manifest[path.name]
        data[seed] = io.load_arrays(path); assert set(data[seed]) == {'x', 'v', 'q'}
    save(OUT/'protocol.json', dict(source_sha256=sources, methods=model.METHODS, stages=[4, 8, 16, 24],
        inputs_sha256={str(seed): manifest[f'{seed}.npz'] for seed in data}, queries=257,
        query_targets_accessed=False, performance_comparison=False, independent_confirmation=False))
    intervals = [(F(0), F(0)), (F(1), F(1)), (F(1, 3), F(1, 3)),
                 (F(1, 2**1076), F(3, 2**1076)), (F(1)-F(1, 2**60), F(1))]
    for _ in range(128):
        intervals.append(tuple(sorted(F(int(i), 100003) for i in rng.integers(0, 100004, 2))))
    rounded = model.outward(intervals); auditor.check_bounds(rounded, intervals)
    checks['outward_boundary_intervals'] += len(intervals)
    for (low, high), (a, b) in zip(rounded, intervals):
        for r in [0., 1., float(a), float(b), float((a+b)/2)]:
            p = float(np.clip(r, low, high))
            for truth in [a, b, (a+b)/2]:
                assert (F(r)-truth)**2-(F(p)-truth)**2 >= (F(r)-F(p))**2
                checks['exact_projection_loss_inequalities'] += 1
    before = model.registry_identity()
    try:
        with model.registered():
            raise RuntimeError('test restoration')
    except RuntimeError:
        pass
    assert model.registry_identity() == before; checks['exception_registry_restoration'] += 1
    pending_audits = []; records = []
    for method in model.METHODS:
        for n in [4, 8, 16, 24]:
            d = data[5920000]; folder = OUT/f'direct_{method}_n{n}'
            a, m, e = direct(folder, d['x'][:n], d['v'][:n], d['q'], seed=5920000, method=method)
            pending_audits.append((folder, a, m, e))
            row = dict(method=method, n=n, branch=m['branch'], seconds=m['contiguous_seconds'],
                       packets=len(e), directory=folder.relative_to(ROOT).as_posix())
            records.append(row); print(dict(stage='continuous', **row), flush=True)
    save(OUT/'direct_rows.json', records)
    # Audits run after all diagnostic continuous calls; not interleaved as timing evidence.
    for folder, a, m, e in pending_audits:
        one = auditor.audit_call(a, m, e); checks.update(one); save(folder/'audit.json', dict(one))
    print(dict(stage='direct_audit_complete', calls=len(pending_audits)), flush=True)
    # The no-accumulation control must really keep accumulated multipliers zero.
    found = False
    for _, a, m, _ in pending_audits:
        if m['search'] is None:
            continue
        for i, row in enumerate(m['search']['batches']):
            prefix = f'search_batch_{i}_'; eligible = ~(a[prefix+'c5'] | a[prefix+'c20'])
            if int(eligible.sum()) < 64:
                continue
            k = row['k']; regs = a[prefix+'regs'][eligible][:64]
            aa, mm = model.search.retirement.solve(a['search_x_observed'][:k], a['search_v_observed'][:k],
                                                   regs, family='regional_instant', steps=512)
            assert not np.any(aa['stopped_p']) and not np.any(aa['stopped_a'])
            found = True; checks['instant_zero_accumulated_multiplier_checks'] += 1
            break
        if found:
            break
    assert found
    # Fixed work caps, not wall time, are needed for deterministic isolation tests.
    deterministic = dict(search_seconds=20., max_expanded=1024)
    for method in model.METHODS:
        d = data[5920000]; other = data[5920001]; q = d['q']
        aa, am = model.fit(d['x'], d['v'], q, seed=5920000, method=method, **deterministic)
        model.fit(other['x'], other['v'], q, seed=5920001, method=method, **deterministic)
        again, _ = model.fit(d['x'], d['v'], q, seed=5920000, method=method, **deterministic)
        reverse, _ = model.fit(d['x'], d['v'], q[::-1], seed=5920000, method=method, **deterministic)
        for key in aa:
            np.testing.assert_array_equal(aa[key], again[key]); checks['a_b_a_arrays'] += 1
        # Prior-kernel BLAS paths can differ by roundoff when query chunks reverse.
        # Exact certificate endpoints and all support-only search arrays must agree.
        np.testing.assert_allclose(aa['prediction'], reverse['prediction'][::-1], rtol=0, atol=1e-12)
        for key in ['cheap_bounds', 'range_bounds']:
            if key in aa:
                np.testing.assert_array_equal(aa[key], reverse[key][::-1])
                checks['query_permutation_exact_bound_arrays'] += 1
        for key in aa:
            if key.startswith('search_'):
                np.testing.assert_array_equal(aa[key], reverse[key]); checks['query_independent_search_arrays'] += 1
        checks['query_permutation_predictions'] += 1
    # Actual workers for every new family. The active family additionally does A/B/A,
    # timeout of its own worker, and clean recreation with deterministic work.
    sessions = []
    for method in model.METHODS:
        cfg = dict(kind='frontier', name='frontier_'+method, method=method, **deterministic)
        session = worker.Session(cfg, ROOT); predictions = []
        jobs = [(5920000, 20.)]
        if method == 'regional_active512':
            jobs = [(5920000, 20.), (5920001, 20.), (5920000, 20.), (5920000, 1e-6), (5920000, 20.)]
        try:
            for j, (seed, budget) in enumerate(jobs):
                d = data[seed]; result = session.run(d['x'], d['v'], d['q'], seed=seed, budget=budget)
                pred, meta, events, archive = result
                io.save_call(ROOT, OUT/f'worker_{method}_{j}', *result)
                assert meta['error'] is None, meta
                if budget < 1:
                    assert meta['selected'] == 'constant' and meta['terminated_owned_worker'] and archive is None
                    checks['tiny_deadline_owned_worker_termination'] += 1
                else:
                    assert meta['selected'] == 'final' and archive is not None
                    np.testing.assert_array_equal(pred, archive['arrays']['prediction'])
                    checks.update(auditor.audit_call(archive['arrays'], archive['metadata'], events))
                    if seed == 5920000:
                        predictions.append(pred)
                checks['actual_new_worker_calls'] += 1
        finally:
            lifecycle = session.close(); sessions.append(dict(method=method, **lifecycle))
        if method == 'regional_active512':
            assert len(lifecycle['setups']) == len(lifecycle['closures']) == 2
            for p in predictions[1:]:
                np.testing.assert_array_equal(p, predictions[0]); checks['worker_repeat_recreation_predictions'] += 1
        print(dict(stage='worker_checked', method=method), flush=True)
    # Matched routing of regression, official TTT and the strongest BP portfolio.
    import prefix_matched_controls_v1 as controls
    configs = [dict(name='prior4096_ridge', kind='regression', method='prior4096_ridge'),
               dict(name='adam_gn64_256_anytime', kind='portfolio'),
               next(c for c in controls.cohort_configs(ROOT) if c['name'] == 'official_ttt_native_prior256_32_p4')]
    for cfg in configs:
        results = []; d = data[5920000]
        for label, cls in [('old402', worker.old.Session), ('new417', worker.Session)]:
            session = cls(cfg, ROOT)
            try:
                result = session.run(d['x'][:4], d['v'][:4], d['q'], seed=5920000, budget=20.)
                io.save_call(ROOT, OUT/f"route_{cfg['name']}_{label}", *result)
                assert result[1]['selected'] == 'final' and result[1]['error'] is None
                results.append(result)
            finally:
                sessions.append(dict(method=cfg['name'], route=label, **session.close()))
        left, right = results
        assert left[3]['arrays'].keys() == right[3]['arrays'].keys()
        for key, value in left[3]['arrays'].items():
            np.testing.assert_array_equal(value, right[3]['arrays'][key]); checks['unchanged_baseline_arrays'] += 1
        assert len(left[2]) == len(right[2])
        for a, b in zip(left[2], right[2]):
            assert a['kind'] == b['kind']; np.testing.assert_array_equal(a['prediction'], b['prediction'])
            checks['unchanged_baseline_packets'] += 1
        checks['unchanged_baseline_routes'] += 1
        print(dict(stage='baseline_route_checked', method=cfg['name']), flush=True)
    assert worker.Session.run is worker.old.Session.run
    assert model.registry_identity() == before
    save(OUT/'sessions.json', sessions); io.verify_hashes(ROOT, sources)
    files = {p.relative_to(OUT).as_posix(): io.sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()}
    summary = dict(passed=True, checks=dict(checks), seconds=time.perf_counter()-begin,
        query_targets_accessed=False, performance_comparison=False, independent_confirmation=False,
        core_research_goal_complete=False, source_sha256=sources, outputs_sha256=files)
    save(OUT/'summary.json', summary)
    print({k: v for k, v in summary.items() if k not in ['source_sha256', 'outputs_sha256']}, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        main()
    except Exception:
        save(OUT/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
        raise
