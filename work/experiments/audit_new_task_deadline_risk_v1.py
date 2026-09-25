"""387 independent receipt/deadline, state, proof and baseline replay; no truth."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import os
import time
import traceback
import numpy as np
import deadline_risk_io_v1 as io
import deadline_risk_registry_v1 as registry


def readout_check(a, m):
    import complete_credit_mode_geometry_v1 as classifier
    import candidate_set_readout_v1 as reader
    counts = Counter(); keys = sorted(r.tobytes().hex() for r in a['search_regions'])
    assert keys == m['classified_modes'] == sorted(m['classifications_detail'])
    kinds = Counter(); positive = []; unresolved = []
    for key in keys:
        note = m['classifications_detail'][key]; _, aa, rhs, _, _ = classifier.matrices(a['x'], a['v'], key)
        counts['exact_geometry_certificates'] += classifier.verify_certificates(aa, rhs, note)
        strict = any(c.get('strict_interior', False) for c in note['certificates'])
        infeasible = any(c['type'] == 'negative_constant_row' or c.get('conclusion') == 'infeasible' for c in note['certificates'])
        zero = any(c.get('conclusion') == 'zero_volume_or_empty' for c in note['certificates'])
        assert not (strict and (infeasible or zero))
        if note['classification'] == 'positive_volume': assert strict and note['positive_volume_certified']
        elif note['classification'] == 'infeasible': assert infeasible and not strict
        elif note['classification'] == 'zero_volume_or_empty': assert zero and not (strict or infeasible)
        else: assert note['classification'] == 'unresolved' and not (strict or infeasible or zero)
        if note['numerical_volume_available']:
            assert note['classification'] == 'positive_volume'; positive.append(key)
        if note['classification'] == 'unresolved' or (note['classification'] == 'positive_volume' and not note['numerical_volume_available']): unresolved.append(key)
        kinds[note['classification']] += 1
    assert positive == m['positive_modes'] and unresolved == m['unresolved_modes'] and dict(kinds) == m['classification_counts']
    assert m['readout_available'] == bool(positive)
    assert m['full_candidate_set_resolved'] == (m['search_completed'] and not unresolved)
    if positive:
        assert int(a['allocation'].sum()) == len(a['points']) == m['particles']
        np.testing.assert_array_equal(a['positive_volumes'], [m['classifications_detail'][k]['volume'] for k in positive])
        for p in a['points']:
            assert np.max(abs(reader.shared.geometry.base.forward(a['x'], p)-a['v'])) <= .001+1e-7
            counts['posterior_support_checks'] += 1
        pred = reader.project(reader.shared.geometry.make_predict(a['points'])(a['q']))
        np.testing.assert_array_equal(pred, a['prediction']); counts['posterior_prediction_replays'] += 1
    else: assert a['prediction'].size == a['points'].size == a['allocation'].size == 0
    assert m['returned_array_bytes'] == sum(v.nbytes for v in a.values())
    return counts


def audit(root, out):
    begin = time.perf_counter(); source = root/io.BASE/'development_v1'; summary = io.read(source/'summary.json')
    assert summary['passed'] and summary['all_predictions_sealed'] and not (source/'failure.json').exists()
    for p, h in summary['outputs_sha256'].items(): assert io.sha(source/p) == h
    protocol = io.read(source/'protocol.json')
    io.verify_hashes(root, protocol['source_sha256']); io.verify_hashes(root, protocol['pretrained_sha256'])
    configs = {c['name']:c for c in registry.configs()}; rows = io.read(source/'rows.json')
    assert protocol['configs'] == registry.configs() and protocol['seeds'] == registry.SEEDS and protocol['budgets'] == registry.BUDGETS
    assert len(rows) == 2304 and len({(r['seed'], r['method'], r['budget']) for r in rows}) == 2304
    expected = []
    for block, first in enumerate(range(0, 32, registry.BLOCK_SIZE)):
        for ci in np.random.default_rng(np.random.SeedSequence([387101, block])).permutation(36):
            cc = registry.configs()[int(ci)]; jobs = [(s, b) for s in registry.SEEDS[first:first+4] for b in registry.BUDGETS]
            for ji in np.random.default_rng(np.random.SeedSequence([387103, block, int(ci)])).permutation(8):
                s, b = jobs[int(ji)]; expected.append((block, int(ci), s, cc['name'], b))
    assert [(r['block'], r['config_index'], r['seed'], r['method'], r['budget']) for r in rows] == expected
    inputs = {}
    for name, h in io.read(source/'inputs_manifest.json').items():
        assert io.sha(source/'inputs'/name) == h; a = io.load_arrays(source/'inputs'/name); seed = int(Path(name).stem)
        x, v = io.observations(seed); np.testing.assert_array_equal(a['x'], x); np.testing.assert_array_equal(a['v'], v)
        np.testing.assert_array_equal(a['q'], np.linspace(0., 1., 257)); inputs[seed] = a
    import independent_hybrid_memory as regression
    import n24_meta_input_adapter_v1 as meta
    import n24_optimizer_controls_v1 as optimizer
    import audit_contiguous_regional_repeat_v1 as whole
    from test_deadline_risk_session_v1 import kernel_reference
    meta.torch.set_num_threads(1); meta.torch.set_num_interop_threads(1); models, model_manifest = meta.load(root)
    fallback = {}; controls = {}; banks = {}; counts = Counter(); records = []; by_session = defaultdict(list)
    for i, row in enumerate(rows):
        seed = row['seed']; cfg = configs[row['method']]; inp = inputs[seed]; x, v, q = inp['x'], inp['v'], inp['q']
        directory = root/row['directory']
        for p, h in row['files'].items(): assert io.sha(directory/p) == h
        output = io.load_arrays(directory/'outputs.npz'); metadata = io.read(directory/'metadata.json'); events = io.read(directory/'events.json')
        assert metadata['config'] == cfg and metadata['budget_seconds'] == row['budget'] and metadata['error'] is None
        assert not metadata['query_targets_accessed'] and not metadata['shared_scientific_task_state_reused']
        assert row['received_final'] == metadata['received_final'] and row['selected'] == metadata['selected']
        assert metadata['controller_overrun_seconds'] == max(0., metadata['decision_seconds']-row['budget'])
        assert row['online_seconds'] == metadata['decision_seconds']
        for name in ['setup_seconds', 'cleanup_seconds', 'task_cleanup_seconds', 'archive_seconds',
                     'controller_overrun_seconds', 'worker_max_sampled_rss', 'controller_max_sampled_rss', 'worker_pid', 'generation']:
            assert row[name] == metadata[name]
        assert all(e['generation'] == metadata['generation'] for e in events)
        assert all(a['received_seconds'] <= b['received_seconds'] for a, b in zip(events, events[1:]))
        assert all(0 <= e['worker_seconds'] <= e['received_seconds']+1e-3 for e in events)
        chosen = [(j, e) for j, e in enumerate(events) if e['kind'] in ['fallback', 'final'] and e['received_seconds'] <= row['budget']]
        if chosen:
            j, event = chosen[-1]; prediction = output[f'event_{j}_prediction']; selected = event['kind']
        else: prediction = np.full_like(q, .5); selected = 'constant'
        np.testing.assert_array_equal(prediction, output['prediction']); assert selected == metadata['selected']
        for key, pred in output.items():
            assert pred.shape == q.shape and np.isfinite(pred).all() and np.all((pred >= 0) & (pred <= 1))
        if seed not in fallback:
            predictor, _, _ = regression.regression(x, v, 'prior4096_ridge'); fallback[seed] = np.clip(predictor(q), 0., 1.)
        for j, event in enumerate(events):
            if event['kind'] == 'fallback':
                np.testing.assert_array_equal(output[f'event_{j}_prediction'], fallback[seed]); counts['paid_fallback_replays'] += 1
        resources = metadata['resource_samples']
        assert all(s['worker_rss'] > 0 and s['controller_rss'] > 0 and s['worker_cpu_seconds'] >= -1e-9 for s in resources)
        assert all(a['seconds'] <= b['seconds'] for a, b in zip(resources, resources[1:]))
        assert metadata['worker_max_sampled_rss'] == max((s['worker_rss'] for s in resources), default=metadata['setup_worker_memory']['rss'])
        checks = Counter()
        if metadata['received_final']:
            assert not metadata['terminated_owned_worker'] and metadata['model_fingerprint_verified']
            a = io.load_arrays(directory/'state.npz'); m = io.read(directory/'state.json')
            for k in ['x', 'v', 'q']: np.testing.assert_array_equal(a[k], inp[k])
            np.testing.assert_array_equal(a['prediction'], output[f'event_{len(events)-1}_prediction'])
            if cfg['kind'] != 'regression' or cfg['method'] in ['prior16384_ridge', 'prior65536_ridge']:
                np.testing.assert_array_equal(a['fallback_prediction'], fallback[seed]); assert m['fallback_seconds'] > 0
            if cfg['kind'] == 'regional':
                scientific = {k:value for k, value in a.items() if k.startswith(('search_', 'readout_'))}
                checks.update(whole.audit_canonical(scientific, m))
                full = m['full_candidate_set_resolved'] and m['readout_available']
                np.testing.assert_array_equal(a['prediction'], a['readout_prediction'] if full else fallback[seed])
                counts['complete_regional_deliveries'] += int(full and metadata['selected'] == 'final')
            elif cfg['kind'] == 'optimizer':
                key = seed, cfg['family'], cfg['steps'], cfg['restarts']
                if key not in banks:
                    with optimizer.prior_bounds():
                        if cfg['family'] in ['adam', 'gauss_newton']:
                            bank, _, _ = optimizer.old.run_bp(optimizer.starts(cfg['restarts']), x, v, cfg['family'], cfg['steps'], trace=False)
                        else: bank, _, _ = optimizer.old.run_local(optimizer.starts(cfg['restarts']), x, v, cfg['family'], cfg['steps'], trace=False)
                    banks[key] = bank
                np.testing.assert_array_equal(a['starts'], optimizer.starts(cfg['restarts']))
                np.testing.assert_array_equal(a['best_bank'], banks[key]); assert np.max(abs(a['best_bank'])) <= .12+1e-14
                index, point = optimizer.old.select(banks[key], x, v)
                assert index == m['selected_index']; np.testing.assert_array_equal(point, a['selected_point'])
                pp = np.clip(optimizer.old.base.forward(q, point), 0., 1.); np.testing.assert_array_equal(pp, a['point_prediction'])
                if cfg['readout'] == 'posterior_union':
                    ra = {k[8:]:value for k, value in a.items() if k.startswith('readout_')}
                    checks.update(readout_check(ra, m['readout_metadata']))
                    if m['readout_metadata']['readout_available']: pp = ra['prediction']
                np.testing.assert_array_equal(pp, a['prediction']); checks['optimizer_bank_replays'] += 1
            else:
                key = seed, cfg['name']
                if key not in controls:
                    if cfg['kind'] == 'meta':
                        state = None
                        for n in ([4, 8, 16, 24] if cfg['warm_trajectory'] else [24]):
                            predictor, state, _ = meta.fit(models[cfg['method']], x[:n], v[:n], state)
                        controls[key] = predictor(q)
                    elif cfg['method'].startswith('prior'):
                        controls[key] = kernel_reference(x, v, q, int(cfg['method'].split('_')[0][5:]))
                    elif cfg['method'] == 'residual_linear_ridge':
                        phi = np.column_stack([np.ones(len(x)), x]); target = v-regression.base.forward(x, np.zeros(4))
                        w = np.linalg.solve(phi.T@phi+1e-4*np.eye(2), phi.T@target)
                        controls[key] = np.clip(w[0]+q*w[1]+regression.base.forward(q, np.zeros(4)), 0., 1.)
                    else:
                        predictor, _, _ = regression.regression(x, v, cfg['method']); controls[key] = np.clip(predictor(q), 0., 1.)
                np.testing.assert_allclose(a['prediction'], controls[key], rtol=1e-10, atol=1e-10)
                checks['regression_or_meta_replays'] += 1
            records.append(dict(directory=row['directory'], checks=dict(checks))); counts.update(checks)
        else:
            assert metadata['terminated_owned_worker'] and 'state.npz' not in row['files'] and metadata['cleanup_seconds'] >= 0
        counts['independent_deadline_choices'] += 1; counts['process_resource_records'] += 1
        by_session[row['block'], row['method']].append(row)
        if (i+1) % 64 == 0: print(dict(stage='deadline_audit', calls=i+1, seconds=time.perf_counter()-begin), flush=True)
    sessions = io.read(source/'sessions.json'); assert len(sessions) == len(by_session) == 288
    for session in sessions:
        rr = by_session[session['block'], session['method']]; assert len(rr) == session['calls'] == 8
        assert rr[0]['generation'] == 0 and rr[0]['setup_seconds'] > 0
        for previous, current in zip(rr, rr[1:]):
            if previous['received_final']:
                assert current['generation'] == previous['generation']+1 and current['worker_pid'] == previous['worker_pid'] and current['setup_seconds'] == 0
            else: assert current['generation'] == 0 and current['setup_seconds'] > 0
        assert math.isclose(session['total_setup_seconds'], math.fsum(r['setup_seconds'] for r in rr), rel_tol=0, abs_tol=1e-8)
        assert session['total_closure_seconds']+1e-8 >= math.fsum(r['cleanup_seconds'] for r in rr)
        counts['reusable_session_lifecycles'] += 1
    io.verify_hashes(root, protocol['source_sha256']); io.verify_hashes(root, protocol['pretrained_sha256'])
    io.save(out/'checks.json', records)
    result = dict(passed=True, calls=2304, checks=dict(counts), seconds=time.perf_counter()-begin,
        query_targets_accessed=False, core_research_goal_complete=False,
        development_summary_sha256=io.sha(source/'summary.json'), outputs_sha256={'checks.json':io.sha(out/'checks.json')})
    io.save(out/'summary.json', result); print(dict(stage='deadline_audit_complete', **result), flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/io.BASE/'audit_v1'; out.mkdir(parents=True, exist_ok=False)
    try: audit(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
