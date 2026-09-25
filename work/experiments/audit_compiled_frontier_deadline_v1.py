"""427 complete legacy audit plus exact receipts for interrupted new methods."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import math
import shutil
from types import SimpleNamespace
import time
import numpy as np
import compiled_frontier_deadline_io_v1 as io
import frontier_search_receipt_v1 as receipt
import audit_continuous_frontier_v1 as continuous
from deadline_prediction_worker_v1 import choose


def legacy_view(root, source, out, protocol, rows, sessions):
    """A labeled subset of the SAME run, never a reconstructed timing experiment."""
    folder = out/'legacy_view'; folder.mkdir(); (folder/'inputs').mkdir()
    original = io.previous.configs(root)
    selected = {c['name'] for c in original}
    rr = [r for r in rows if r['method'] in selected]
    ss = [s for s in sessions if s['method'] in selected]
    assert len(rr) == 1344 and len(ss) == 42
    derived = deepcopy(protocol)
    derived.update(configs=original, expected_calls=len(rr), derived_audit_view=True,
        parent_protocol_sha256=io.sha(source/'protocol.json'), parent_summary_sha256=io.sha(source/'summary.json'))
    io.save(folder/'protocol.json', derived); io.save(folder/'rows.json', rr); io.save(folder/'sessions.json', ss)
    shutil.copyfile(source/'inputs_manifest.json', folder/'inputs_manifest.json')
    for name in io.read(source/'inputs_manifest.json'):
        shutil.copyfile(source/'inputs'/name, folder/'inputs'/name)
    io.save(folder/'summary.json', dict(passed=True, all_predictions_sealed=True, calls=len(rr),
        derived_audit_view=True, not_an_independent_experiment=True,
        parent_summary_sha256=io.sha(source/'summary.json'),
        outputs_sha256={f: io.sha(folder/f) for f in ['protocol.json', 'rows.json', 'sessions.json', 'inputs_manifest.json']}))
    def jobs(configurations, seeds, stages, budgets):
        assert configurations == original
        return ((ci, cfg, jj) for ci, cfg, jj in io.jobs(protocol['configs'], seeds, stages, budgets)
                if cfg['name'] in selected)
    proxy = SimpleNamespace(old=io.old, configs=lambda root: original, jobs=jobs, STAGES=io.STAGES,
        read=io.read, save=io.save, sha=io.sha, load_arrays=io.load_arrays, observations=io.observations)
    import audit_prefix_deadline_pilot_v1 as legacy
    before = legacy.io; checked = out/'legacy_audit'; checked.mkdir()
    try:
        legacy.io = proxy
        legacy.audit(root, folder, checked)
    finally:
        legacy.io = before
    summary = io.read(checked/'summary.json')
    assert summary['passed'] and summary['calls'] == 1344
    return summary


def audit(root, out):
    begin = time.perf_counter(); source = root/io.BASE/'development_v1'
    summary = io.read(source/'summary.json'); protocol = io.read(source/'protocol.json')
    assert summary['passed'] and summary['all_predictions_sealed'] and not (source/'failure.json').exists()
    for f, digest in summary['outputs_sha256'].items():
        assert io.sha(source/f) == digest
    io.old.verify_hashes(root, protocol['source_sha256']); io.old.verify_hashes(root, protocol['pretrained_sha256'])
    configurations = io.configs(root); assert configurations == protocol['configs']
    rows = io.read(source/'rows.json'); sessions = io.read(source/'sessions.json')
    expected = [(ci, c['name'], seed, n, budget) for ci, c, jj in
                io.jobs(configurations, protocol['seeds'], protocol['stages'], protocol['budgets']) for seed, n, budget in jj]
    assert [(r['config_index'], r['method'], r['seed'], r['n'], r['budget']) for r in rows] == expected
    assert len(rows) == protocol['expected_calls'] == summary['calls'] == 1792
    assert len(sessions) == 56
    legacy = legacy_view(root, source, out, protocol, rows, sessions)
    cfgs = {c['name']: c for c in configurations}; inputs = {}
    for name, digest in io.read(source/'inputs_manifest.json').items():
        path = source/'inputs'/name; assert io.sha(path) == digest
        d = io.load_arrays(path); assert set(d) == {'x', 'v', 'q'}
        x, v = io.observations(int(path.stem))
        np.testing.assert_array_equal(x, d['x']); np.testing.assert_array_equal(v, d['v'])
        np.testing.assert_array_equal(d['q'], np.linspace(0., 1., 257)); inputs[int(path.stem)] = d
    counts = Counter(); records = []; by_session = defaultdict(list)
    for row in rows:
        cfg = cfgs[row['method']]
        if cfg['kind'] not in ['frontier_receipt', 'compiled_frontier_receipt']:
            continue
        d = inputs[row['seed']]; x, v, q = d['x'][:row['n']], d['v'][:row['n']], d['q']
        folder = root/row['directory']
        for f, digest in row['files'].items():
            assert io.sha(folder/f) == digest
        values = io.load_arrays(folder/'outputs.npz'); m = io.read(folder/'metadata.json')
        events = io.read(folder/'events.json')
        assert m['config'] == cfg and m['budget_seconds'] == row['budget'] and m['error'] is None
        assert not m['query_targets_accessed'] and not m['shared_scientific_task_state_reused']
        assert m['controller_overrun_seconds'] == max(0., m['decision_seconds']-row['budget'])
        for field, stored in [('online_seconds', 'decision_seconds'), *[(k, k) for k in
            ['received_final', 'selected', 'setup_seconds', 'cleanup_seconds', 'task_cleanup_seconds',
             'archive_seconds', 'controller_overrun_seconds', 'worker_max_sampled_rss',
             'controller_max_sampled_rss', 'worker_pid', 'generation']]]:
            assert row[field] == m[stored]
        assert [e['ordinal'] for e in events] == list(range(len(events)))
        assert all(e['generation'] == m['generation'] for e in events)
        assert all(a['received_seconds'] <= b['received_seconds'] for a, b in zip(events, events[1:]))
        for i, e in enumerate(events):
            assert e['kind'] in ['fallback', 'final']
            assert 0 <= e['worker_seconds'] <= e['received_seconds']+1e-3
            e['prediction'] = values[f'event_{i}_prediction']
        for value in values.values():
            assert value.shape == q.shape and np.isfinite(value).all() and np.all((0 <= value) & (value <= 1))
        selected, kind = choose(events, row['budget'], q)
        np.testing.assert_array_equal(values['prediction'], selected); assert kind == m['selected']
        local = Counter()
        if m['received_final']:
            assert events[-1]['kind'] == 'final' and m['model_fingerprint_verified'] and not m['terminated_owned_worker']
            a = io.load_arrays(folder/'state.npz'); mm = io.read(folder/'state.json')
            for key, value in [('x', x), ('v', v), ('q', q)]:
                np.testing.assert_array_equal(a[key], value)
            np.testing.assert_array_equal(a['prediction'], events[-1]['prediction'])
            local.update(continuous.audit_call(a, mm, events))
            if cfg['kind'] == 'compiled_frontier_receipt' and mm['branch'] == 'frontier':
                rm = mm['range_readout']; programs = rm['compiled_programs']
                assert rm['compiled_inside_paid_call'] and rm['program_bytes_are_not_peak_RSS']
                assert rm['compile_seconds'] >= 0 and rm['state_accounting_seconds'] >= 0
                assert rm['program_pickle_bytes'] > 0 and len(programs) == len(rm['coordinate_boxes'])+1
                assert sum(rm['compiled_routes'].values()) == len(programs)*len(q)
                assert all(p['max_segments'] == 4096 for p in programs)
                local['compiled_online_state_records'] += 1
        else:
            assert m['terminated_owned_worker'] and 'state.npz' not in row['files']
        if any('frontier_receipt' in e for e in events):
            for e in events:
                if 'frontier_receipt' in e:
                    rr = e['frontier_receipt']
                    assert rr['method'] == cfg['method'] and rr['original_max_expanded'] == cfg['max_expanded']
                    assert rr['batch_queries'] == cfg['batch_queries']
            local.update(receipt.audit_events(x, v, q, events))
            if not m['received_final']:
                local['interrupted_calls_with_certified_deliveries'] += 1
        elif not m['received_final']:
            # No frontier packet exists: only the paid common fallback/envelope.
            assert len(events) <= 2 and all(e['kind'] == 'fallback' for e in events)
            if events:
                from test_deadline_risk_session_v1 import kernel_reference
                np.testing.assert_allclose(events[0]['prediction'], kernel_reference(x, v, q, 4096), rtol=1e-10, atol=1e-10)
                local['interrupted_prior_replays'] += 1
            if len(events) == 2:
                bounds = receipt.model.outward(receipt.model.ranges.core.lipschitz_intervals(x, v, q))
                np.testing.assert_array_equal(events[1]['prediction'], np.clip(events[0]['prediction'], bounds[:, 0], bounds[:, 1]))
                local['interrupted_cheap_replays'] += 1
        samples = m['resource_samples']
        assert all(s['worker_rss'] > 0 and s['controller_rss'] > 0 and s['worker_cpu_seconds'] >= -1e-9 for s in samples)
        assert all(a['seconds'] <= b['seconds'] for a, b in zip(samples, samples[1:]))
        assert m['worker_max_sampled_rss'] == max((s['worker_rss'] for s in samples), default=m['setup_worker_memory']['rss'])
        local['independent_deadline_choices'] += 1; local['resource_records'] += 1
        counts.update(local); records.append(dict(directory=row['directory'], checks=dict(local)))
        by_session[row['method']].append(row)
        if len(records) % 16 == 0:
            print(dict(stage='frontier_receipt_audit', calls=len(records), seconds=time.perf_counter()-begin), flush=True)
    assert len(records) == 448 and len(by_session) == 14
    for session in sessions:
        if session['method'] not in by_session:
            continue
        rr = by_session[session['method']]; assert len(rr) == session['calls'] == 32
        assert rr[0]['generation'] == 0 and rr[0]['setup_seconds'] > 0
        for a, b in zip(rr, rr[1:]):
            if a['received_final']:
                assert b['generation'] == a['generation']+1 and b['worker_pid'] == a['worker_pid'] and b['setup_seconds'] == 0
            else:
                assert b['generation'] == 0 and b['setup_seconds'] > 0
        assert math.isclose(session['total_setup_seconds'], math.fsum(r['setup_seconds'] for r in rr), rel_tol=0, abs_tol=1e-8)
        assert session['total_closure_seconds']+1e-8 >= math.fsum(r['cleanup_seconds'] for r in rr)
        counts['session_lifecycles'] += 1
    io.old.verify_hashes(root, protocol['source_sha256']); io.old.verify_hashes(root, protocol['pretrained_sha256'])
    io.save(out/'new_checks.json', records)
    merged = Counter(legacy['checks']); merged.update(counts)
    result = dict(passed=True, calls=1792, legacy_calls=1344, new_calls=448, checks=dict(merged),
        new_checks=dict(counts), query_targets_accessed=False, core_research_goal_complete=False,
        prediction_summary_sha256=io.sha(source/'summary.json'), seconds=time.perf_counter()-begin,
        outputs_sha256={f: io.sha(out/f) for f in ['new_checks.json', 'legacy_audit/summary.json',
            'legacy_audit/checks.json', 'legacy_view/summary.json', 'legacy_view/protocol.json',
            'legacy_view/rows.json', 'legacy_view/sessions.json', 'legacy_view/inputs_manifest.json']})
    io.save(out/'summary.json', result); print(result, flush=True)
