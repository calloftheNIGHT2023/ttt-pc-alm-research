"""387 old-task preflight of 36 controls, large regressions, task isolation."""
from pathlib import Path
import os
import time
import traceback
import numpy as np
import deadline_prediction_worker_v2 as model
import deadline_risk_registry_v1 as registry
import deadline_risk_io_v1 as io


def kernel_reference(x, v, q, features):
    bank = np.random.default_rng(731).uniform(-.12, .12, (features, 4))
    def phi(z):
        h = np.broadcast_to(z, (features, len(z)))
        for j in range(4): h = np.maximum(0., 1.-np.abs(2*(h+bank[:,j,None])-1.))
        return h
    feature = phi(x); mu = feature.mean(0); centered = feature-mu
    alpha = np.linalg.solve(centered.T@centered/features+np.eye(len(x))*.001**2/3, v-mu)
    answer = []
    for start in range(0, len(q), 64):
        f = phi(q[start:start+64]); m = f.mean(0)
        answer.append(m+(f-m).T@centered@alpha/features)
    return np.clip(np.concatenate(answer), 0., 1.)


def main(root, out):
    h = io.prerequisites(root); history = io.explicit_seed_history(root, registry.SEEDS)
    io.save(out/'protocol.json', dict(**h, history=history, configs=registry.configs(), old_seeds=[5920000, 5920001],
        source_scope='Model/registry/preflight frozen here; subsequently added runner/audit/evaluation sources frozen before formal run',
        query_targets_accessed=False, new_supports_generated=False))
    x, v = io.observations(5920000); xb, vb = io.observations(5920001); q = np.linspace(0., 1., 257)
    oldrows = io.read(root/'results/contiguous_regional_repeat/development_v1/rows.json')
    reference_meta = io.load_arrays(root/'results/n24_task_baselines/preflight_v2/arrays.npz')
    records = []; checks = dict(configs=0, historical_scientific_arrays=0, prior_kernel_references=0,
        task_isolation_arrays=0, task_isolation_sequences=0, extended_restart_banks=0)
    repeat = {registry.PRIMARY, 'prior65536_ridge', 'meta_ridge128', 'meta_shallow64_20', 'adam240_r256_posterior_union'}
    for cfg in registry.configs():
        session = model.Session(cfg, root)
        try:
            result = session.run(x, v, q, seed=5920000, budget=20.)
            pred, meta, events, archive = result
            row = io.save_call(root, out/cfg['name']/'a0', *result)
            assert meta['selected'] == 'final' and meta['error'] is None and archive is not None, meta
            a = archive['arrays']; np.testing.assert_array_equal(pred, a['prediction'])
            if cfg['kind'] == 'regional':
                rr = next(r for r in oldrows if r['seed'] == 5920000 and r['method'] == cfg['method'] and
                    r['schedule'] == cfg['schedule'] and r['ordering'] == cfg['ordering'] and r['canonical'])
                assert io.sha(root/rr['array_file']) == rr['array_sha256']
                gold = io.load_arrays(root/rr['array_file'])
                for k in gold: assert a[k].tobytes() == gold[k].tobytes(); checks['historical_scientific_arrays'] += 1
            elif cfg['kind'] == 'meta':
                mode = 'trajectory4_8_16_24' if cfg['warm_trajectory'] else 'cold24'
                np.testing.assert_array_equal(pred, reference_meta[cfg['method']+'_'+mode+'_prediction'])
            elif cfg['kind'] == 'regression':
                if cfg['method'] in ['prior16384_ridge', 'prior65536_ridge']:
                    n = int(cfg['method'].split('_')[0][5:]); ref = kernel_reference(x, v, q, n)
                    error = float(np.max(abs(pred-ref))); assert error < 1e-10, error
                    row['independent_kernel_max_error'] = error; checks['prior_kernel_references'] += 1
                    np.testing.assert_array_equal(a['fallback_prediction'], np.clip(reference_meta['prior4096_ridge_prediction'], 0., 1.))
                elif cfg['method'] in ['residual_linear_ridge', 'residual_linear_rls']:
                    np.testing.assert_allclose(pred, np.clip(reference_meta['rls_prediction'], 0., 1.), atol=1e-10, rtol=1e-10)
                else: np.testing.assert_array_equal(pred, np.clip(reference_meta[cfg['method']+'_prediction'], 0., 1.))
            else:
                if cfg['restarts'] == 64:
                    name = 'point' if cfg['readout'] == 'point' else 'union'
                    gold = io.load_arrays(root/f"results/n24_optimizer_controls/preflight_v1/{cfg['family']}/{name}.npz")
                    for k in gold: assert a[k].tobytes() == gold[k].tobytes(); checks['historical_scientific_arrays'] += 1
                else:
                    import n24_optimizer_controls_v1 as control
                    with control.prior_bounds():
                        bank, _, _ = control.old.run_bp(control.starts(256), x, v, cfg['family'], cfg['steps'], trace=False)
                    np.testing.assert_array_equal(bank, a['best_bank']); checks['extended_restart_banks'] += 1
            if cfg['name'] in repeat:
                second = session.run(xb, vb, q, seed=5920001, budget=20.)
                second_row = io.save_call(root, out/cfg['name']/'b', *second)
                assert second[1]['error'] is None and second[1]['selected'] == 'final'
                third = session.run(x, v, q, seed=5920000, budget=20.)
                third_row = io.save_call(root, out/cfg['name']/'a1', *third)
                assert third[1]['error'] is None and third[1]['generation'] == 2
                assert meta['worker_pid'] == second[1]['worker_pid'] == third[1]['worker_pid']
                for k in a:
                    assert a[k].tobytes() == third[3]['arrays'][k].tobytes(), (cfg['name'], k)
                    checks['task_isolation_arrays'] += 1
                row['isolation'] = [second_row, third_row]; checks['task_isolation_sequences'] += 1
            records.append(dict(method=cfg['name'], **row)); checks['configs'] += 1
        finally:
            io.save(out/cfg['name']/'session.json', session.close())
        print(dict(stage='session_preflight', configs=checks['configs'], total=36), flush=True)
    probe = model.Session(dict(kind='probe', final_delay=.4), root); probe_rows = []
    try:
        for i, budget in enumerate([0., 2., .08, 2.]):
            result = probe.run(x, v, q, seed=5920000, budget=budget)
            probe_rows.append(io.save_call(root, out/f'probe_{i}', *result))
            assert result[1]['error'] is None
            assert result[1]['selected'] == ['constant', 'final', 'fallback', 'final'][i]
        assert probe_rows[1]['worker_pid'] == probe_rows[2]['worker_pid']
        assert probe_rows[0]['worker_pid'] != probe_rows[1]['worker_pid'] != probe_rows[3]['worker_pid']
    finally: probe_summary = probe.close()
    assert len(probe_summary['setups']) == 3 and all(c['exitcode'] is not None for c in probe_summary['closures'])
    io.verify_hashes(root, h['source_sha256']); io.verify_hashes(root, h['pretrained_sha256'])
    io.save(out/'records.json', records); io.save(out/'probes.json', dict(rows=probe_rows, session=probe_summary))
    result = dict(passed=True, checks=checks, deadline_recreation_probe_calls=4, no_new_tasks=True,
        query_targets_accessed=False, outputs_sha256={p:io.sha(out/p) for p in ['protocol.json', 'records.json', 'probes.json']})
    io.save(out/'summary.json', result); print(dict(stage='session_preflight_complete', **result), flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/io.BASE/'preflight_v1'; out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
