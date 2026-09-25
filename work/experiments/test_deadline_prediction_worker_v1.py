"""386 controlled deadline probes and old n24 wide-budget equivalence only."""
from pathlib import Path
import os
import traceback
import numpy as np
import deadline_prediction_worker_v1 as model
import run_independent_hybrid_memory as old
from run_prefix_obstruction_v3 import read, sha, save


def save_case(out, name, prediction, metadata, events, archive):
    folder = out/name; folder.mkdir()
    arrays = dict(prediction=prediction)
    event_rows = []
    for i, event in enumerate(events):
        row = {k:value for k, value in event.items() if k != 'prediction'}
        if 'prediction' in event: arrays[f'event_{i}_prediction'] = event['prediction']
        event_rows.append(row)
    if archive is not None:
        arrays.update({'archive_'+k:value for k, value in archive['arrays'].items()})
        save(folder/'archive_metadata.json', archive['metadata'])
    np.savez_compressed(folder/'arrays.npz', **arrays)
    save(folder/'events.json', event_rows); save(folder/'metadata.json', metadata)
    return dict(case=name, selected=metadata['selected'], received_final=metadata['received_final'],
        setup_seconds=metadata['setup_seconds'], decision_seconds=metadata['decision_seconds'],
        controller_overrun_seconds=metadata['controller_overrun_seconds'], cleanup_seconds=metadata['cleanup_seconds'],
        terminated_owned_worker=metadata['terminated_owned_worker'], error=metadata['error'],
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()})


def main(root, out):
    sources = [Path(__file__), Path(model.__file__)]
    save(out/'protocol.json', dict(old_seed=5920000, new_query_targets_accessed=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        purpose='Boundary and implementation tests only, not formal timing or risk comparison'))
    q = np.linspace(0., 1., 257); x = np.array([.1, .2]); v = np.array([.3, .4])
    # Exact boundary cases do not depend on OS scheduling.
    e = [dict(kind='fallback', received_seconds=.1, prediction=np.full_like(q, .25)),
         dict(kind='final', received_seconds=.2, prediction=np.full_like(q, .75))]
    for budget, value, label in [(0., .5, 'constant'), (.1, .25, 'fallback'), (.199, .25, 'fallback'), (.2, .75, 'final')]:
        prediction, selected = model.choose(e, budget, q)
        np.testing.assert_array_equal(prediction, np.full_like(q, value)); assert selected == label
    cases = [
        ('zero', dict(kind='probe', first_delay=.2), 0., 'constant'),
        ('early', dict(kind='probe', final_delay=.02), 2., 'final'),
        ('late', dict(kind='probe', final_delay=2.), .3, 'fallback'),
        ('all_late', dict(kind='probe', first_delay=2.), .2, 'constant'),
        ('error', dict(kind='probe', raise_after_fallback=True), 2., 'fallback')]
    records = []
    for name, config, budget, expected in cases:
        result = model.execute(config, x, v, q, seed=38601, budget=budget, root=root)
        prediction, metadata, events, archive = result
        assert metadata['selected'] == expected, (name, metadata, events)
        expected_prediction, expected_selected = model.choose(events, budget, q)
        np.testing.assert_array_equal(prediction, expected_prediction); assert expected_selected == metadata['selected']
        if name in ['zero', 'late', 'all_late']: assert metadata['terminated_owned_worker']
        assert (metadata['error'] is not None) == (name == 'error')
        records.append(save_case(out, name, *result))
    x, v = old.observations(5920000)
    # Require a sealed, independently audited 384 run before using it as gold.
    parent = root/'results/contiguous_regional_repeat'
    audit = read(parent/'audit_v1/summary.json'); assert audit['passed']
    assert audit['development_summary_sha256'] == sha(parent/'development_v1/summary.json')
    gold_row = next(r for r in read(parent/'development_v1/rows.json') if r['seed'] == 5920000 and
        r['method'] == 'regional_active1024' and r['schedule'] == 'bfs' and r['ordering'] == 'observed' and r['canonical'])
    assert sha(root/gold_row['array_file']) == gold_row['array_sha256']
    gold = np.load(root/gold_row['array_file'], allow_pickle=False)
    config = dict(kind='regional', method='regional_active1024', schedule='bfs', ordering='observed')
    result = model.execute(config, x, v, q, seed=5920000, budget=20., root=root)
    prediction, metadata, events, archive = result
    assert metadata['error'] is None and metadata['selected'] == 'final' and archive is not None
    np.testing.assert_array_equal(prediction, gold['readout_prediction'])
    array_checks = 0
    for key in gold.files:
        assert archive['arrays'][key].tobytes() == gold[key].tobytes(), key
        array_checks += 1
    assert archive['metadata']['fallback_seconds'] > 0
    assert events[0]['kind'] == 'fallback' and events[-1]['kind'] == 'final'
    records.append(save_case(out, 'regional_wide', *result))
    result = model.execute(dict(kind='regression', method='prior4096_ridge'), x, v, q, seed=5920000, budget=20., root=root)
    prediction, metadata, events, archive = result
    assert metadata['error'] is None and metadata['selected'] == 'final' and len(events) == 1
    baseline = read(root/'results/n24_task_baselines/preflight_v2/summary.json'); assert baseline['passed']
    reference = np.load(root/'results/n24_task_baselines/preflight_v2/arrays.npz', allow_pickle=False)
    np.testing.assert_array_equal(prediction, np.clip(reference['prior4096_ridge_prediction'], 0., 1.))
    records.append(save_case(out, 'prior_wide', *result))
    for method, warm in [('meta_ridge128', False), ('meta_shallow64_20', True)]:
        result = model.execute(dict(kind='meta', method=method, warm_trajectory=warm), x, v, q,
            seed=5920000, budget=20., root=root)
        prediction, metadata, events, archive = result
        assert metadata['error'] is None and metadata['selected'] == 'final'
        mode = 'trajectory4_8_16_24' if warm else 'cold24'
        np.testing.assert_array_equal(prediction, reference[method+'_'+mode+'_prediction'])
        records.append(save_case(out, method+'_wide', *result))
    for family, steps, readout in [('adam', 240, 'posterior_union'), ('alm', 128, 'point')]:
        result = model.execute(dict(kind='optimizer', family=family, steps=steps, restarts=64, readout=readout),
            x, v, q, seed=5920000, budget=20., root=root)
        prediction, metadata, events, archive = result
        assert metadata['error'] is None and metadata['selected'] == 'final'
        basename = 'union' if readout == 'posterior_union' else 'point'
        with np.load(root/f'results/n24_optimizer_controls/preflight_v1/{family}/{basename}.npz', allow_pickle=False) as gold_optimizer:
            for key in gold_optimizer.files:
                assert gold_optimizer[key].tobytes() == archive['arrays'][key].tobytes(), (family, key)
            np.testing.assert_array_equal(prediction, gold_optimizer['prediction'])
        records.append(save_case(out, family+'_'+basename+'_wide', *result))
    save(out/'records.json', records)
    result = dict(passed=True, exact_boundary_cases=4, process_probe_cases=5, real_wide_cases=6,
        contiguous384_arrays_bitwise_equal=array_checks, prior_prediction_bitwise_equal=True,
        no_new_query_targets=True, not_a_task_risk_experiment=True, no_peak_rss_claim=True,
        parent384_audit_sha256=sha(parent/'audit_v1/summary.json'),
        outputs_sha256={p:sha(out/p) for p in ['protocol.json', 'records.json']})
    save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/'results/deadline_prediction/preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
