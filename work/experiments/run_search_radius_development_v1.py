"""340 support-only calls, old predictor references, then a durable query-blind seal."""
import argparse
import gc
import math
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import search_radius_development_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box
from run_online_credit_fresh_v2 import signature, validate, byte_fields
from test_budget_reinvestment_v1 import check_nested


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z: return {n: z[n] for n in z.files}


def original_rows(root, seeds, index):
    folder = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'
    originals = suite.budget.resources.catalogue(root); rows = []
    inputs, _, _ = suite.budget.observed(root, seeds)
    for seed in seeds:
        for cfg in originals:
            r = index[seed, cfg['name']]
            assert suite.sha(folder/r['file']) == r['sha256']
            assert suite.sha(folder/r['metadata_file']) == r['metadata_sha256']
            a = load_arrays(folder/r['file']); validate(a)
            assert signature(a['x_observed'], a['v_observed'], a['q_observed']) == signature(*inputs[seed])
            rows.append(dict(seed=seed, method=cfg['name'], origin='frozen_328', order=None,
                file=str((folder/r['file']).relative_to(root)), sha256=r['sha256'],
                metadata_file=str((folder/r['metadata_file']).relative_to(root)), metadata_sha256=r['metadata_sha256'],
                seconds=None, original_seconds_not_current_benchmark=r['seconds'], execution_failed=r['execution_failed']))
    return rows


def invariant(root, seed, cfg, a, m, index):
    if cfg['family'] not in ['radius_online', 'budgeted_online_credit'] or m['execution_failed']: return 0
    ref = index[seed, cfg['base_config']['name']]
    folder = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'
    assert suite.sha(folder/ref['file']) == ref['sha256']
    assert suite.sha(folder/ref['metadata_file']) == ref['metadata_sha256']
    b = {n: value for n, value in load_arrays(folder/ref['file']).items()
         if n not in ['x_observed', 'v_observed', 'q_observed']}
    old = suite.read(folder/ref['metadata_file'])['metadata']; assert not old['execution_failed']
    assert m['original_positive_modes'] == old['original_positive_modes']
    assert m['selected_state'] == old['selected_state']
    assert m['no_global_bp_guard_enabled'] == (cfg['base_config']['channel'] != 'bp')
    keys = set(a)-{'points', 'allocation', 'prediction'}
    checks = suite.budget.array_checks({n: a[n] for n in keys}, {n: b[n] for n in keys})
    if cfg['family'] == 'budgeted_online_credit' or cfg['k'] == 8:
        assert set(old['positive_modes']) <= set(m['positive_modes'])
        assert all(p in m['proposal']['proposals'] for p in old['proposal']['proposals'])
    if old['positive_modes'] == m['positive_modes']:
        checks += suite.budget.array_checks({n: a[n] for n in ['points', 'allocation', 'prediction']},
                                            {n: b[n] for n in ['points', 'allocation', 'prediction']})
    if cfg['family'] == 'budgeted_online_credit' and seed in suite.budget.SEEDS:
        cal = root/suite.budget.BASE/'calibration_v1'; file = f'{seed}_{cfg["name"]}.npz'
        assert suite.sha(cal/file) == suite.read(cal/'files.json')[file]
        checks += suite.budget.array_checks(a, load_arrays(cal/file))
    return checks


def verify(root, folder, protocol):
    rows = suite.read(folder/'rows.json'); expected = {(s, n) for s in protocol['seeds'] for n in protocol['methods']}
    assert {(r['seed'], r['method']) for r in rows} == expected and len(rows) == len(expected)
    inputs, _, _ = suite.budget.observed(root, protocol['seeds']); checks = 0
    for r in rows:
        assert suite.sha(root/r['file']) == r['sha256']
        assert suite.sha(root/r['metadata_file']) == r['metadata_sha256']
        log = suite.read(root/r['metadata_file']); m = log['metadata']
        assert not m['query_targets_accessed'] and m['execution_failed'] == r['execution_failed']
        a = load_arrays(root/r['file']); validate(a)
        assert signature(a['x_observed'], a['v_observed'], a['q_observed']) == signature(*inputs[r['seed']])
        if r['origin'] == 'new_340':
            assert r['seconds'] > 0 and math.isfinite(r['seconds']) and r['seconds'] == m['charged_complete_seconds']
            assert log['protocol_sha256'] == suite.sha(folder/'protocol.json')
            assert log['seed'] == r['seed'] and log['method'] == r['method'] and log['order'] == r['order']
        else: assert r['origin'] == 'frozen_328' and r['seconds'] is None
        checks += len(a)
    for seed in protocol['seeds']:
        rr = sorted([r for r in rows if r['seed'] == seed and r['origin'] == 'new_340'], key=lambda r: r['order'])
        wanted = [protocol['configs'][int(j)]['name'] for j in
                  np.random.default_rng(np.random.SeedSequence([340929, seed])).permutation(len(protocol['configs']))]
        assert [r['method'] for r in rr] == wanted
        commit = suite.read(folder/str(seed)/'commit.json')
        assert commit['rows'] == rr and not commit['query_targets_accessed']
        assert commit['protocol_sha256'] == suite.sha(folder/'protocol.json')
    return dict(predictors=len(rows), numeric_array_checks=checks,
                failures=sum(r['execution_failed'] for r in rows))


def run(root, out, stage):
    begin = time.perf_counter(); hashes = suite.gate(root); cfgs = suite.catalogue(root)
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    preflight = root/suite.BASE/'preflight_predictions_v1'
    if stage == 'development':
        tested = suite.complete(preflight); assert tested['new_calls'] == len(cfgs)
        assert suite.read(preflight/'protocol.json')['source_sha256'] == hashes
    env = suite.budget.resources.old.environment_snapshot(root)
    assert not env['other_research_or_git_pack_processes'], env
    tick = time.perf_counter(); loaded, checkpoint = suite.budget.resources.old.resources.legacy.oldfit.meta.load(root)
    loading = time.perf_counter()-tick
    cal = root/suite.budget.BASE/'calibration_v1'
    assert checkpoint == suite.read(cal/'protocol.json')['checkpoint_manifest']
    seeds = suite.SEEDS[:1] if stage == 'preflight' else suite.SEEDS
    inputs, input_manifest, index = suite.budget.observed(root, seeds)
    originals = suite.budget.resources.catalogue(root)
    protocol = dict(source_sha256=hashes, stage=stage, seeds=seeds, configs=cfgs,
        methods=[c['name'] for c in originals+cfgs], primary=suite.PRIMARY, order_seed=340929,
        observed_inputs=input_manifest, checkpoint_manifest=checkpoint, model_loading_seconds=loading,
        resource_selection=suite.read(cal/'selection.json'), environment_start=env,
        query_targets_accessed=False, posterior_reference_accessed=False, new_blind_tasks=False,
        inference='Descriptive development only; old times are not current paired benchmarks',
        new_fit_time_scope='Complete fit and projection; excludes I/O, loading, checks and manual GC',
        failure_policy='Preserve guarded numerical fallback and charge failed fit; forbidden BP/solver access is fatal')
    suite.save(out/'protocol.json', protocol); ph = suite.sha(out/'protocol.json')
    rows = original_rows(root, seeds, index); counts = dict(invariant_arrays=0, nested_pairs=0, support_particles=0, replay_arrays=0)
    calls = 0
    with discovery_box(.12):
        for seed in seeds:
            env = suite.budget.resources.old.environment_snapshot(root)
            assert not env['other_research_or_git_pack_processes'], env
            directory = out/str(seed); directory.mkdir(); suite.save(directory/'environment.json', env)
            x, v, q = inputs[seed]; observed = signature(x, v, q); records = []; data = {}
            for j in np.random.default_rng(np.random.SeedSequence([340929, seed])).permutation(len(cfgs)):
                cfg = cfgs[int(j)]; name = cfg['name']; gc.collect()
                a, m, seconds = suite.invoke(cfg, x, v, q, seed, loaded)
                assert signature(x, v, q) == observed
                counts['invariant_arrays'] += invariant(root, seed, cfg, a, m, index)
                if not m['execution_failed'] and m.get('positive_modes'):
                    h = np.broadcast_to(x, (len(a['points']), len(x)))
                    for layer in range(4): h = np.maximum(0., 1.-abs(2*(h+a['points'][:, layer, None])-1.))
                    assert float(np.max(abs(h-v))) <= .001+1e-7
                    counts['support_particles'] += len(a['points'])
                if cfg['family'] == 'radius_online': data[name] = a, m
                a.update(x_observed=x.copy(), v_observed=v.copy(), q_observed=q.copy()); validate(a)
                if stage == 'development' and seed == suite.SEEDS[0]:
                    pp = preflight/str(seed)/(name+'.npz')
                    rr = next(r for r in suite.read(preflight/'rows.json') if r['method'] == name)
                    assert suite.sha(pp) == rr['sha256']
                    counts['replay_arrays'] += suite.budget.array_checks(a, load_arrays(pp))
                path = directory/(name+'.npz'); mp = directory/(name+'.json')
                with path.open('xb') as f: np.savez_compressed(f, **a); f.flush(); os.fsync(f.fileno())
                suite.save(mp, dict(seed=seed, method=name, order=len(records), protocol_sha256=ph, metadata=m))
                records.append(dict(seed=seed, method=name, origin='new_340', order=len(records),
                    file=str(path.relative_to(root)), sha256=suite.sha(path), metadata_file=str(mp.relative_to(root)),
                    metadata_sha256=suite.sha(mp), seconds=seconds, execution_failed=m['execution_failed'], named_byte_fields=byte_fields(m)))
                calls += 1
            for channel in suite.budget.fast.original.CHANNELS:
                for k1, k2 in [(1, 2), (2, 4), (4, 8)]:
                    a, m = data[f'radius_first_fit_{channel}__k{k1}__sall']
                    b, n = data[f'radius_first_fit_{channel}__k{k2}__sall']
                    if not m['execution_failed'] and not n['execution_failed']:
                        counts['invariant_arrays'] += check_nested((a, m), (b, n)); counts['nested_pairs'] += 1
            suite.save(directory/'commit.json', dict(seed=seed, protocol_sha256=ph, rows=records,
                observed_sha256=observed, query_targets_accessed=False))
            rows.extend(records)
            print(dict(stage=stage, tasks=seeds.index(seed)+1, total=len(seeds), new_calls=calls,
                       seconds=time.perf_counter()-begin, query_targets_accessed=False), flush=True)
    suite.save(out/'rows.json', rows); checked = verify(root, out, protocol)
    assert suite.gate(root) == hashes
    suite.save(out/'before_query_manifest.json', dict(protocol_sha256=ph, rows_sha256=suite.sha(out/'rows.json'),
        source_sha256=hashes, query_targets_accessed=False, checks=checked, invariants=counts))
    result = dict(passed=True, tasks=len(seeds), new_calls=calls, cached_predictors=len(seeds)*51,
        checks=checked, invariants=counts, seconds=time.perf_counter()-begin, query_targets_accessed=False,
        core_research_goal_complete=False, outputs_sha256={n: suite.sha(out/n) for n in
            ['protocol.json', 'rows.json', 'before_query_manifest.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['preflight', 'development'], required=True)
    args = ap.parse_args(); root = Path(__file__).resolve().parents[2]
    out = root/suite.BASE/f'{args.stage}_predictions_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out, args.stage)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
