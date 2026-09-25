"""348 full current calls, inherited 110 predictors and query-blind sealing."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import strong_pool_online_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box
from run_online_credit_fresh_v2 import signature, validate
from run_support_language_online_v1 import load, same


def verify(root, out, p):
    rows = suite.read(out/'rows.json'); index = {(r['seed'], r['method']): r for r in rows}
    assert len(rows) == len(index) == len(p['seeds'])*len(p['methods'])
    assert set(index) == {(s, m) for s in p['seeds'] for m in p['methods']}
    for r in rows:
        assert suite.sha(root/r['file']) == r['sha256'] and suite.sha(root/r['metadata_file']) == r['metadata_sha256']
        a = load(root/r['file']); validate(a); m = suite.read(root/r['metadata_file'])['metadata']
        assert not m['query_targets_accessed'] and m['execution_failed'] == r['execution_failed']
        ref = p['observed_inputs'][str(r['seed'])]; assert suite.sha(root/ref['file']) == ref['sha256']
        same(a, load(root/ref['file']), ['x_observed', 'v_observed', 'q_observed'])
        if r['origin'] == 'new_348': assert r['seconds'] == m['charged_complete_seconds'] > 0
        else: assert r['origin'] == 'frozen_pre348' and r['seconds'] is None
    for seed in p['seeds']:
        rr = sorted([r for r in rows if r['seed'] == seed and r['origin'] == 'new_348'], key=lambda r: r['order'])
        expected = [p['configs'][int(j)]['name'] for j in np.random.default_rng(np.random.SeedSequence([348929, seed])).permutation(19)]
        assert [r['method'] for r in rr] == expected and suite.read(out/str(seed)/'commit.json')['rows'] == rr
    return dict(predictors=len(rows), failures=sum(r['execution_failed'] for r in rows))


def run(root, out, stage):
    begin = time.perf_counter(); hashes = suite.gate(root); configs = suite.catalogue(root)
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    pre = root/suite.BASE/'preflight_predictions_v1'; bridge_pre = root/'results/strong_pool_credit_bridge/preflight_v1'
    bridge_files = suite.read(bridge_pre/'files.json')
    if stage == 'development':
        suite.complete(pre); assert suite.read(pre/'protocol.json')['source_sha256'] == hashes
    seeds = suite.SEEDS[:1] if stage == 'preflight' else suite.SEEDS
    inputs, input_manifest, _ = suite.budget.observed(root, seeds)
    parent = root/'results/support_language_online/development_predictions_v1'; suite.complete(parent)
    inherited = [r for r in suite.read(parent/'rows.json') if r['seed'] in seeds]
    oldindex = {(r['seed'], r['method']): r for r in inherited}
    env = suite.budget.resources.old.environment_snapshot(root)
    assert not env['other_research_or_git_pack_processes'], env
    p = dict(stage=stage, source_sha256=hashes, seeds=seeds, configs=configs, primary=suite.PRIMARY,
        methods=suite.read(parent/'protocol.json')['methods']+[c['name'] for c in configs],
        observed_inputs=input_manifest, parent_prediction_summary_sha256=suite.sha(parent/'summary.json'),
        query_targets_accessed=False, posterior_reference_accessed=False, new_blind_tasks=False, environment=env)
    suite.save(out/'protocol.json', p); ph = suite.sha(out/'protocol.json')
    rows = []
    for r in inherited:
        assert suite.sha(root/r['file']) == r['sha256'] and suite.sha(root/r['metadata_file']) == r['metadata_sha256']
        rows.append(dict(r, origin='frozen_pre348', historical_origin=r['origin'], historical_seconds=r['seconds'], seconds=None, order=None))
    counts = dict(new_calls=0, historical_base_arrays=0, bridge_replay_arrays=0,
                  preflight_replay_arrays=0, independent_original_arrays=0, trajectory_states=0,
                  shared_trigger_arrays=0, support_particles=0)
    with discovery_box(.12):
        for seed in seeds:
            directory = out/str(seed); directory.mkdir(); records = []; data = {}; metas = {}
            env = suite.budget.resources.old.environment_snapshot(root)
            assert not env['other_research_or_git_pack_processes'], env
            x, v, q = inputs[seed]; observed = signature(x, v, q)
            historical = oldindex[seed, 'probe_all_alm64']; olda = load(root/historical['file']); oldm = suite.read(root/historical['metadata_file'])['metadata']
            for j in np.random.default_rng(np.random.SeedSequence([348929, seed])).permutation(19):
                cfg = configs[int(j)]; name = cfg['name']; gc.collect()
                a, m, seconds = suite.invoke(cfg, x, v, q, seed)
                assert signature(x, v, q) == observed
                if 'reference_347' in cfg and not m['execution_failed']:
                    assert m['visited_modes'] == oldm['visited_modes'] and m['feasible_modes'] == oldm['feasible_modes']
                    assert m['original_positive_modes'] == oldm['positive_modes']
                    counts['historical_base_arrays'] += same(a, olda, ['best_bank', 'selected_b', 'point_prediction'])
                    assert set(m['original_positive_modes']) <= set(m['positive_modes'])
                    if m['positive_modes'] == m['original_positive_modes']:
                        counts['historical_base_arrays'] += same(a, olda, ['points', 'allocation', 'prediction'])
                    if seed < 328000008:
                        f = f"{seed}_{cfg['reference_347']}.npz"; mp = f.replace('.npz', '.json')
                        assert suite.sha(bridge_pre/f) == bridge_files[f] and suite.sha(bridge_pre/mp) == bridge_files[mp]
                        counts['bridge_replay_arrays'] += same(a, load(bridge_pre/f), list(a))
                        assert m['trajectory_state_sha256'] == suite.read(bridge_pre/mp)['trajectory_state_sha256']
                if stage == 'preflight' and 'channel' not in cfg['kwargs'] and not cfg['kwargs'].get('watch'):
                    refa, refm, fingerprints = suite.old_reference(root, cfg, x, v, q, seed)
                    assert not refm['execution_failed'] and not m['execution_failed']
                    counts['independent_original_arrays'] += same(a, refa, list(a))
                    assert m['visited_modes'] == refm['visited_modes'] and m['positive_modes'] == refm['positive_modes']
                    assert m['trajectory_state_sha256'] == fingerprints; counts['trajectory_states'] += len(fingerprints)
                if not m['execution_failed'] and m['positive_modes']:
                    yy = np.broadcast_to(x, (len(a['points']), len(x)))
                    for l in range(4): yy = np.maximum(0., 1.-abs(2*(yy+a['points'][:, l, None])-1.))
                    assert float(np.max(abs(yy-v))) <= .001+1e-7; counts['support_particles'] += len(a['points'])
                a.update(x_observed=x.copy(), v_observed=v.copy(), q_observed=q.copy()); validate(a)
                if stage == 'development' and seed == seeds[0]:
                    counts['preflight_replay_arrays'] += same(a, load(pre/str(seed)/(name+'.npz')), list(a))
                data[name] = a; metas[name] = m
                ap = directory/(name+'.npz'); mp = directory/(name+'.json')
                with ap.open('xb') as f: np.savez_compressed(f, **a); f.flush(); os.fsync(f.fileno())
                suite.save(mp, dict(seed=seed, method=name, order=len(records), protocol_sha256=ph, metadata=m))
                records.append(dict(seed=seed, method=name, origin='new_348', order=len(records),
                    file=str(ap.relative_to(root)), sha256=suite.sha(ap), metadata_file=str(mp.relative_to(root)),
                    metadata_sha256=suite.sha(mp), seconds=seconds, execution_failed=m['execution_failed']))
                counts['new_calls'] += 1
            native = metas['strong_native_alm64']; watch = metas['strong_watch_alm64']
            for name in ['strong_watch_alm64']+['strong_language_'+ch for ch in suite.bridge.CHANNELS]:
                m = metas[name]
                if not m['execution_failed'] and not native['execution_failed']:
                    assert m['trajectory_state_sha256'] == native['trajectory_state_sha256']; counts['trajectory_states'] += len(m['trajectory_state_sha256'])
                if name != 'strong_watch_alm64' and not m['execution_failed'] and not watch['execution_failed']:
                    assert m['selected_state'] == watch['selected_state']
                    counts['shared_trigger_arrays'] += same(data[name], data['strong_watch_alm64'], ['trigger_b', 'trigger_h', 'trigger_u', 'trigger_location'])
            suite.save(directory/'commit.json', dict(seed=seed, protocol_sha256=ph, rows=records, query_targets_accessed=False))
            rows.extend(records)
            print(dict(stage=stage, tasks=seeds.index(seed)+1, total=len(seeds), new_calls=counts['new_calls'], seconds=time.perf_counter()-begin, query_targets_accessed=False), flush=True)
    suite.save(out/'rows.json', rows); checks = verify(root, out, p); assert suite.gate(root) == hashes
    suite.save(out/'before_query_manifest.json', dict(protocol_sha256=ph, rows_sha256=suite.sha(out/'rows.json'), source_sha256=hashes,
        counts=counts, checks=checks, query_targets_accessed=False))
    result = dict(passed=True, tasks=len(seeds), counts=counts, checks=checks, seconds=time.perf_counter()-begin,
        query_targets_accessed=False, core_research_goal_complete=False,
        outputs_sha256={n: suite.sha(out/n) for n in ['protocol.json', 'rows.json', 'before_query_manifest.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['preflight', 'development'], required=True); args = ap.parse_args()
    root = Path(__file__).resolve().parents[2]; out = root/suite.BASE/(args.stage+'_predictions_v1'); out.mkdir(parents=True, exist_ok=False)
    try: run(root, out, args.stage)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
