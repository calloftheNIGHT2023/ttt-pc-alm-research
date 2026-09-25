"""345 support-only predictions and durable seal before any new risk access."""
import argparse
import gc
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
import support_language_online_suite_v1 as suite
from posterior_confirmation_pipeline import discovery_box
from run_online_credit_fresh_v2 import signature, validate


def load(path):
    with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}


def same(a, b, keys):
    for k in keys:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes(), k
    return len(keys)


def verify(root, out, p):
    rows = suite.read(out/'rows.json'); lookup = {(r['seed'], r['method']): r for r in rows}
    assert len(rows) == len(lookup) == len(p['seeds'])*len(p['methods'])
    assert set(lookup) == {(s, m) for s in p['seeds'] for m in p['methods']}
    for r in rows:
        assert suite.sha(root/r['file']) == r['sha256'] and suite.sha(root/r['metadata_file']) == r['metadata_sha256']
        a = load(root/r['file']); validate(a); m = suite.read(root/r['metadata_file'])['metadata']
        assert not m['query_targets_accessed'] and m['execution_failed'] == r['execution_failed']
        observed = load(root/p['observed_inputs'][str(r['seed'])]['file'])
        same(a, observed, ['x_observed', 'v_observed', 'q_observed'])
        if r['origin'] == 'new_345': assert r['seconds'] == m['charged_complete_seconds'] > 0
        else: assert r['origin'] == 'frozen_pre345' and r['seconds'] is None
    for seed in p['seeds']:
        rr = sorted([r for r in rows if r['seed'] == seed and r['origin'] == 'new_345'], key=lambda r: r['order'])
        assert [r['method'] for r in rr] == [p['configs'][int(i)]['name'] for i in np.random.default_rng(np.random.SeedSequence([345929, seed])).permutation(6)]
        assert suite.read(out/str(seed)/'commit.json')['rows'] == rr
    return dict(predictors=len(rows), failures=sum(r['execution_failed'] for r in rows))


def run(root, out, stage):
    begin = time.perf_counter(); hashes = suite.gate(root); cfgs = suite.catalogue(root)
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    pre = root/suite.BASE/'preflight_predictions_v1'
    if stage == 'development':
        suite.complete(pre); assert suite.read(pre/'protocol.json')['source_sha256'] == hashes
    seeds = suite.SEEDS[:1] if stage == 'preflight' else suite.SEEDS
    inputs, input_manifest, _ = suite.budget.observed(root, seeds)
    old = root/'results/search_radius_development/development_predictions_v1'; suite.complete(old)
    inherited = [r for r in suite.read(old/'rows.json') if r['seed'] in seeds]
    oldindex = {(r['seed'], r['method']): r for r in inherited}
    env = suite.budget.resources.old.environment_snapshot(root)
    assert not env['other_research_or_git_pack_processes'], env
    p = dict(stage=stage, source_sha256=hashes, seeds=seeds, configs=cfgs, primary=suite.PRIMARY,
        methods=suite.read(old/'protocol.json')['methods']+[c['name'] for c in cfgs],
        observed_inputs=input_manifest, parent_prediction_summary_sha256=suite.sha(old/'summary.json'),
        query_targets_accessed=False, posterior_reference_accessed=False, new_blind_tasks=False, environment=env,
        new_fit_time_scope='Complete fit/projection; excludes I/O, loading, verification and manual GC')
    suite.save(out/'protocol.json', p); ph = suite.sha(out/'protocol.json')
    rows = []
    for r in inherited:
        assert suite.sha(root/r['file']) == r['sha256'] and suite.sha(root/r['metadata_file']) == r['metadata_sha256']
        rows.append(dict(r, origin='frozen_pre345', historical_origin=r['origin'], historical_seconds=r['seconds'], seconds=None, order=None))
    counts = dict(invariant_arrays=0, replay_arrays=0, support_particles=0, new_calls=0)
    with discovery_box(.12):
        for seed in seeds:
            directory = out/str(seed); directory.mkdir(); records = []
            env = suite.budget.resources.old.environment_snapshot(root)
            assert not env['other_research_or_git_pack_processes'], env
            x, v, q = inputs[seed]; observed = signature(x, v, q)
            for j in np.random.default_rng(np.random.SeedSequence([345929, seed])).permutation(6):
                cfg = cfgs[int(j)]; name = cfg['name']; gc.collect()
                a, m, seconds = suite.invoke(cfg, x, v, q, seed)
                assert signature(x, v, q) == observed
                a.update(x_observed=x.copy(), v_observed=v.copy(), q_observed=q.copy()); validate(a)
                prior = oldindex[seed, cfg['base_config']['name']]
                olda = load(root/prior['file']); oldm = suite.read(root/prior['metadata_file'])['metadata']
                if not m['execution_failed']:
                    assert m['original_positive_modes'] == oldm['original_positive_modes'] and m['selected_state'] == oldm['selected_state']
                    assert m['no_global_bp_guard_enabled'] == (cfg['channel'] != 'bp')
                    counts['invariant_arrays'] += same(a, olda, sorted(set(a)-{'points', 'allocation', 'prediction'}))
                    if m['positive_modes'] == oldm['positive_modes']:
                        counts['invariant_arrays'] += same(a, olda, ['points', 'allocation', 'prediction'])
                    if m['positive_modes']:
                        h = np.broadcast_to(x, (len(a['points']), len(x)))
                        for l in range(4): h = np.maximum(0., 1.-abs(2*(h+a['points'][:, l, None])-1.))
                        assert float(np.max(abs(h-v))) <= .001+1e-7
                        counts['support_particles'] += len(a['points'])
                if stage == 'development' and seed == seeds[0]:
                    counts['replay_arrays'] += same(a, load(pre/str(seed)/(name+'.npz')), list(a))
                ap = directory/(name+'.npz'); mp = directory/(name+'.json')
                with ap.open('xb') as f: np.savez_compressed(f, **a); f.flush(); os.fsync(f.fileno())
                suite.save(mp, dict(seed=seed, method=name, order=len(records), protocol_sha256=ph, metadata=m))
                records.append(dict(seed=seed, method=name, origin='new_345', order=len(records),
                    file=str(ap.relative_to(root)), sha256=suite.sha(ap), metadata_file=str(mp.relative_to(root)),
                    metadata_sha256=suite.sha(mp), seconds=seconds, execution_failed=m['execution_failed']))
                counts['new_calls'] += 1
            suite.save(directory/'commit.json', dict(seed=seed, protocol_sha256=ph, rows=records, query_targets_accessed=False))
            rows.extend(records)
            print(dict(stage=stage, tasks=seeds.index(seed)+1, total=len(seeds), new_calls=counts['new_calls'],
                       seconds=time.perf_counter()-begin, query_targets_accessed=False), flush=True)
    suite.save(out/'rows.json', rows); checks = verify(root, out, p)
    assert suite.gate(root) == hashes
    suite.save(out/'before_query_manifest.json', dict(protocol_sha256=ph, rows_sha256=suite.sha(out/'rows.json'),
        source_sha256=hashes, checks=checks, counts=counts, query_targets_accessed=False))
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
