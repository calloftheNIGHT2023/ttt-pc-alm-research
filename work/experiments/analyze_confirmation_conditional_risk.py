"""266C / 267: all frozen predictors, no query answers, fixed cross-batch risks."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha, dump

METRICS = ['actual_excess', 'policy_excess', 'read_error', 'cross_term',
           'expected_mc', 'resampled_expected_excess', 'bayes_variance',
           'conditional_total']
PAIRED_METRICS = ['actual_excess', 'policy_excess', 'resampled_expected_excess']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    src = Path(__file__).parent
    base = root/'results/confirmation_conditional_risk'
    inp, audit = base/'moments', base/'moments_audit'
    out = base/'decomposition'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    aa = json.loads((audit/'summary.json').read_text())
    assert aa['passed'] and not aa['phase_accesses_query_targets']
    pa = json.loads((audit/'protocol.json').read_text())
    assert sha(inp/'summary.json') == pa['moments_summary_sha256']
    hashes = dict(pa['source_sha256'])
    hashes[Path(__file__).name] = sha(Path(__file__))
    for n, h in hashes.items():
        assert sha(src/n) == h, n
    bp = json.loads((inp/'protocol.json').read_text())
    bs = json.loads((inp/'summary.json').read_text())
    assert sha(inp/'tasks.json') == bs['tasks_sha256']
    tasks = json.loads((inp/'tasks.json').read_text())
    old = root/'results/matched_budget_confirmation/conditioned_confirmation'
    manifest = json.loads((old/'before_query_manifest.json').read_text())
    for n in ['protocol', 'rows']:
        assert sha(old/f'{n}.json') == manifest[f'{n}_sha256']
    op = json.loads((old/'protocol.json').read_text())
    rows0 = json.loads((old/'rows.json').read_text())
    lookup = {(r['seed'], r['method']): r for r in rows0}
    names = [c['name'] for c in op['configs']]
    seeds = bp['seeds']
    assert len(names) == 46 and len(seeds) == 64 and len(lookup) == 2944
    ref_audit = base/'reference_audit'
    ras = json.loads((ref_audit/'summary.json').read_text())
    assert sha(ref_audit/'summary.json') == bp['reference_audit_sha256']
    assert sha(ref_audit/'coverage.json') == ras['coverage_sha256']
    coverage = {r['seed']: r for r in json.loads((ref_audit/'coverage.json').read_text())}
    docs = root/'outputs/ttt-pc-alm-research'
    designs = {n: sha(docs/n) for n in ['266_confirmation_conditional_risk_protocol.md',
                                      '267_conditional_risk_accounting_addendum.md']}
    assert designs['266_confirmation_conditional_risk_protocol.md'] == bp['design_sha256']
    protocol = dict(source_sha256=hashes, design_sha256=designs,
        moments_audit_sha256=sha(audit/'summary.json'),
        frozen_predictor_manifest_sha256=sha(old/'before_query_manifest.json'),
        seeds=seeds, methods=names, primary=op['primary'], grids=[257, 129],
        batch_pairs=[[0, 1], [2, 3]], metrics=METRICS, paired_metrics=PAIRED_METRICS,
        bootstrap_seed=262193, bootstrap_replicates=20000, particles_per_readout=2048,
        phase_accesses_query_targets=False,
        scope='Post-result diagnostic; arithmetic grid means; all46x64 retained; empty pools and heads use frozen p as strategy; unadjusted descriptive intervals, no method tuning')
    dump(out/'protocol.json', protocol)
    begin = time.perf_counter()
    rows, files, sensitivity, reference_gaps = [], {}, [], []
    max_identity, max_total = 0., 0.
    for task in tasks:
        seed = task['seed']
        assert seed in seeds and coverage[seed]['complete_up_to_certified_zero_volume']
        path = inp/task['file']
        assert sha(path) == task['sha256']
        with np.load(path) as z:
            q, volumes = z['q'].copy(), z['volumes'].copy()
            means, second = z['means'].copy(), z['seconds'].copy()
        weights = volumes/volumes.sum()
        full = np.einsum('k,kbq->bq', weights, means)
        fullsecond = np.einsum('k,kbq->bq', weights, second)
        masks = np.zeros((46, len(weights)), dtype=bool)
        predictions = np.empty((46, 257))
        strategies = np.empty((46, 4, 257))
        strategy_seconds = np.empty_like(strategies)
        kinds = []
        masses = []
        cover = {r['method']: r for r in coverage[seed]['coverage']}
        for j, name in enumerate(names):
            r = lookup[seed, name]
            assert sha(old/r['file']) == r['sha256'] == manifest['prediction_files'][r['file']]
            with np.load(old/r['file']) as z:
                p = z['prediction'].copy()
                assert p.shape == (257,) and np.isfinite(p).all()
                assert q.tobytes() == z['q_observed'].tobytes()
            predictions[j] = p
            if r['readout'] == 'mode':
                keys = r['metadata']['positive_modes']
                assert sorted(keys) == sorted(cover[name]['found_modes'])
                assert set(keys) <= set(task['keys'])
                masks[j] = np.isin(task['keys'], keys)
                mass = float(weights[masks[j]].sum())
                assert abs(mass-cover[name]['numerical_posterior_mass_fraction']) < 1e-12
                kind = 'nonempty_pool' if keys else 'empty_pool_fallback'
                if keys:
                    w = weights*masks[j]/mass
                    strategies[j] = np.einsum('k,kbq->bq', w, means)
                    strategy_seconds[j] = np.einsum('k,kbq->bq', w, second)
                else:
                    strategies[j] = p
                    strategy_seconds[j] = p*p
            else:
                kind, mass = 'fixed_prediction_head', None
                strategies[j] = p
                strategy_seconds[j] = p*p
            kinds.append(kind)
            masses.append(mass)
            taskrows = {}
            for grid in protocol['grids']:
                ii = np.arange(257) if grid == 257 else np.arange(0, 257, 2)
                pp = p[ii]
                estimates = []
                for a, b in protocol['batch_pairs']:
                    ma, mb = full[a, ii], full[b, ii]
                    ka, kb = strategies[j, a, ii], strategies[j, b, ii]
                    actual = float(np.mean((pp-ma)*(pp-mb)))
                    policy = float(np.mean((ka-ma)*(kb-mb)))
                    read = float(np.mean((pp-ka)*(pp-kb)))
                    cross = float(np.mean((ka-ma)*(pp-kb)) + np.mean((kb-mb)*(pp-ka)))
                    mc = float(np.mean((strategy_seconds[j,a,ii]+strategy_seconds[j,b,ii])/2-ka*kb)/2048) if kind == 'nonempty_pool' else 0.
                    bayes = float(np.mean((fullsecond[a,ii]+fullsecond[b,ii])/2-ma*mb))
                    total = bayes+actual
                    identity = abs(actual-policy-read-cross)
                    totalgap = abs(total-np.mean(pp*pp-pp*(ma+mb)+(fullsecond[a,ii]+fullsecond[b,ii])/2))
                    assert identity < 1e-12 and totalgap < 1e-12
                    max_identity, max_total = max(max_identity, identity), max(max_total, totalgap)
                    estimates.append(dict(pair=[a,b], actual_excess=actual, policy_excess=policy,
                        read_error=read, cross_term=cross, expected_mc=mc,
                        resampled_expected_excess=policy+mc, bayes_variance=bayes,
                        conditional_total=total, decomposition_residual=identity, total_residual=float(totalgap)))
                averages = {k:float(np.mean([e[k] for e in estimates])) for k in METRICS}
                rr = dict(seed=seed, method=name, grid=grid, kind=kind, posterior_mass=mass,
                    found_modes=int(masks[j].sum()) if r['readout']=='mode' else None,
                    complete_call_seconds=r['seconds'], pairs=estimates, **averages)
                rows.append(rr)
                taskrows[grid] = rr
            sensitivity.append(dict(seed=seed, method=name,
                grid_257_minus_129={k:taskrows[257][k]-taskrows[129][k] for k in METRICS},
                pair01_minus_pair23={k:taskrows[257]['pairs'][0][k]-taskrows[257]['pairs'][1][k] for k in METRICS}))
        for grid in protocol['grids']:
            ii = np.arange(257) if grid == 257 else np.arange(0,257,2)
            reference_gaps.append(dict(seed=seed,grid=grid,
                pair_l2=[float(np.mean((full[a,ii]-full[b,ii])**2)) for a,b in protocol['batch_pairs']]))
        filename = f'{seed}_curves.npz'
        np.savez_compressed(out/filename, q=q, weights=weights, keys=np.array(task['keys']),
            methods=np.array(names), kinds=np.array(kinds), masks=masks, predictions=predictions,
            full_means=full, full_second_moments=fullsecond, strategy_means=strategies,
            strategy_second_moments=strategy_seconds)
        files[filename] = sha(out/filename)
        print(json.dumps(dict(tasks_done=len(files),total=64,seconds=time.perf_counter()-begin)),flush=True)
    bykey = {(r['seed'],r['method'],r['grid']):r for r in rows}
    method_summary, paired = [], []
    ids = np.random.default_rng(262193).integers(0,64,(20000,64))
    for grid in protocol['grids']:
        for name in names:
            rr = [bykey[s,name,grid] for s in seeds]
            mean = {k:float(np.mean([r[k] for r in rr])) for k in METRICS}
            method_summary.append(dict(method=name,grid=grid,**mean,
                empty_pool_tasks=sum(r['kind']=='empty_pool_fallback' for r in rr),
                nonempty_pool_tasks=sum(r['kind']=='nonempty_pool' for r in rr),
                head_tasks=sum(r['kind']=='fixed_prediction_head' for r in rr),
                negative_task_estimates={k:sum(r[k]<0 for r in rr) for k in METRICS},
                mean_complete_call_seconds=float(np.mean([r['complete_call_seconds'] for r in rr])),
                mean_pair01_minus_pair23={k:float(np.mean([r['pairs'][0][k]-r['pairs'][1][k] for r in rr])) for k in METRICS},
                maximum_abs_pair_difference={k:float(max(abs(r['pairs'][0][k]-r['pairs'][1][k]) for r in rr)) for k in METRICS}))
            if name == protocol['primary']:
                continue
            for metric in PAIRED_METRICS:
                delta = np.array([bykey[s,protocol['primary'],grid][metric]-bykey[s,name,grid][metric] for s in seeds])
                paired.append(dict(primary=protocol['primary'],control=name,grid=grid,metric=metric,
                    mean_difference=float(delta.mean()),descriptive_ci95=np.quantile(delta[ids].mean(1),[.025,.975]).tolist(),
                    lower_tasks=int((delta < -1e-12).sum()),equal_tasks=int((abs(delta)<=1e-12).sum()),higher_tasks=int((delta>1e-12).sum())))
    dump(out/'rows.json', rows)
    dump(out/'methods.json', method_summary)
    dump(out/'paired.json', paired)
    dump(out/'sensitivity.json', sensitivity)
    dump(out/'reference_gaps.json', reference_gaps)
    dump(out/'files.json', files)
    for n,h in hashes.items():
        assert sha(src/n)==h,n
    ans = dict(passed=True,tasks=64,methods=46,grids=2,task_method_grid_rows=len(rows),
        cross_pair_rows=2*len(rows),paired_intervals=len(paired),curve_files=len(files),
        max_decomposition_residual=max_identity,max_total_risk_residual=max_total,
        phase_accesses_query_targets=False,seconds=time.perf_counter()-begin,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','rows.json','methods.json','paired.json',
            'sensitivity.json','reference_gaps.json','files.json']},
        next='Independent decomposition audit before scientific interpretation; no changed predictors or new blind tasks')
    dump(out/'summary.json',ans)
    print(json.dumps(ans),flush=True)


if __name__ == '__main__':
    main()
