"""353 independent scalar risk, pool, support and certificate audit."""
from collections import Counter
import math
from pathlib import Path
import time
import traceback
import numpy as np
from audit_search_radius_development_v1 import read, sha, save, load, same, close, complete
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates

BASE = 'results/unvisited_online'


def run(root, out):
    start = time.perf_counter(); pred = root/BASE/'development_predictions_v1'; scored = root/BASE/'development_evaluation_v1'
    ps, es = complete(pred), complete(scored); p, ep = read(pred/'protocol.json'), read(scored/'protocol.json')
    pre = root/BASE/'preflight_predictions_v1'; complete(pre)
    assert p['source_sha256'] == ep['source_sha256'] == read(pre/'protocol.json')['source_sha256']
    for name, digest in p['source_sha256'].items(): assert sha(root/name) == digest
    seal = read(pred/'before_query_manifest.json'); assert not seal['query_targets_accessed']
    assert seal['rows_sha256'] == sha(pred/'rows.json') and seal['protocol_sha256'] == sha(pred/'protocol.json')
    assert seal['source_sha256'] == p['source_sha256']
    assert ep['prediction_summary_sha256'] == sha(pred/'summary.json') and ep['before_query_manifest_sha256'] == sha(pred/'before_query_manifest.json')
    seeds, names = p['seeds'], p['methods']; assert seeds == list(range(328000000, 328000032)) and len(names) == 137
    rows = read(pred/'rows.json'); index = {(r['seed'], r['method']): r for r in rows}; assert len(rows) == len(index) == 32*137
    z = load(scored/'task_metrics.npz'); metrics = ['mse257', 'mse129', 'point_mse257', 'point_mse129']
    assert list(z['seeds']) == seeds and list(z['methods']) == names and list(z['metric_names']) == metrics
    configs = {c['name']: c for c in p['configs']}; assert len(configs) == 8
    counts = Counter(); risk = np.empty_like(z['risk']); groups = []; mechanisms = []
    census = root/'results/unvisited_language/census_v1'; complete(census)
    cindex = {r['seed']: r for r in read(census/'tasks.json')}
    for i, seed in enumerate(seeds):
        b = np.random.default_rng(seed).uniform(-.12, .12, 4); truth = []
        for q in np.linspace(0, 1, 257):
            y = float(q)
            for bias in b: y = max(0., 1.-abs(2.*(y+float(bias))-1.))
            truth.append(y)
        expected_order = [p['configs'][int(k)]['name'] for k in np.random.default_rng(np.random.SeedSequence([353929, seed])).permutation(8)]
        rr = sorted([r for r in rows if r['seed'] == seed and r['origin'] == 'new_353'], key=lambda r: r['order'])
        assert [r['method'] for r in rr] == expected_order and read(pred/str(seed)/'commit.json')['rows'] == rr
        observed = p['observed_inputs'][str(seed)]; assert sha(root/observed['file']) == observed['sha256']; inputs = load(root/observed['file'])
        cpath = census/str(seed)/'calls.json'; assert sha(cpath) == cindex[seed]['files']['calls.json']; calls = read(cpath)
        for j, name in enumerate(names):
            r = index[seed, name]; assert sha(root/r['file']) == r['sha256'] and sha(root/r['metadata_file']) == r['metadata_sha256']
            a = load(root/r['file']); m = read(root/r['metadata_file'])['metadata']
            assert not m['query_targets_accessed'] and m['execution_failed'] == r['execution_failed']
            counts['input_arrays'] += same(a, inputs, ['x_observed', 'v_observed', 'q_observed'])
            if name == p['primary']: groups.append('execution_failed' if m['execution_failed'] else 'support_fit' if m['selected_state']['support_fit'] else 'fallback')
            if name in configs:
                cfg = configs[name]; assert r['origin'] == 'new_353' and r['seconds'] == m['charged_complete_seconds'] > 0
                if seed == seeds[0]: counts['preflight_replay_arrays'] += same(a, load(pre/str(seed)/(name+'.npz')), list(a))
                if not m['execution_failed']:
                    oldr = index[seed, cfg['reference_name']]; olda = load(root/oldr['file']); oldm = read(root/oldr['metadata_file'])['metadata']
                    assert m['restarts'] == m['atomic_trial_points'] == 629 and m['prior_starts'] == 17
                    assert m['visited_modes'] == oldm['visited_modes'] and m['original_positive_modes'] == oldm['original_positive_modes']
                    assert m['trajectory_state_sha256'] == oldm['trajectory_state_sha256']; counts['trajectory_states'] += len(m['trajectory_state_sha256'])
                    assert m['no_global_bp_guard_enabled'] == (m['solver'] != 'adam' and m['channel'] != 'bp')
                    if 'channel' in cfg:
                        counts['invariant_arrays'] += same(a, olda, ['best_bank', 'selected_b', 'point_prediction', 'trigger_b', 'trigger_h', 'trigger_u', 'trigger_credit', 'trigger_location'])
                        assert m['selected_state'] == oldm['selected_state'] and set(oldm['positive_modes']) <= set(m['positive_modes'])
                        c = calls[cfg['channel']]
                        assert {k: val for k, val in m['proposal'].items() if k != 'meta'} == {k: val for k, val in c['new'].items() if k != 'meta'}
                        assert m['new_positive_modes'] == c['new_positive_modes'] and m['visited_excluded_before_k']
                        proposed = {pp['mode'] for pp in m['proposal']['proposals']}
                        assert not proposed & set(m['visited_modes'])
                        assert m['proposal']['meta']['forbidden_modes'] == len(m['visited_modes'])
                        assert m['visited_source'] == 'Own current live trajectory, not old files or another solver'
                        mechanisms.append(dict(seed=seed, method=name, new_positive_modes=m['new_positive_modes'],
                            additional_positive_vs_old=sorted(set(m['positive_modes'])-set(oldm['positive_modes'])),
                            candidate_count=len(proposed), trie_nodes=m['proposal']['meta']['trie_nodes'],
                            max_dp_entries=m['proposal']['meta']['max_retained_dp_entries']))
                    else: counts['native_replay_arrays'] += same(a, olda, list(a))
                    if m['positive_modes'] == oldm['positive_modes']: counts['identical_pool_arrays'] += same(a, olda, ['points', 'allocation', 'prediction'])
                    if m['positive_modes']:
                        yy = np.broadcast_to(a['x_observed'], (len(a['points']), 4))
                        for l in range(4): yy = np.maximum(0., 1.-abs(2*(yy+a['points'][:, l, None])-1.))
                        assert float(np.max(abs(yy-a['v_observed']))) <= .001+1e-7; counts['support_particles'] += len(a['points'])
                    for key, note in m['new_mode_classifications_detail'].items():
                        aa, rhs = inequalities(a['x_observed'], a['v_observed'], key); counts['independent_certificates'] += certificates(aa, rhs, note)
            else: assert r['origin'] == 'frozen_pre353' and r['seconds'] is None
            values = []
            for field in ['prediction', 'point_prediction']:
                assert a[field].shape == (257,) and np.isfinite(a[field]).all() and np.all((a[field] >= 0) & (a[field] <= 1))
                for stride in [1, 2]: values.append(math.fsum((float(v)-t)**2 for v, t in zip(a[field][::stride], truth[::stride]))/len(truth[::stride]))
            risk[i, j] = values; counts['risk_fields'] += 4
    gap = float(np.max(abs(risk-z['risk']))); assert gap < 2e-12 and groups == list(z['groups'])
    methods = read(scored/'methods.json'); assert len(methods) == len(names)
    for m in methods:
        name = m['method']; j = names.index(name); rr = [index[s, name] for s in seeds]
        assert m['failures'] == sum(r['execution_failed'] for r in rr) and m['current_development_time_available'] == (name in configs)
        if name in configs: close(m['mean_current_seconds'], math.fsum(r['seconds'] for r in rr)/len(seeds))
        else: assert m['mean_current_seconds'] is None
        for k, metric in enumerate(metrics):
            close(m['metrics'][metric], math.fsum(map(float, risk[:, j, k]))/len(seeds))
            for g in set(groups): close(m['group_metrics'][g][metric], math.fsum(float(risk[ii, j, k]) for ii, gg in enumerate(groups) if gg == g)/groups.count(g))
    comparisons = read(scored/'comparisons.json'); expected = {(c, b, k) for c in configs for b in names if b != c for k in metrics}
    assert len(comparisons) == len(expected) == 4352 and {(r['candidate'], r['control'], r['metric']) for r in comparisons} == expected
    for r in comparisons:
        d = z['risk'][:, names.index(r['candidate']), metrics.index(r['metric'])]-z['risk'][:, names.index(r['control']), metrics.index(r['metric'])]
        total = math.fsum(map(float, d)); largest = int(np.argmax(abs(d))); absolute = math.fsum(map(float, abs(d))); n = len(seeds)
        close(r['mean_difference'], total/n); assert r['improved'] == sum(d < 0) and r['equal'] == sum(d == 0) and r['worse'] == sum(d > 0)
        assert r['largest_absolute_task_seed'] == seeds[largest]; close(r['largest_absolute_task_delta'], float(d[largest]))
        close(r['largest_absolute_share'], float(abs(d[largest])/absolute) if absolute else 0.)
        close(r['leave_largest_absolute_out_mean'], (total-float(d[largest]))/(n-1))
        close(r['worst_leave_one_out_mean'], max((total-float(v))/(n-1) for v in d)); counts['comparison_rows'] += 1
    save(out/'mechanisms.json', mechanisms)
    summary = dict(passed=True, counts=dict(counts), maximum_scalar_risk_gap=gap,
        prediction_summary_sha256=sha(pred/'summary.json'), evaluation_summary_sha256=sha(scored/'summary.json'),
        source_sha256=sha(Path(__file__)), seconds=time.perf_counter()-start, core_research_goal_complete=False,
        outputs_sha256={'mechanisms.json': sha(out/'mechanisms.json')})
    save(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'development_audit_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
