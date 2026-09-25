"""340 independent scalar truth, array replay, ordering and every descriptive result."""
import hashlib
import json
import math
from pathlib import Path
import traceback
import numpy as np

BASE = 'results/search_radius_development'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path, value):
    with path.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, allow_nan=False)
def load(path):
    with np.load(path, allow_pickle=False) as z: return {n: z[n] for n in z.files}
def close(a, b): assert abs(a-b) < 2e-12, (a, b)
def same(a, b, keys):
    for key in keys:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        assert a[key].tobytes() == b[key].tobytes(), key
    return len(keys)


def complete(folder):
    s = read(folder/'summary.json'); assert s['passed']
    for n, digest in s.get('outputs_sha256', {}).items(): assert sha(folder/n) == digest
    return s


def run(root, out):
    pred = root/BASE/'development_predictions_v1'; scored = root/BASE/'development_evaluation_v1'
    ps = complete(pred); es = complete(scored); p = read(pred/'protocol.json'); ep = read(scored/'protocol.json')
    assert p['seeds'] == list(range(328000000, 328000032))
    assert not p['query_targets_accessed'] and ep['query_targets_accessed'] and not ep['new_blind_tasks']
    assert p['source_sha256'] == ep['source_sha256']
    for path, digest in p['source_sha256'].items(): assert sha(root/path) == digest
    assert ep['prediction_summary_sha256'] == sha(pred/'summary.json')
    assert ep['before_query_manifest_sha256'] == sha(pred/'before_query_manifest.json')
    seal = read(pred/'before_query_manifest.json')
    assert not seal['query_targets_accessed'] and seal['source_sha256'] == p['source_sha256']
    assert seal['protocol_sha256'] == sha(pred/'protocol.json') and seal['rows_sha256'] == sha(pred/'rows.json')
    rows = read(pred/'rows.json'); seeds, names = p['seeds'], p['methods']; n = len(seeds)
    lookup = {(r['seed'], r['method']): r for r in rows}
    assert len(rows) == len(lookup) == n*len(names)
    assert set(lookup) == {(s, m) for s in seeds for m in names}
    z = load(scored/'task_metrics.npz'); assert list(z['seeds']) == seeds and list(z['methods']) == names
    assert list(z['metric_names']) == ['mse257', 'mse129', 'point_mse257', 'point_mse129']
    risk = np.empty_like(z['risk']); checks = replay = nested = particle_count = 0; labels = []
    pre = root/BASE/'preflight_predictions_v1'; complete(pre)
    assert read(pre/'protocol.json')['source_sha256'] == p['source_sha256']
    preindex = {(r['seed'], r['method']): r for r in read(pre/'rows.json')}
    old = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'; complete(old)
    oldindex = {(r['seed'], r['method']): r for r in read(old/'rows.json')}
    configs = {c['name']: c for c in p['configs']}
    for i, seed in enumerate(seeds):
        bias = np.random.default_rng(seed).uniform(-.12, .12, 4); target = []
        for x in np.linspace(0, 1, 257):
            y = float(x)
            for b in bias: y = max(0., 1.-abs(2.*(y+float(b))-1.))
            target.append(y)
        rr = sorted([r for r in rows if r['seed'] == seed and r['origin'] == 'new_340'], key=lambda r: r['order'])
        expected = [p['configs'][int(j)]['name'] for j in np.random.default_rng(np.random.SeedSequence([340929, seed])).permutation(len(configs))]
        assert [r['method'] for r in rr] == expected
        assert read(pred/str(seed)/'commit.json')['rows'] == rr
        case = {}; meta = {}
        for j, name in enumerate(names):
            r = lookup[seed, name]
            assert sha(root/r['file']) == r['sha256'] and sha(root/r['metadata_file']) == r['metadata_sha256']
            a = load(root/r['file']); m = read(root/r['metadata_file'])['metadata']; case[name] = a; meta[name] = m
            assert not m['query_targets_accessed'] and m['execution_failed'] == r['execution_failed']
            input_ref = p['observed_inputs'][str(seed)]; assert sha(root/input_ref['file']) == input_ref['sha256']
            observed = load(root/input_ref['file']); checks += same(a, observed, ['x_observed', 'v_observed', 'q_observed'])
            if r['origin'] == 'frozen_328':
                prior = oldindex[seed, name]
                assert sha(old/prior['file']) == r['sha256'] and sha(old/prior['metadata_file']) == r['metadata_sha256']
                assert r['seconds'] is None
            else:
                assert r['origin'] == 'new_340' and r['seconds'] == m['charged_complete_seconds'] and r['seconds'] > 0
                if seed == seeds[0]:
                    prior = preindex[seed, name]; assert sha(root/prior['file']) == prior['sha256']
                    replay += same(a, load(root/prior['file']), list(a))
                cfg = configs[name]
                if cfg['family'] in ['radius_online', 'budgeted_online_credit'] and not m['execution_failed']:
                    prior = oldindex[seed, cfg['base_config']['name']]
                    olda = load(old/prior['file']); oldm = read(old/prior['metadata_file'])['metadata']
                    assert m['selected_state'] == oldm['selected_state'] and m['original_positive_modes'] == oldm['original_positive_modes']
                    assert m['no_global_bp_guard_enabled'] == (cfg['base_config']['channel'] != 'bp')
                    checks += same(a, olda, list(set(a)-{'points', 'allocation', 'prediction'}))
                    if cfg['family'] == 'budgeted_online_credit' or cfg['k'] == 8:
                        assert set(oldm['positive_modes']) <= set(m['positive_modes'])
                    if oldm['positive_modes'] == m['positive_modes']: checks += same(a, olda, ['points', 'allocation', 'prediction'])
                if not m['execution_failed'] and m.get('positive_modes'):
                    h = np.broadcast_to(a['x_observed'], (len(a['points']), 4))
                    for layer in range(4): h = np.maximum(0., 1.-abs(2*(h+a['points'][:, layer, None])-1.))
                    assert float(np.max(abs(h-a['v_observed']))) <= .001+1e-7; particle_count += len(a['points'])
            values = []
            for field in ['prediction', 'point_prediction']:
                assert a[field].shape == (257,) and np.isfinite(a[field]).all() and np.all((a[field] >= 0) & (a[field] <= 1))
                for stride in [1, 2]:
                    values.append(math.fsum((float(u)-v)**2 for u, v in zip(a[field][::stride], target[::stride]))/len(target[::stride]))
            risk[i, j] = values
        main = meta[p['primary']]
        labels.append('execution_failed' if main['execution_failed'] else 'support_fit' if main['selected_state']['support_fit'] else 'fallback')
        for channel in ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']:
            for k1, k2 in [(1, 2), (2, 4), (4, 8)]:
                an, bn = [f'radius_first_fit_{channel}__k{k}__sall' for k in [k1, k2]]
                a, b, ma, mb = case[an], case[bn], meta[an], meta[bn]
                if ma['execution_failed'] or mb['execution_failed']: continue
                assert set(ma['positive_modes']) <= set(mb['positive_modes'])
                assert all(v in mb['proposal']['proposals'] for v in ma['proposal']['proposals'])
                checks += same(a, b, list(set(a)-{'points', 'allocation', 'prediction'})); nested += 1
                if ma['positive_modes'] == mb['positive_modes']: checks += same(a, b, ['points', 'allocation', 'prediction'])
    gap = float(np.max(abs(risk-z['risk']))); assert gap < 2e-12 and list(z['groups']) == labels
    methods = read(scored/'methods.json'); assert len(methods) == len(names)
    for m in methods:
        j = names.index(m['method']); rr = [lookup[s, m['method']] for s in seeds]
        assert m['failures'] == sum(r['execution_failed'] for r in rr)
        current = m['method'] in configs; assert m['current_development_time_available'] == current
        if current: close(m['mean_current_seconds'], math.fsum(r['seconds'] for r in rr)/n)
        else: assert m['mean_current_seconds'] is None
        for k, metric in enumerate(z['metric_names']):
            close(m['metrics'][metric], math.fsum(map(float, risk[:, j, k]))/n)
            for g in set(labels): close(m['group_metrics'][g][metric], math.fsum(float(risk[i, j, k]) for i, label in enumerate(labels) if label == g)/labels.count(g))
    comparisons = read(scored/'comparisons.json'); expected = {(c, b, metric) for c in ep['candidates'] for b in names if b != c for metric in z['metric_names']}
    assert {(r['candidate'], r['control'], r['metric']) for r in comparisons} == expected and len(comparisons) == len(expected)
    for r in comparisons:
        # Every risk cell was independently checked above. Preserve the sealed
        # matrix's exact floating-point tie convention for discrete sign counts.
        k = list(z['metric_names']).index(r['metric']); values = z['risk'][:, names.index(r['candidate']), k]-z['risk'][:, names.index(r['control']), k]
        total = math.fsum(map(float, values)); i = int(np.argmax(abs(values))); absolute = math.fsum(map(float, abs(values)))
        close(r['mean_difference'], total/n); assert r['descriptive_only']
        for key, wanted in [('improved', sum(values < 0)), ('equal', sum(values == 0)), ('worse', sum(values > 0))]: assert r[key] == wanted
        assert r['largest_absolute_task_seed'] == seeds[i]; close(r['largest_absolute_task_delta'], float(values[i]))
        close(r['largest_absolute_share'], abs(float(values[i]))/absolute if absolute else 0.)
        close(r['leave_largest_absolute_out_mean'], (total-float(values[i]))/(n-1))
        close(r['worst_leave_one_out_mean'], max((total-float(v))/(n-1) for v in values))
    assert not es['statistical_superiority_claim'] and not es['matched_resource_superiority_claim'] and not es['core_research_goal_complete']
    assert ps['new_calls'] == n*len(configs) and ps['cached_predictors'] == n*51
    result = dict(passed=True, tasks=n, methods=len(names), independent_risk_fields=int(risk.size), maximum_risk_gap=gap,
        invariant_array_checks=checks, preflight_replay_arrays=replay, nested_pairs=nested, support_particles=particle_count,
        comparison_rows=len(comparisons), prediction_summary_sha256=sha(pred/'summary.json'), evaluation_summary_sha256=sha(scored/'summary.json'),
        query_targets_accessed=True, new_blind_tasks=False, core_research_goal_complete=False, auditor_source_sha256=sha(Path(__file__)))
    save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'development_audit_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
