"""Full-state equivalence and task-paired complete-cost audit of study 109."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import batched_credit_bank_memory as model


def ci(delta, alpha=.05):
    ids = np.random.default_rng(88761).integers(0, len(delta), (20000, len(delta)))
    return np.quantile(delta[ids].mean(1), [alpha / 2, 1 - alpha / 2]).tolist()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); a = p.parse_args()
    root = a.project / 'results/batched_credit_bank'; inp = root / 'development'; out = root / 'analysis'
    refroot = a.project / 'results/split_activity_modes/development'
    protocol = json.loads((inp / 'protocol.json').read_text()); rows = json.loads((inp / 'episodes.json').read_text())
    configs = {c['name']: c for c in protocol['configs']}; seeds = list(range(protocol['seed0'], protocol['seed0'] + protocol['count']))
    assert len(rows) == len(configs) * len(seeds) * protocol['repetitions'] * len(protocol['stages'])
    for name, expected in protocol['source_sha256'].items():
        assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256(Path(__file__).with_name('batched_credit_bank_development.json').read_bytes()).hexdigest() == protocol['config_sha256']
    refs = {(r['seed'], r['method'], r['n_context']): r for r in json.loads((refroot / 'episodes.json').read_text())}
    lookup = {(r['repetition'], r['seed'], r['method'], r['n_context']): r for r in rows}; assert len(lookup) == len(rows)
    cache = {}; cuts = 0
    for row in rows:
        key = row['seed'], configs[row['method']]['reference'], row['n_context']; ref = refs[key]
        if key not in cache:
            path = refroot / ref['state_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == ref['state_sha256']
            with np.load(path) as z: cache[key] = z['anchor'].copy(), z['samples'].copy()
            rng = np.random.default_rng(row['seed']); truth = rng.uniform(-.12, .12, 4); rng.uniform(0, 1, 24); q = rng.uniform(0, 1, protocol['queries'])
            pred = model.posterior.make_predict(cache[key][1])(q)
            assert float(np.mean((pred - model.base.forward(q, truth)) ** 2)) == ref['raw_query_mse']
        path = inp / row['state_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == row['state_sha256']
        with np.load(path) as z:
            assert np.array_equal(z['anchor'], cache[key][0]) and np.array_equal(z['samples'], cache[key][1]), key
        for field in ['query_mse', 'raw_query_mse', 'positive_mode_keys']: assert row[field] == ref[field], (key, field)
        if row['n_context'] != 4: assert not row['batch_credit_active'] and row['effective_generator'] == 'direct'
        if row['batch_credit_active']:
            for prefix in ['matching', 'cross']:
                keys = row[prefix + '_mode_keys']; lower = row[prefix + '_lower_bounds']
                assert len(keys) == len(lower) == row[prefix + '_certified_patterns']
                assert all(b > 0 for b in lower) and not set(keys) & set(row['positive_mode_keys']); cuts += len(keys)
            assert not set(row['matching_mode_keys']) & set(row['cross_mode_keys'])
            assert row['retained_geometry_patterns'] == row['geometry_calls']
    summary = []; vectors = {}; times = {}
    for name, cfg in configs.items():
        tt = np.array([[[lookup[rep, s, name, n]['adaptation_seconds'] + lookup[rep, s, name, n]['read_queries_seconds'] for n in protocol['stages']] for s in seeds] for rep in range(protocol['repetitions'])])
        total = tt.sum(2); vectors[name] = total.mean(0); times[name] = tt
        first = [lookup[rep, s, name, 4] for rep in range(protocol['repetitions']) for s in seeds]
        geometry = [r.get('geometry_calls', r.get('initial_geometry_calls', 0) + r.get('completion_geometry_calls', 0)) for r in first]
        credit = [r.get('credit_collection_seconds', 0) + r.get('strict_matching_seconds', 0) + r.get('cross_credit_seconds', 0) + r.get('credit_check_seconds', 0) for r in first]
        entry = dict(method=name, reference=cfg['reference'], query_mse=float(np.mean([lookup[0, s, name, n]['query_mse'] for s in seeds for n in protocol['stages']])),
            mean_full_seconds=float(vectors[name].mean()), median_task_mean_full_seconds=float(np.median(vectors[name])), mean_full_seconds_by_repeat=total.mean(1).tolist(),
            mean_first_seconds=float(tt[:, :, 0].mean()), mean_first_geometry_calls=float(np.mean(geometry)), mean_first_credit_seconds=float(np.mean(credit)),
            maximum_old_plus_new_state_bytes=max(r['persistent_state_bytes'] + r['previous_state_bytes'] for r in rows if r['method'] == name))
        for field in ['matching_certified_patterns', 'cross_certified_patterns', 'direction_bank_size', 'discovery_seconds', 'strict_matching_seconds', 'cross_credit_seconds']:
            entry['mean_first_' + field] = float(np.mean([r.get(field, 0) for r in first]))
        for field in ['pending_credit_numeric_bytes', 'direction_bank_numeric_bytes', 'main_cross_batch_array_bytes_subtotal']:
            entry['max_' + field] = max(r.get(field, 0) for r in first)
        summary.append(entry)
    candidate = protocol['primary_candidate']; paired = []
    comparisons = [(candidate, name) for name in configs if name != candidate] + [(f'{name}_bp_bank', f'{name}_none') for name in ['adam8', 'adam16']]
    for left, right in comparisons:
        delta = vectors[left] - vectors[right]; ratio = vectors[left] / vectors[right] - 1
        first_delta = (times[left][:, :, 0] - times[right][:, :, 0]).mean(0)
        later_delta = (times[left][:, :, 1:] - times[right][:, :, 1:]).sum(2).mean(0)
        paired.append(dict(candidate=left, comparator=right, primary=left == candidate and right in protocol['primary_cost_comparators'], mean_full_cost_difference=float(delta.mean()),
            descriptive_ci95=ci(delta), mean_task_relative_change=float(ratio.mean()), relative_change_ci95=ci(ratio), faster_tasks=int((delta < 0).sum()), tasks=len(delta),
            first_write_mean_difference=float(first_delta.mean()), first_write_descriptive_ci95=ci(first_delta),
            later_writes_mean_difference=float(later_delta.mean()), later_write_descriptive_ci95=ci(later_delta),
            three_comparison_bonferroni_interval=ci(delta, .05 / 3) if left == candidate and right in protocol['primary_cost_comparators'] else None))
    result = dict(audit=dict(source_hashes=len(protocol['source_sha256']), state_hashes_and_exact_reference_matches=len(rows), independently_replayed_unique_reference_states=len(cache),
        positive_bound_records=cuts, all_positive_sets_unchanged=True, later_stages_direct_only=True), summary=summary, paired=paired,
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='16 old tasks; 2 timing repeats averaged within task; descriptive task bootstrap, not independent confirmation')
    out.mkdir(parents=True, exist_ok=True); (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 7), sharey=True); pos = np.arange(len(summary))
    colors = ['#167e79' if r['method'].startswith('alm') else '#d79434' if r['method'].startswith('adam') else '#778995' for r in summary]
    for ax, field, label in zip(axes, ['mean_full_seconds', 'mean_first_geometry_calls', 'mean_first_credit_seconds'], ['Mean full stream fit + read (s)', 'First-write geometry calls', 'All credit handling (ms)']):
        values = np.array([r[field] for r in summary]) * (1000 if field.endswith('credit_seconds') else 1)
        ax.barh(pos, values, color=colors); ax.set_xlim(0, float(values.max()) * 1.22); ax.grid(axis='x', alpha=.2); ax.set_xlabel(label)
        for y, value in zip(pos, values): ax.text(value + values.max() * .02, y, f'{value:.3f}' if field == 'mean_full_seconds' else f'{value:.1f}', va='center', fontsize=7)
    axes[0].set_yticks(pos, [r['method'] for r in summary], fontsize=8); axes[0].invert_yaxis()
    fig.suptitle('Common batched credit banks: actual total cost, bitwise-preserved predictions'); fig.tight_layout(); fig.savefig(out / 'batched_credit_bank_results.png', dpi=170); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
