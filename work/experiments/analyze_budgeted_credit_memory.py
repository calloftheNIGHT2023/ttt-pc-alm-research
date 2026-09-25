"""Budgeted online output audits, conditional risk, and paired complete costs."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import budgeted_credit_memory as model


def ci(delta):
    ids = np.random.default_rng(192823).integers(0, len(delta), (20000, len(delta)))
    return np.quantile(delta[ids].mean(1), [.025, .975]).tolist()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', type=Path, required=True); args = p.parse_args()
    root = args.project / 'results/budgeted_credit_memory'; inp = root / 'development'; out = root / 'analysis'
    protocol = json.loads((inp / 'protocol.json').read_text()); rows = json.loads((inp / 'episodes.json').read_text())
    cfgs = {c['name']: c for c in protocol['configs']}; seeds = list(range(protocol['seed0'], protocol['seed0'] + protocol['count'])); stages = protocol['stages']
    assert len(rows) == len(cfgs) * len(seeds) * len(stages) * protocol['repetitions']
    for name, h in protocol['source_sha256'].items(): assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == h, name
    assert hashlib.sha256(Path(__file__).with_name('budgeted_credit_development.json').read_bytes()).hexdigest() == protocol['config_sha256']
    lookup = {(r['repetition'], r['seed'], r['method'], r['n_context']): r for r in rows}; assert len(lookup) == len(rows)
    oldroot = args.project / 'results/batched_credit_bank/development'; oldrows = {(r['seed'], r['method'], r['n_context']): r for r in json.loads((oldroot / 'episodes.json').read_text()) if r['repetition'] == 0}
    diag = {(r['seed'], r['method']): r for r in json.loads((args.project / 'results/finite_geometry_prefix/diagnostic/audits.json').read_text())}
    refroot = args.project / 'results/posterior_state_reuse'; cache = {}; oldcache = {}; full_matches = 0; prefix_matches = 0; conditional = []
    for seed in seeds:
        rng = np.random.default_rng(seed); truth = rng.uniform(-.12, .12, 4); rng.uniform(0, 1, 24); q = rng.uniform(0, 1, protocol['queries']); target = model.base.forward(q, truth)
        audit = json.loads((refroot / f'conditional_risk/audit_{seed}.json').read_text()); path = refroot / 'conditional_risk' / audit['curve_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == audit['curve_sha256']
        with np.load(path) as curve:
            positive = set(curve['patterns']); weights = curve['weights']; means = curve['region_means']; grid = curve['q']; full = np.einsum('k,rkq->rq', weights, means)
            for name, cfg in cfgs.items():
                for rep in range(protocol['repetitions']):
                    for n in stages:
                        row = lookup[rep, seed, name, n]; path = inp / row['state_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == row['state_sha256']
                        with np.load(path) as z: state = z['anchor'].copy(), z['samples'].copy()
                        key = seed, name, n
                        if key not in cache:
                            cache[key] = state; pred = model.old.posterior.make_predict(state[1])(q)
                            assert float(np.mean((pred - target) ** 2)) == row['raw_query_mse']
                            assert float(np.mean((np.clip(pred, 0, 1) - target) ** 2)) == row['query_mse']
                        else:
                            assert np.array_equal(state[0], cache[key][0]) and np.array_equal(state[1], cache[key][1])
                            assert row['query_mse'] == lookup[0, seed, name, n]['query_mse']
                        assert np.array_equal(state[0], row['anchor_output'])
                        if n != 4: assert not row['budget_active'] and row['effective_generator'] == 'direct'
                        if 'full_reference' in cfg:
                            oldkey = seed, cfg['full_reference'], n; oldrow = oldrows[oldkey]
                            if oldkey not in oldcache:
                                path = oldroot / oldrow['state_file']; assert hashlib.sha256(path.read_bytes()).hexdigest() == oldrow['state_sha256']
                                with np.load(path) as z: oldcache[oldkey] = z['anchor'].copy(), z['samples'].copy()
                            assert np.array_equal(state[0], oldcache[oldkey][0]) and np.array_equal(state[1], oldcache[oldkey][1]); full_matches += 1
                        if n == 4 and 'geometry_budget' in cfg:
                            assert row['geometry_calls'] <= cfg['geometry_budget']
                            assert row['evaluated_pattern_keys'] == row['unbudgeted_pattern_keys'][:cfg['geometry_budget']]
                            assert set(row['positive_mode_keys']) == set(row['evaluated_pattern_keys']) & positive
                        if n == 4 and 'diagnostic_method' in cfg:
                            expected = diag[seed, cfg['diagnostic_method']]['sequences'][cfg['diagnostic_screen']]
                            assert row['unbudgeted_pattern_keys'] == expected
                            assert set(row['positive_mode_keys']) == set(expected[:cfg['geometry_budget']]) & positive; prefix_matches += 1
                row = lookup[0, seed, name, 4]; found = set(row['positive_mode_keys']); assert found <= positive
                state = cache[seed, name, 4]; actual = model.old.posterior.make_predict(state[1])(grid)
                ce = float(np.mean([np.trapezoid((actual - full[i]) * (actual - full[j]), x=grid) for i, j in [(0, 1), (2, 3)]]))
                mask = np.array([key in found for key in curve['patterns']]); mass = float(weights[mask].sum()); trunc = None
                if mass:
                    ideal = np.einsum('k,rkq->rq', weights * mask / mass, means)
                    trunc = float(np.mean([np.trapezoid((ideal[i] - full[i]) * (ideal[j] - full[j]), x=grid) for i, j in [(0, 1), (2, 3)]]))
                conditional.append(dict(seed=seed, method=name, conditional_excess=ce, ideal_truncation=trunc, mass=mass))
        print(json.dumps(dict(audited_seed=seed, unique_states=len(cache))), flush=True)
    summary = []; risk = {}; time_arrays = {}; costs = {}
    for name in cfgs:
        risk[name] = np.array([np.mean([lookup[0, s, name, n]['query_mse'] for n in stages]) for s in seeds])
        tt = np.array([[[lookup[rep, s, name, n]['adaptation_seconds'] + lookup[rep, s, name, n]['read_queries_seconds'] for n in stages] for s in seeds] for rep in range(protocol['repetitions'])]); time_arrays[name] = tt; costs[name] = tt.sum(2).mean(0)
        first = [lookup[rep, s, name, 4] for rep in range(protocol['repetitions']) for s in seeds]; cr = [r for r in conditional if r['method'] == name]
        geometry = [r.get('geometry_calls', r.get('initial_geometry_calls', 0) + r.get('completion_geometry_calls', 0)) for r in first]
        summary.append(dict(method=name, mean_query_mse=float(risk[name].mean()), stage_query_mse=[float(np.mean([lookup[0, s, name, n]['query_mse'] for s in seeds])) for n in stages],
            mean_full_seconds=float(costs[name].mean()), median_task_mean_full_seconds=float(np.median(costs[name])), mean_first_seconds=float(tt[:, :, 0].mean()),
            mean_first_geometry_calls=float(np.mean(geometry)), mean_conditional_excess=float(np.mean([r['conditional_excess'] for r in cr])),
            mean_ideal_truncation=None if any(r['ideal_truncation'] is None for r in cr) else float(np.mean([r['ideal_truncation'] for r in cr])), mean_mass=float(np.mean([r['mass'] for r in cr])),
            feasible_streams=sum(all(lookup[0, s, name, n]['support_feasible'] for n in stages) for s in seeds),
            maximum_old_plus_new_state_bytes=max(r['previous_state_bytes'] + r['persistent_state_bytes'] for r in rows if r['method'] == name),
            global_bp_credit_first_write=any(r['global_bp_credit_used'] for r in first)))
    candidate = protocol['primary_candidate']; pairs = [(candidate, name) for name in cfgs if name != candidate] + [tuple(protocol['primary_increment_pair'])]; paired = []
    for left, right in pairs:
        delta = risk[left] - risk[right]; time_delta = costs[left] - costs[right]
        first_delta = (time_arrays[left][:, :, 0] - time_arrays[right][:, :, 0]).mean(0)
        paired.append(dict(candidate=left, comparator=right, primary=(left == candidate and right in protocol['primary_comparators']) or [left, right] == protocol['primary_increment_pair'],
            mse_delta=float(delta.mean()), descriptive_mse_ci95=ci(delta), better_tasks=int((delta < -1e-15).sum()), equal_tasks=int((np.abs(delta) <= 1e-15).sum()),
            mean_full_time_delta=float(time_delta.mean()), descriptive_time_ci95=ci(time_delta), mean_first_time_delta=float(first_delta.mean()), descriptive_first_time_ci95=ci(first_delta)))
    result = dict(audit=dict(source_hashes=len(protocol['source_sha256']), state_hashes=len(rows), unique_states_query_replayed=len(cache), full_reference_states_bitwise=full_matches,
        diagnostic_first_prefix_matches=prefix_matches, repeat_states_bitwise=True, later_stages_direct_only=True), summary=summary, paired=paired,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), scope='old 16 tasks; quality repetitions are identical, not independent; conditional risk is numerical reference-based')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8'); (out / 'conditional_risks.json').write_text(json.dumps(conditional, indent=2), encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(15, 7.4), sharey=True); pos = np.arange(len(summary)); colors = ['#167e79' if r['method'].startswith('alm') else '#d79434' if r['method'].startswith('adam') else '#778995' for r in summary]
    for ax, field, label in zip(axes, ['mean_query_mse', 'mean_full_seconds', 'mean_conditional_excess'], ['Actual unseen-query MSE, full stream', 'Mean full stream fit + read (s)', 'First-write conditional excess risk']):
        values = np.array([r[field] for r in summary]); ax.barh(pos, values, color=colors); ax.set_xlim(0, float(values.max()) * 1.24); ax.grid(axis='x', alpha=.2); ax.set_xlabel(label)
        for y, value in zip(pos, values): ax.text(value + values.max() * .015, y, f'{value:.4f}' if field != 'mean_full_seconds' else f'{value:.3f}', va='center', fontsize=7)
    axes[0].set_yticks(pos, [r['method'] for r in summary], fontsize=8); axes[0].invert_yaxis(); fig.suptitle('Finite first-write geometry budget: actual causal states and all online costs'); fig.tight_layout(); fig.savefig(out / 'budgeted_credit_results.png', dpi=170); plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
