"""340 descriptive development scores only after every predictor is sealed."""
import math
from pathlib import Path
import time
import traceback
import numpy as np
import search_radius_development_suite_v1 as suite
import run_search_radius_development_v1 as pipeline
from analyze_recovered_online_comparison import forward
from counterfactual_fresh_pilot_v1 import scalar_truth

METRICS = ['mse257', 'mse129', 'point_mse257', 'point_mse129']


def contrast(values, seeds):
    values = np.asarray(values); n = len(values); total = math.fsum(map(float, values))
    i = int(np.argmax(abs(values))); absolute = math.fsum(map(float, abs(values)))
    return dict(mean_difference=total/n, improved=int(np.sum(values < 0)), equal=int(np.sum(values == 0)),
        worse=int(np.sum(values > 0)), largest_absolute_task_seed=int(seeds[i]),
        largest_absolute_task_delta=float(values[i]), largest_absolute_share=float(abs(values[i])/absolute) if absolute else 0.,
        leave_largest_absolute_out_mean=(total-float(values[i]))/(n-1),
        worst_leave_one_out_mean=max((total-float(v))/(n-1) for v in values))


def run(root, out):
    begin = time.perf_counter(); hashes = suite.gate(root)
    folder = root/suite.BASE/'development_predictions_v1'; summary = suite.complete(folder)
    p = suite.read(folder/'protocol.json'); assert p['source_sha256'] == hashes and p['seeds'] == suite.SEEDS
    checks = pipeline.verify(root, folder, p); assert checks == summary['checks']
    seal = suite.read(folder/'before_query_manifest.json')
    assert not seal['query_targets_accessed'] and seal['source_sha256'] == hashes
    assert seal['protocol_sha256'] == suite.sha(folder/'protocol.json') and seal['rows_sha256'] == suite.sha(folder/'rows.json')
    names = p['methods']; seeds = p['seeds']; rows = suite.read(folder/'rows.json')
    lookup = {(r['seed'], r['method']): r for r in rows}
    candidates = [f'radius_first_fit_{channel}__k{k}__sall'
                  for channel in ['dual', 'dual_plus_residual'] for k in [1, 2, 4, 8]]
    suite.save(out/'protocol.json', dict(source_sha256=hashes, seeds=seeds, methods=names, candidates=candidates,
        primary=suite.PRIMARY, metrics=METRICS, query_targets_accessed=True, new_blind_tasks=False,
        prediction_summary_sha256=suite.sha(folder/'summary.json'), before_query_manifest_sha256=suite.sha(folder/'before_query_manifest.json'),
        inference='Descriptive development; no significance claims or new-task confirmation',
        timings_scope='New calls only; old baseline runtimes are not compared to new calls'))
    risk = np.empty((len(seeds), len(names), 4)); truth_gap = scalar_gap = 0.
    groups = []
    for i, seed in enumerate(seeds):
        q = np.linspace(0, 1, 257); b = np.random.default_rng(seed).uniform(-.12, .12, 4)
        truth = forward(q, b[None])[0]; separate = scalar_truth(seed, q)
        truth_gap = max(truth_gap, float(np.max(abs(truth-separate)))); assert truth_gap < 2e-12
        primary = suite.read(root/lookup[seed, suite.PRIMARY]['metadata_file'])['metadata']
        groups.append('execution_failed' if primary['execution_failed'] else
                      'support_fit' if primary['selected_state']['support_fit'] else 'fallback')
        for j, name in enumerate(names):
            a = pipeline.load_arrays(root/lookup[seed, name]['file']); values = []; independent = []
            for field in ['prediction', 'point_prediction']:
                for stride in [1, 2]:
                    values.append(float(np.mean((a[field][::stride]-truth[::stride])**2)))
                    independent.append(math.fsum((float(x)-float(y))**2 for x, y in
                                                  zip(a[field][::stride], separate[::stride]))/len(a[field][::stride]))
            risk[i, j] = values; scalar_gap = max(scalar_gap, float(np.max(abs(np.array(values)-independent))))
            assert scalar_gap < 2e-12
    assert np.isfinite(risk).all() and np.all((risk >= 0) & (risk <= 1))
    with (out/'task_metrics.npz').open('xb') as f:
        np.savez_compressed(f, seeds=seeds, methods=names, metric_names=METRICS, risk=risk, groups=groups)
    methods = []
    for j, name in enumerate(names):
        rr = [lookup[s, name] for s in seeds]; current = all(r['origin'] == 'new_340' for r in rr)
        methods.append(dict(method=name, metrics={m: math.fsum(map(float, risk[:, j, k]))/len(seeds) for k, m in enumerate(METRICS)},
            failures=sum(r['execution_failed'] for r in rr), current_development_time_available=current,
            mean_current_seconds=math.fsum(r['seconds'] for r in rr)/len(seeds) if current else None,
            old_runtime_not_a_current_benchmark=not current,
            group_metrics={g: {m: math.fsum(float(risk[i, j, k]) for i, label in enumerate(groups) if label == g)/groups.count(g)
                               for k, m in enumerate(METRICS)} for g in sorted(set(groups))}))
    comparisons = []
    for candidate in candidates:
        for control in names:
            if control == candidate: continue
            for k, metric in enumerate(METRICS):
                values = risk[:, names.index(candidate), k]-risk[:, names.index(control), k]
                comparisons.append(dict(candidate=candidate, control=control, metric=metric, descriptive_only=True,
                                        **contrast(values, seeds)))
    suite.save(out/'methods.json', methods); suite.save(out/'comparisons.json', comparisons)
    suite.save(out/'groups.json', dict(seeds=seeds, groups=groups, counts={g: groups.count(g) for g in sorted(set(groups))},
                                     query_quality_used_for_grouping=False))
    assert suite.gate(root) == hashes
    result = dict(passed=True, tasks=len(seeds), methods=len(names), scored_predictors=len(seeds)*len(names),
        comparisons=len(comparisons), maximum_truth_gap=truth_gap, maximum_scalar_risk_gap=scalar_gap,
        query_targets_accessed=True, new_blind_tasks=False, statistical_superiority_claim=False,
        matched_resource_superiority_claim=False, core_research_goal_complete=False, seconds=time.perf_counter()-begin,
        outputs_sha256={n: suite.sha(out/n) for n in ['protocol.json', 'task_metrics.npz', 'methods.json', 'comparisons.json', 'groups.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/suite.BASE/'development_evaluation_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
