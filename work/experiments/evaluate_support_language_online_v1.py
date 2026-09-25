"""345 four risks and every prelisted descriptive paired comparison after seal."""
import math
from pathlib import Path
import time
import traceback
import numpy as np
import support_language_online_suite_v1 as suite
import run_support_language_online_v1 as pipeline
from analyze_recovered_online_comparison import forward
from evaluate_search_radius_development_v1 import contrast, METRICS


def run(root, out):
    begin = time.perf_counter(); hashes = suite.gate(root)
    pred = root/suite.BASE/'development_predictions_v1'; summary = suite.complete(pred)
    p = suite.read(pred/'protocol.json'); assert p['source_sha256'] == hashes
    assert p['seeds'] == suite.SEEDS and pipeline.verify(root, pred, p) == summary['checks']
    seal = suite.read(pred/'before_query_manifest.json')
    assert not seal['query_targets_accessed'] and seal['rows_sha256'] == suite.sha(pred/'rows.json')
    assert seal['protocol_sha256'] == suite.sha(pred/'protocol.json')
    names, seeds = p['methods'], p['seeds']; candidates = [c['name'] for c in p['configs']]
    lookup = {(r['seed'], r['method']): r for r in suite.read(pred/'rows.json')}
    suite.save(out/'protocol.json', dict(source_sha256=hashes, methods=names, seeds=seeds, candidates=candidates,
        primary=suite.PRIMARY, metrics=METRICS, new_blind_tasks=False, query_targets_accessed=True,
        prediction_summary_sha256=suite.sha(pred/'summary.json'), before_query_manifest_sha256=suite.sha(pred/'before_query_manifest.json'),
        inference='Descriptive exposed development; not significance or matched-resource superiority'))
    risk = np.empty((len(seeds), len(names), 4)); groups = []
    for i, seed in enumerate(seeds):
        q = np.linspace(0, 1, 257); b = np.random.default_rng(seed).uniform(-.12, .12, 4); truth = forward(q, b[None])[0]
        m = suite.read(root/lookup[seed, suite.PRIMARY]['metadata_file'])['metadata']
        groups.append('execution_failed' if m['execution_failed'] else 'support_fit' if m['selected_state']['support_fit'] else 'fallback')
        for j, name in enumerate(names):
            a = pipeline.load(root/lookup[seed, name]['file'])
            risk[i, j] = [float(np.mean((a[field][::stride]-truth[::stride])**2))
                          for field in ['prediction', 'point_prediction'] for stride in [1, 2]]
    assert np.isfinite(risk).all() and np.all((risk >= 0) & (risk <= 1))
    with (out/'task_metrics.npz').open('xb') as f:
        np.savez_compressed(f, seeds=seeds, methods=names, metric_names=METRICS, risk=risk, groups=groups)
    methods = []
    for j, name in enumerate(names):
        rr = [lookup[s, name] for s in seeds]; current = name in candidates
        methods.append(dict(method=name, metrics={metric: math.fsum(map(float, risk[:, j, k]))/len(seeds) for k, metric in enumerate(METRICS)},
            failures=sum(r['execution_failed'] for r in rr), current_development_time_available=current,
            mean_current_seconds=math.fsum(r['seconds'] for r in rr)/len(seeds) if current else None,
            group_metrics={g: {metric: math.fsum(float(risk[i, j, k]) for i, label in enumerate(groups) if label == g)/groups.count(g)
                               for k, metric in enumerate(METRICS)} for g in sorted(set(groups))}))
    comparisons = [dict(candidate=c, control=b, metric=metric, descriptive_only=True,
                        **contrast(risk[:, names.index(c), k]-risk[:, names.index(b), k], seeds))
                   for c in candidates for b in names if c != b for k, metric in enumerate(METRICS)]
    suite.save(out/'methods.json', methods); suite.save(out/'comparisons.json', comparisons)
    suite.save(out/'groups.json', dict(seeds=seeds, groups=groups, counts={g: groups.count(g) for g in set(groups)}))
    assert suite.gate(root) == hashes
    result = dict(passed=True, tasks=len(seeds), methods=len(names), scored_predictors=len(seeds)*len(names),
        comparisons=len(comparisons), seconds=time.perf_counter()-begin, query_targets_accessed=True,
        new_blind_tasks=False, statistical_superiority_claim=False, matched_resource_superiority_claim=False,
        core_research_goal_complete=False,
        outputs_sha256={n: suite.sha(out/n) for n in ['protocol.json', 'task_metrics.npz', 'methods.json', 'comparisons.json', 'groups.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/suite.BASE/'development_evaluation_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
