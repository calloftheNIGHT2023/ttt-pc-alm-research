"""387 query targets created only after complete prediction seal and input audit."""
from collections import defaultdict
from pathlib import Path
import math
import os
import time
import traceback
import numpy as np
import deadline_risk_io_v1 as io
import deadline_risk_registry_v1 as registry


def evaluate(root, out):
    begin = time.perf_counter(); source = root/io.BASE/'development_v1'; audit_dir = root/io.BASE/'audit_v1'
    summary = io.read(source/'summary.json'); audit = io.read(audit_dir/'summary.json')
    assert summary['passed'] and summary['all_predictions_sealed'] and audit['passed']
    assert audit['development_summary_sha256'] == io.sha(source/'summary.json')
    assert not (source/'failure.json').exists() and not (audit_dir/'failure.json').exists()
    for p, h in summary['outputs_sha256'].items(): assert io.sha(source/p) == h
    for p, h in audit['outputs_sha256'].items(): assert io.sha(audit_dir/p) == h
    protocol = io.read(source/'protocol.json'); io.verify_hashes(root, protocol['source_sha256']); io.verify_hashes(root, protocol['pretrained_sha256'])
    rows = io.read(source/'rows.json'); q = np.linspace(0., 1., 257)
    io.save(out/'before_query.json', dict(prediction_summary_sha256=io.sha(source/'summary.json'),
        audit_summary_sha256=io.sha(audit_dir/'summary.json'), protocol_sha256=io.sha(source/'protocol.json'),
        evaluator_sha256=io.sha(Path(__file__)), complete_prediction_count=len(rows)))
    # First and only query-target creation in this workflow occurs after this seal.
    import streaming_branch_projection as base
    truth = {seed:base.forward(q, np.random.default_rng(seed).uniform(-.12, .12, 4)) for seed in registry.SEEDS}
    np.savez_compressed(out/'query_truth.npz', q=q, seeds=np.array(registry.SEEDS), targets=np.stack([truth[s] for s in registry.SEEDS]))
    risk_rows = []; groups = defaultdict(list)
    for row in rows:
        directory = root/row['directory']; assert io.sha(directory/'outputs.npz') == row['files']['outputs.npz']
        prediction = io.load_arrays(directory/'outputs.npz')['prediction']
        assert prediction.shape == q.shape and np.all((prediction >= 0.) & (prediction <= 1.))
        mse = float(np.mean((prediction-truth[row['seed']])**2))
        record = dict(row, query_mse=mse); risk_rows.append(record); groups[row['method'], row['budget']].append(record)
    assert len(risk_rows) == 2304 and len(groups) == 72
    sessions = io.read(source/'sessions.json'); extra_cleanup = defaultdict(float)
    for session in sessions:
        extra_cleanup[session['method']] += math.fsum(s['seconds'] for s in session['closures'] if not s['force'])
    summaries = []
    for cfg in registry.configs():
        for budget in registry.BUDGETS:
            rr = groups[cfg['name'], budget]; assert sorted(r['seed'] for r in rr) == registry.SEEDS
            summaries.append(dict(method=cfg['name'], kind=cfg['kind'], budget=budget, independent_tasks=32,
                mean_query_mse=math.fsum(r['query_mse'] for r in rr)/32,
                final=sum(r['selected'] == 'final' for r in rr), fallback=sum(r['selected'] == 'fallback' for r in rr),
                constant=sum(r['selected'] == 'constant' for r in rr),
                mean_online_seconds=math.fsum(r['online_seconds'] for r in rr)/32,
                mean_setup_seconds=math.fsum(r['setup_seconds'] for r in rr)/32,
                mean_cleanup_seconds=math.fsum(r['cleanup_seconds']+r['task_cleanup_seconds'] for r in rr)/32,
                mean_archive_seconds=math.fsum(r['archive_seconds'] for r in rr)/32,
                mean_controller_overrun_seconds=math.fsum(r['controller_overrun_seconds'] for r in rr)/32,
                max_controller_overrun_seconds=max(r['controller_overrun_seconds'] for r in rr),
                mean_worker_sampled_rss=math.fsum(r['worker_max_sampled_rss'] for r in rr)/32,
                max_worker_sampled_rss=max(r['worker_max_sampled_rss'] for r in rr),
                method_total_nonforced_session_close_seconds=extra_cleanup[cfg['name']],
                session_close_cost_scope='Once per method across both budgets; do not double count this repeated reference'))
    lookup = {(r['method'], r['budget'], r['seed']):r for r in risk_rows}
    comparisons = []
    controls = [c['name'] for c in registry.configs() if c['name'] != registry.PRIMARY]
    for bi, budget in enumerate(registry.BUDGETS):
        for ci, control in enumerate(controls):
            primary = np.array([lookup[registry.PRIMARY, budget, s]['query_mse'] for s in registry.SEEDS])
            other = np.array([lookup[control, budget, s]['query_mse'] for s in registry.SEEDS]); delta = primary-other
            rng = np.random.default_rng(np.random.SeedSequence([387199, bi, ci]))
            boot = delta[rng.integers(0, 32, (100000, 32))].mean(1)
            comparisons.append(dict(primary=registry.PRIMARY, control=control, budget=budget,
                primary_budget=budget == registry.PRIMARY_BUDGET, predeclared_main_control=control in registry.MAIN_CONTROLS,
                mean_primary=float(primary.mean()), mean_control=float(other.mean()), mean_paired_difference=float(delta.mean()),
                bootstrap95=np.quantile(boot, [.025, .975]).tolist(),
                bonferroni_one_sided_upper=float(np.quantile(boot, 1.-.05/35)),
                primary_better_tasks=int((delta < 0).sum()), exact_equal_tasks=int((delta == 0).sum()),
                approximate_bootstrap_not_finite_sample_guarantee=True, independent_tasks=32))
    main = [c for c in comparisons if c['primary_budget'] and c['predeclared_main_control']]
    result = dict(passed=True, calls=2304, independent_tasks=32, groups=72,
        primary_budget_lower_mean_all_eight_main_controls=all(c['mean_paired_difference'] < 0 for c in main),
        primary_budget_adjusted_upper_below_zero_all_eight=all(c['bonferroni_one_sided_upper'] < 0 for c in main),
        new_development_not_independent_confirmation=True, official_matched_ttt_included=False,
        core_research_goal_complete=False, seconds=time.perf_counter()-begin)
    io.save(out/'risk_rows.json', risk_rows); io.save(out/'groups.json', summaries); io.save(out/'comparisons.json', comparisons)
    result['outputs_sha256'] = {p:io.sha(out/p) for p in ['before_query.json', 'query_truth.npz', 'risk_rows.json', 'groups.json', 'comparisons.json']}
    io.save(out/'summary.json', result); print(dict(stage='deadline_risk_evaluation_complete', **result), flush=True)


if __name__ == '__main__':
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    root = Path(__file__).resolve().parents[2]; out = root/io.BASE/'evaluation_v1'; out.mkdir(parents=True, exist_ok=False)
    try: evaluate(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
