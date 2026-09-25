"""305 independent arithmetic, journals and frozen trace/no-trace array audit."""
import argparse
from collections import Counter
import math
from pathlib import Path
import statistics
import time

import numpy as np
from diagnose_gradient_flat_split_states_v1 import read, save, sha

PRIMARY = 'online_stasis_alm_keep64'
SEEDS = [5910000,5910001,5910008,5910016,5910032,5910048,5910053,5910063]


def close(a, b):
    assert math.isclose(a, b, rel_tol=2e-13, abs_tol=1e-14), (a, b)


def percentile(values, p):
    x = sorted(values)
    position = (len(x)-1)*p
    k = math.floor(position)
    return x[k]+(position-k)*(x[min(k+1, len(x)-1)]-x[k])


def run(root, out):
    start = time.perf_counter()
    folder = root/'results/online_stasis_resources/calibration_v1'
    summary = read(folder/'summary.json')
    assert summary['passed'] and summary['timing_runs'] == 912 and summary['methods'] == 38
    assert not summary['query_targets_accessed'] and not summary['new_task_quality_test']
    for n, d in summary['outputs_sha256'].items():
        assert sha(folder/n) == d, n
    protocol = read(folder/'protocol.json')
    assert protocol['seeds'] == SEEDS and protocol['repeats'] == 3 and protocol['order_seed'] == 305929
    assert protocol['primary'] == PRIMARY and protocol['trace'] is False
    assert protocol['threads'] == dict(blas=1, omp=1, torch=1)
    assert sha(root/'outputs/ttt-pc-alm-research/305_online_stasis_resource_protocol.md') == protocol['design_sha256']
    for n, d in protocol['source_sha256'].items():
        assert sha(root/'work/experiments'/n) == d, n
    old = root/'results/counterfactual_resources/calibration_v1'
    functional = root/'results/online_stasis_memory/functional_v2'
    assert sha(old/'summary.json') == protocol['original_summary_sha256']
    assert sha(functional/'summary.json') == protocol['functional_summary_sha256']
    assert protocol['configs'][:-1] == read(old/'protocol.json')['configs']
    cfgs = protocol['configs']; names = [c['name'] for c in cfgs]
    timings = read(folder/'timings.json'); warmup = read(folder/'warmup.json')
    files = read(folder/'files.json'); checks = read(folder/'checks.json')
    assert len(names) == len(set(names)) == 38 and len(files) == 304
    assert len(timings) == 912 and len(warmup) == 38 and len(checks) == 950
    expected = [(s, n, r) for s in SEEDS for n in names for r in range(3)]
    order = np.random.default_rng(305929).permutation(912)
    actual = [(r['seed'], r['method'], r['repeat']) for r in timings]
    assert actual == [expected[int(i)] for i in order]
    assert read(folder/'schedule.json') == [dict(seed=s, method=n, repeat=r) for s, n, r in actual]
    assert [(r['seed'], r['method']) for r in warmup] == [(5910001, n) for n in names]
    counts = Counter(); reference_rows = {}
    for r in read(old/'timings.json'):
        reference_rows.setdefault((r['seed'], r['method']), (old, r))
    for r in read(functional/'rows.json'):
        reference_rows[r['seed'], PRIMARY] = (functional, r)
    first = {}; seen = set(); evidence = []
    for row, check in zip(warmup+timings, checks):
        kind, idx = row['kind'], row['order']
        callname = f'{kind}_{idx:04d}.json'
        assert read(folder/'calls'/callname) == row
        assert read(folder/'attempts'/callname) == {k:v for k,v in row.items() if k not in ['file','sha256']}
        assert math.isfinite(row['seconds']) and row['seconds'] > 0
        assert row['metadata']['charged_complete_seconds'] == row['seconds']
        assert files[row['file']] == row['sha256']
        for k in ['kind','order','seed','method']:
            assert check[k] == row[k]
        ref_folder, ref = reference_rows[row['seed'], row['method']]
        assert str((ref_folder/ref['file']).relative_to(root)) == check['reference']
        assert check['reference_sha256'] == ref['sha256']
        assert check['execution_failed'] == row['metadata']['execution_failed']
        if 'positive_modes' in row['metadata'] and not check['execution_failed']:
            assert row['metadata']['positive_modes'] == ref['metadata']['positive_modes']
            counts['positive_pool_call_checks'] += 1
        if row['file'] not in first:
            first[row['file']] = (row, check)
        with np.load(folder/row['file'], allow_pickle=False) as z:
            assert check['duplicate_array_checks'] == (len(z.files) if row['file'] in seen else 0)
            if not check['execution_failed']:
                assert sorted(check['frozen_fields']) == sorted(z.files)
        seen.add(row['file'])
        counts[kind+'_journals'] += 1
    for name, digest in files.items():
        assert sha(folder/name) == digest
        row, check = first[name]
        ref_path = root/check['reference']
        assert sha(ref_path) == check['reference_sha256']
        with np.load(folder/name, allow_pickle=False) as a, np.load(ref_path, allow_pickle=False) as ref:
            for k in a.files:
                value = a[k]
                if np.issubdtype(value.dtype, np.number):
                    assert np.isfinite(value).all(), (name,k)
                if k in ['prediction','point_prediction']:
                    assert value.shape == (257,) and value.dtype == np.float64
                    assert all(0 <= float(x) <= 1 for x in value)
                if not check['execution_failed']:
                    expected_value = ref[k]
                    if k in ['prediction','point_prediction']:
                        expected_value = np.asarray([min(1.,max(0.,float(x))) for x in expected_value], dtype=np.float64)
                    assert value.shape == expected_value.shape and value.dtype == expected_value.dtype
                    assert value.tobytes() == expected_value.tobytes(), (name,k)
                    counts['unique_frozen_arrays'] += 1
                    if row['method'] == PRIMARY:
                        counts['unique_online_trace_arrays'] += 1
            evidence.append(dict(file=name, frozen_fields=list(a.files), passed=True,
                                 reference=check['reference'], reference_sha256=check['reference_sha256']))
    assert len(first) == 304
    environments = read(folder/'environments.json')
    for i, r in enumerate(environments, 1):
        assert read(folder/'calls'/f'environment_{i:04d}.json') == r
        assert r['paused'] == bool(r['other_research_or_git_pack_processes'])
    assert sum(r['paused'] for r in environments) == summary['interference_snapshots']
    primary = {(r['seed'],r['repeat']):r['seconds'] for r in timings if r['method'] == PRIMARY}
    cap = statistics.fmean(primary.values())
    table = read(folder/'methods.json')
    assert [r['method'] for r in table] == names
    means = {}
    for cfg, reported in zip(cfgs, table):
        rr = sorted((r for r in timings if r['method']==cfg['name']), key=lambda r:(r['seed'],r['repeat']))
        assert len(rr) == 24
        seconds = [r['seconds'] for r in rr]
        means[cfg['name']] = statistics.fmean(seconds)
        close(reported['mean_seconds'], means[cfg['name']])
        close(reported['median_seconds'], statistics.median(seconds))
        close(reported['p90_seconds'], percentile(seconds,.9))
        close(reported['maximum_seconds'], max(seconds))
        close(reported['mean_over_candidate'], means[cfg['name']]/cap)
        close(reported['paired_mean_difference'], statistics.fmean(r['seconds']-primary[r['seed'],r['repeat']] for r in rr))
        close(reported['paired_mean_ratio'], statistics.fmean(r['seconds']/primary[r['seed'],r['repeat']] for r in rr))
        assert reported['failures'] == sum(r['metadata']['execution_failed'] for r in rr)
        assert len(reported['pairs']) == 24
        for pair, r in zip(reported['pairs'],rr):
            candidate = primary[r['seed'],r['repeat']]
            assert pair == dict(seed=r['seed'],repeat=r['repeat'],method_seconds=r['seconds'],
                candidate_seconds=candidate,method_minus_candidate=r['seconds']-candidate,method_over_candidate=r['seconds']/candidate)
        counts['method_tables'] += 1
        counts['paired_cost_rows'] += len(rr)
    selection = read(folder/'selection.json')
    close(selection['budget_seconds'],cap)
    assert selection['within_budget'] == [n for n in names if means[n] <= cap]
    assert selection['over_budget'] == [n for n in names if means[n] > cap]
    assert selection['sensitivity_110_percent'] == [n for n in names if means[n] <= 1.10*cap]
    assert selection['all_methods_retained'] and not selection['query_quality_used']
    close(summary['primary_seconds'],cap)
    assert summary['frozen_array_checks'] == sum(len(r['frozen_fields']) for r in checks)
    assert summary['repeated_array_checks'] == sum(r['duplicate_array_checks'] for r in checks)
    assert summary['failed_timing_calls'] == sum(r['metadata']['execution_failed'] for r in timings)
    save(out/'equivalence.json', evidence)
    save(out/'protocol.json', dict(source_sha256={Path(__file__).name:sha(Path(__file__))},
        calibration_summary_sha256=sha(folder/'summary.json'), independent_statistics='statistics.fmean/median and scalar interpolated quantiles',
        repeat_scope='Repeated execution identity was checked in the runner; audit verifies its journals, not separately saved repeated arrays',
        query_targets_accessed=False))
    result = dict(passed=True, methods=38, timing_runs=912, warmup_runs=38, files=304,
        counts=dict(counts), primary_seconds=cap, failed_timing_calls=summary['failed_timing_calls'],
        interference_snapshots=summary['interference_snapshots'], query_targets_accessed=False,
        core_research_goal_complete=False, seconds=time.perf_counter()-start,
        outputs_sha256={n:sha(out/n) for n in ['equivalence.json','protocol.json']})
    save(out/'summary.json',result)
    print(result,flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',required=True,type=Path)
    root=ap.parse_args().project.resolve()
    out=root/'results/online_stasis_resources/audit_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:
        run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise
