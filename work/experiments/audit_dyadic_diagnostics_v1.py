"""Independent scalar reaggregation of 333/334/335; no algorithm invocation."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback

BASE = 'results/online_credit_fresh_pilot'


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def close(a, b):
    assert abs(a-b) < 2e-12, (a, b)


def sealed(folder):
    summary = read(folder/'summary.json'); assert summary['passed']
    for name, digest in summary['outputs_sha256'].items(): assert sha(folder/name) == digest
    return summary


def run(root, out):
    base = root/BASE
    diag = base/'cost_readout_diagnostic_v1'; kernel = base/'dyadic_branch_kernel_v1'; online = base/'dyadic_online_v1'
    ds, ks, os = sealed(diag), sealed(kernel), sealed(online)
    source_checks = 0
    for folder in [diag, kernel, online]:
        p = read(folder/'protocol.json')
        for path, digest in p['source_sha256'].items():
            assert sha(root/path) == digest; source_checks += 1
        for name, digest in p.get('frozen_source_sha256', {}).items():
            assert sha(root/'work/experiments'/name) == digest; source_checks += 1
    assert ds['predictors'] == 8192 and ks['saved_state_equivalences'] == 6144 and os['calls'] == 576
    records = read(diag/'tasks.json'); reported = read(diag/'methods.json')
    published = {r['method']: r for r in read(base/'pilot_evaluation_v2/methods.json')}
    cells = 0
    for m in reported:
        rr = [r for r in records if r['method'] == m['method']]
        assert len(rr) == 512 and len({r['seed'] for r in rr}) == 512
        assert m['single_point_fallbacks'] == sum(r['particles'] == 1 for r in rr)
        assert m['execution_failures'] == sum(r['execution_failed'] for r in rr)
        for r in rr:
            close(r['charged_seconds'], math.fsum(r['components'].values())+r['unallocated_seconds'])
            close(r['actual_mse']-r['estimated_readout_variance'], r['corrected_fixed_teacher_risk_estimate'])
            assert r['estimated_readout_variance'] >= 0
        for key, value in m.items():
            if key in ['method', 'tasks', 'single_point_fallbacks', 'execution_failures', 'components'] or value is None: continue
            close(value, statistics.fmean(r[key] for r in rr)); cells += 1
        for key, value in m['components'].items():
            close(value, statistics.fmean(r['components'][key] for r in rr)); cells += 1
        close(m['actual_mse'], published[m['method']]['metrics']['mse257'])
        close(m['charged_seconds'], published[m['method']]['mean_seconds'])
        cells += 2
    eq = read(kernel/'equivalence.json')
    assert len(eq) == len({(r['seed'], r['method']) for r in eq}) == 6144 and all(r['equivalent'] for r in eq)
    assert set(r['seed'] for r in eq) == set(range(328000000, 328000512))
    kt = read(kernel/'timings.json'); assert len(kt) == 192
    for m in ks['timing_methods']:
        for impl in ['fraction', 'integer']:
            rr = [r for r in kt if r['method'] == m['method'] and r['implementation'] == impl]
            assert len(rr) == 8 and len({(r['seed'], r['repetition']) for r in rr}) == 8
            close(m[impl], statistics.fmean(r['seconds'] for r in rr)); cells += 1
        close(m['fraction_over_integer'], m['fraction']/m['integer']); cells += 1
    ot = read(online/'calls.json'); om = read(online/'methods.json')
    assert len(ot) == len({(r['seed'], r['method'], r['implementation'], r['repetition']) for r in ot}) == 576
    assert set(r['seed'] for r in ot) == set(range(328000000, 328000008))
    assert sum(r['bitwise_arrays'] for r in ot) == os['bitwise_array_checks']
    for r in ot:
        saved = online/str(r['seed'])/f"{r['method']}_{r['implementation']}_{r['repetition']}.json"
        assert read(saved) == r
        assert r['no_global_bp_guard'] == (not r['method'].endswith('_bp'))
    for m in om:
        task_values = {}
        for impl in ['fraction', 'integer_dp', 'integer_dp_trigger']:
            task_values[impl] = []
            for seed in range(328000000, 328000008):
                rr = [r for r in ot if r['seed'] == seed and r['method'] == m['method'] and r['implementation'] == impl]
                assert len(rr) == 2
                task_values[impl].append(statistics.fmean(r['seconds'] for r in rr))
            close(m['mean_seconds'][impl], statistics.fmean(task_values[impl])); cells += 1
        close(m['fractional_reduction'], 1-m['mean_seconds']['integer_dp_trigger']/m['mean_seconds']['fraction'])
        assert m['faster_tasks'] == sum(a < b for a, b in zip(task_values['integer_dp_trigger'], task_values['fraction']))
        cells += 2
    result = dict(passed=True, source_hash_checks=source_checks, scalar_aggregate_checks=cells,
        diagnostic_rows=8192, kernel_equivalences=6144, kernel_timing_calls=192, online_timing_calls=576,
        online_bitwise_array_checks=os['bitwise_array_checks'], query_targets_accessed=False,
        role='Independent reaggregation; does not replace 334 exhaustive rational check or 335 real fit replay',
        input_summary_sha256={p.name: sha(p/'summary.json') for p in [diag, kernel, online]},
        auditor_source_sha256=sha(Path(__file__)), core_research_goal_complete=False)
    write(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'dyadic_diagnostics_audit_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
