"""337 independent resource selection, saved arrays, nested pools and memory checks."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import traceback
import numpy as np

BASE = 'results/budget_reinvestment'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def close(a, b): assert abs(a-b) < 2e-12, (a, b)


def arrays_equal(a, b):
    assert set(a) == set(b)
    for key in a:
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        assert a[key].tobytes() == b[key].tobytes(), key
    return len(a)


def arrays(p):
    with np.load(p, allow_pickle=False) as z: return {n: z[n] for n in z.files}


def run(root, out):
    folder = root/BASE/'calibration_v1'; summary = read(folder/'summary.json'); assert summary['passed']
    for name, digest in summary['outputs_sha256'].items(): assert sha(folder/name) == digest
    p = read(folder/'protocol.json'); source_checks = 0
    for name, digest in p['source_sha256'].items(): assert sha(root/name) == digest; source_checks += 1
    assert p['seeds'] == list(range(328000000, 328000008)) and p['repeats'] == 2 and not p['query_targets_accessed']
    cfgs = p['configs']; assert len(cfgs) == len({c['name'] for c in cfgs}) == 88
    rows, warmup, memory = [read(folder/n) for n in ['timings.json', 'warmup.json', 'memory.json']]
    assert len(rows) == 1408 and len(warmup) == 88
    assert len({(r['seed'], r['method'], r['repeat']) for r in rows}) == 1408
    jobs = [(seed, c['name'], rep) for seed in p['seeds'] for c in cfgs for rep in range(2)]
    for order, j in enumerate(np.random.default_rng(337929).permutation(1408)):
        r = rows[order]; assert (r['seed'], r['method'], r['repeat']) == jobs[int(j)] and r['order'] == order
        assert r['seconds'] > 0 and math.isfinite(r['seconds']) and r['metadata']['charged_complete_seconds'] == r['seconds']
        assert not r['metadata']['query_targets_accessed']
        assert read(folder/'calls'/f'timing_{order:04d}.json') == r
    for order, r in enumerate(warmup): assert read(folder/'calls'/f'warmup_{order:03d}.json') == r
    files = read(folder/'files.json'); assert len(files) == 704
    for name, digest in files.items(): assert sha(folder/name) == digest
    for r in rows+warmup: assert files[r['file']] == r['sha256']
    means = {c['name']: statistics.fmean(r['seconds'] for r in rows if r['method'] == c['name']) for c in cfgs}
    failed = {c['name']: sum(r['metadata']['execution_failed'] for r in rows if r['method'] == c['name']) for c in cfgs}
    cap = means['reference_fraction_first_fit_dual']; credits = sorted({c['base_config']['name'] for c in cfgs if c['family'] == 'budgeted_online_credit'})
    ks = [8, 16, 32, 64]
    eligible = {n: [k for k in ks if means[n+f'__k{k}'] <= cap and failed[n+f'__k{k}'] == 0] for n in credits}
    common = [k for k in ks if all(k in eligible[n] for n in credits)]; assert common
    k = max(common); own = {n: max(values) for n, values in eligible.items()}
    chosen = read(folder/'selection.json'); close(chosen['budget_seconds'], cap)
    assert chosen['common_k'] == k and chosen['admissible_common_k'] == common
    assert chosen['per_credit_largest_k'] == own and chosen['per_credit_admissible_k'] == eligible
    assert not chosen['query_quality_used'] and chosen['all_methods_retained']
    expected_memory = {'reference_fraction_first_fit_dual'} | {c['name'] for c in cfgs if c['family'] not in ['budgeted_online_credit', 'fraction_reference']}
    expected_memory |= {n+f'__k{k}' for n in credits} | {n+f'__k{own[n]}' for n in credits}
    assert chosen['memory_methods'] == sorted(expected_memory)
    assert chosen['within_reference_budget'] == [c['name'] for c in cfgs if means[c['name']] <= cap]
    assert chosen['sensitivity_110_percent'] == [c['name'] for c in cfgs if means[c['name']] <= 1.1*cap]
    assert chosen['highest_adam_still_within'] == (means['probe_then_adam3840_33'] <= cap)
    assert chosen['highest_adam_within_110_percent'] == (means['probe_then_adam3840_33'] <= 1.1*cap)
    bitwise = memory_arrays = nested = support_particles = 0
    assert len(memory) == len(expected_memory)*2
    assert {(r['seed'], r['method']) for r in memory} == {(s, n) for s in [p['seeds'][0], p['seeds'][-1]] for n in expected_memory}
    for r in memory:
        record_path = folder/r['record_file']; assert sha(record_path) == r['record_sha256']
        record = read(record_path); assert record == {k: v for k, v in r.items() if k not in ['record_file', 'record_sha256']}
        assert r['source_sha256'] == p['source_sha256'] and r['checkpoint_manifest'] == p['checkpoint_manifest']
        assert not r['query_targets_accessed'] and r['traced_peak_bytes'] >= r['traced_current_bytes'] >= 0
        ap = record_path.parent/'arrays.npz'; assert sha(ap) == r['arrays_sha256']
        memory_arrays += arrays_equal(arrays(ap), arrays(folder/f'{r["seed"]}_{r["method"]}.npz'))
    lookup = {(r['seed'], r['method']): r for r in rows if r['repeat'] == 0}
    for seed in p['seeds']:
        manifest = p['observed_inputs'][str(seed)]; path = root/manifest['file']; assert sha(path) == manifest['sha256']
        with np.load(path, allow_pickle=False) as z: x, v = z['x_observed'], z['v_observed']
        for name in credits:
            previous = None
            for width in ks:
                r = lookup[seed, name+f'__k{width}']; m = r['metadata']; a = arrays(folder/r['file'])
                if m['execution_failed']:
                    previous = None; continue  # Failure retained; never count as a nested success.
                assert m['no_global_bp_guard_enabled'] == (not name.endswith('_bp'))
                assert m['reinvestment_k'] == width and not m['uses_complete_posterior_reference']
                if m['positive_modes']:
                    h = np.broadcast_to(x, (len(a['points']), len(x)))
                    for j in range(4): h = np.maximum(0., 1.-abs(2*(h+a['points'][:, j, None])-1.))
                    assert float(np.max(abs(h-v))) <= .001+1e-7
                    support_particles += len(a['points'])
                if previous is not None:
                    b, old = previous
                    assert set(old['positive_modes']) <= set(m['positive_modes'])
                    assert old['original_positive_modes'] == m['original_positive_modes'] and old['selected_state'] == m['selected_state']
                    for field in ['current_lower', 'current_structurally_infeasible', 'current_certified_infeasible',
                                  'minimum_nonexcluded_hamming', 'necessary_hamming_lower_bound', 'shell_minima']:
                        assert old['proposal'][field] == m['proposal'][field]
                    for proposal in old['proposal']['proposals']: assert proposal in m['proposal']['proposals']
                    keys = set(a)-{'points', 'allocation', 'prediction'}
                    bitwise += arrays_equal({n: a[n] for n in keys}, {n: b[n] for n in keys})
                    if old['positive_modes'] == m['positive_modes']:
                        bitwise += arrays_equal({n: a[n] for n in ['points', 'allocation', 'prediction']}, {n: b[n] for n in ['points', 'allocation', 'prediction']})
                    nested += 1
                previous = a, m
    tables = read(folder/'methods.json'); assert len(tables) == 88; aggregates = 0
    for m in tables:
        name = m['method']; close(m['mean_seconds'], means[name]); rr = [r['seconds'] for r in rows if r['method'] == name]
        close(m['median_seconds'], statistics.median(rr)); close(m['p90_seconds'], float(np.quantile(rr, .9))); close(m['maximum_seconds'], max(rr))
        assert m['failures'] == failed[name]; mm = [r for r in memory if r['method'] == name]
        assert m['memory_probe_count'] == len(mm)
        assert m['maximum_traced_peak_bytes'] == max((r['traced_peak_bytes'] for r in mm), default=None)
        assert m['maximum_absolute_lifetime_peak_wset'] == max((r['process_after'].get('peak_wset', 0) for r in mm), default=None)
        assert m['maximum_dp_entries'] == max(((r['metadata'].get('proposal') or {}).get('meta', {}).get('max_retained_dp_entries', 0) for r in rows if r['method'] == name), default=0)
        aggregates += 9
    result = dict(passed=True, source_hash_checks=source_checks, timing_calls=1408, warmups=88, methods=88,
        aggregate_fields=aggregates, selected_common_k=k, memory_calls=len(memory), memory_array_checks=memory_arrays,
        nested_successful_pairs=nested, nested_state_arrays=bitwise, checked_support_particles=support_particles,
        numerical_failures=sum(failed.values()), query_targets_accessed=False, core_research_goal_complete=False,
        calibration_summary_sha256=sha(folder/'summary.json'), selection_sha256=sha(folder/'selection.json'),
        auditor_source_sha256=sha(Path(__file__)))
    save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'audit_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
