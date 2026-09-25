"""337 before calibration: exact prefix theorem and real online nested pools."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as suite
import branch_image_chain_v1 as reference
import branch_image_chain_dyadic_v1 as integer
import support_consistency_trigger_dyadic_v1 as support
from posterior_confirmation_pipeline import discovery_box


def exhaustive():
    rng = np.random.default_rng(337193); cases = patterns = comparisons = 0
    for n, d in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        for case in range(4):
            x = rng.uniform(0, 1, n); b = rng.uniform(-.12, .12, d)
            target, original = support.exact_forward(b, x)
            v = np.array(list(map(float, target))); credit = rng.normal(size=(d, n))
            if case == 0: credit[:] = 0
            problem = integer.IntegerProblem(x, v, credit); brute = {}
            for path in product(range(4), repeat=n*d):
                pattern = np.array(path).reshape(d, n)
                value = reference.fixed_value(x, v, credit, pattern)
                if value is not None:
                    distance = int(np.count_nonzero(pattern != original))
                    brute.setdefault(distance, []).append((value, path))
                patterns += 1
            previous = None
            for k in suite.KS:
                pools, _ = integer.integer_shells(problem, original, k)
                converted = {m: [(Fraction(value, problem.denominator), path) for value, path in pool] for m, pool in pools.items()}
                assert converted == {m: sorted(pool)[:k] for m, pool in brute.items()}
                if previous is not None:
                    for m, values in previous.items(): assert converted[m][:len(values)] == values
                previous = converted; comparisons += 1
            cases += 1
    return dict(exhaustive_cases=cases, enumerated_patterns=patterns, large_k_exact_comparisons=comparisons)


def check_nested(smaller, larger):
    a, m = smaller; b, n = larger
    for field in ['current_lower', 'current_structurally_infeasible', 'current_certified_infeasible',
                  'minimum_nonexcluded_hamming', 'necessary_hamming_lower_bound', 'shell_minima']:
        assert m['proposal'][field] == n['proposal'][field], field
    for proposal in m['proposal']['proposals']:
        assert proposal in n['proposal']['proposals'], proposal
    assert set(m['positive_modes']) <= set(n['positive_modes'])
    assert m['original_positive_modes'] == n['original_positive_modes'] and m['selected_state'] == n['selected_state']
    invariant = set(a)-{'points', 'allocation', 'prediction'}
    count = suite.array_checks({k: a[k] for k in invariant}, {k: b[k] for k in invariant})
    if m['positive_modes'] == n['positive_modes']:
        count += suite.array_checks({k: a[k] for k in ['points', 'allocation', 'prediction']},
                                    {k: b[k] for k in ['points', 'allocation', 'prediction']})
    return count


def run(root, out):
    begin = time.perf_counter(); hashes = suite.gate(root); seed = suite.SEEDS[0]
    inputs, manifest, index = suite.observed(root, [seed]); x, v, q = inputs[seed]
    configs = [c for c in suite.catalogue(root) if c['family'] == 'budgeted_online_credit']
    suite.save(out/'protocol.json', dict(source_sha256=hashes, observed_inputs=manifest, seed=seed,
        configs=configs, query_targets_accessed=False, prediction_quality_used=False))
    tests = exhaustive(); suite.save(out/'exhaustive.json', tests)
    checks = pairs = particles = 0; data = {}; records = []; files = {}
    with discovery_box(.12):
        for cfg in configs:
            a, m, seconds = suite.invoke(cfg, x, v, q, seed, {})
            assert not m['execution_failed']
            checks += suite.frozen_check(root, cfg, seed, a, m, index)
            points = a['points']
            if m['positive_modes']:
                h = np.broadcast_to(x, (len(points), len(x)))
                for layer in range(4): h = np.maximum(0., 1.-abs(2*(h+points[:, layer, None])-1.))
                assert float(np.max(abs(h-v))) <= .001+1e-7
                particles += len(points)
            name = cfg['name']; file = name+'.npz'; meta = name+'.json'
            with (out/file).open('xb') as f: np.savez_compressed(f, **a)
            suite.save(out/meta, m); files[file] = suite.sha(out/file); files[meta] = suite.sha(out/meta)
            data[name] = a, m
            records.append(dict(method=name, k=cfg['k'], seconds_not_benchmark=seconds, positive_modes=len(m['positive_modes']),
                proposals=len(m['proposal']['proposals']), file=file, metadata_file=meta))
            if len(records) % 8 == 0: print(dict(preflight_calls=len(records), total=48, seconds=time.perf_counter()-begin), flush=True)
    for name in sorted({c['base_config']['name'] for c in configs}):
        for k1, k2 in zip(suite.KS[:-1], suite.KS[1:]):
            checks += check_nested(data[name+f'__k{k1}'], data[name+f'__k{k2}']); pairs += 1
    assert suite.gate(root) == hashes
    suite.save(out/'calls.json', records); suite.save(out/'files.json', files)
    result = dict(passed=True, **tests, real_calls=48, exact_nested_pairs=pairs,
        checked_returned_arrays=checks, checked_support_particles=particles, seconds=time.perf_counter()-begin,
        query_targets_accessed=False, outputs_sha256={n: suite.sha(out/n) for n in ['protocol.json', 'exhaustive.json', 'calls.json', 'files.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/suite.BASE/'preflight_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
