"""344 independent Fraction reachability plus exhaustive path/DP comparisons."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import branch_image_chain_v1 as reference
import branch_image_chain_dyadic_v1 as integer
import support_language_chain_v1 as language
from test_dyadic_branch_search_v1 import load_case, semantic

DESIGN = 'outputs/ttt-pc-alm-research/344_support_language_primitive_protocol_v1.md'


def possible(x, v, path):
    """Independent rational interval propagation; unbounded saturation branches."""
    low = high = F(float(x)); bound = F(.12)
    for r in path:
        a, b = low-bound, high+bound
        if r == 0:
            b = min(b, F(0))
        elif r == 1:
            a, b = max(a, F(0)), min(b, F(1, 2))
        elif r == 2:
            a, b = max(a, F(1, 2)), min(b, F(1))
        else:
            a = max(a, F(1))
        if a > b: return False
        if r in [0, 3]: low = high = F(0)
        elif r == 1: low, high = 2*a, 2*b
        else: low, high = 2-2*b, 2-2*a
    return max(low, F(float(v))-F(.001), F(0)) <= min(high, F(float(v))+F(.001), F(1))


def exhaustive():
    rng = np.random.default_rng(344929)
    counts = dict(cases=0, patterns=0, singleton_paths=0, dp_comparisons=0, feasible_forward_witnesses=0)
    for n, d in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)]:
        for case in range(6):
            x, v = rng.uniform(0, 1, (2, n)); a = rng.normal(size=(d, n)); original = rng.integers(0, 4, (d, n))
            if case == 1: a[:] = 0
            if case == 2: x[:] = .5; v[:] = 1
            if case == 3: a[:] = float.fromhex('0x0.0000000000001p-1022')
            if case == 4: v[:] = -.002
            if case == 5: x[:] = .12; v[:] = .001
            problem = integer.IntegerProblem(x, v, a)
            for i in range(n):
                wanted = [p for p in product(range(4), repeat=d) if possible(x[i], v[i], p)]
                assert language.accepted_paths(problem, i) == wanted
                counts['singleton_paths'] += 4**d
            brute = {}
            for flat in product(range(4), repeat=n*d):
                pattern = np.array(flat).reshape(d, n); counts['patterns'] += 1
                if not all(possible(x[i], v[i], pattern[:, i]) for i in range(n)): continue
                value = reference.fixed_value(x, v, a, pattern)
                if value is None: continue
                h = int(np.count_nonzero(pattern != original))
                brute.setdefault(h, []).append((value, flat))
            brute = {h: sorted(pool) for h, pool in brute.items()}
            for k in [1, 2, 8]:
                dp, _, _ = language.shells(problem, original, k)
                assert {h: [(F(val, problem.denominator), path) for val, path in pool] for h, pool in dp.items()} == {h: pool[:k] for h, pool in brute.items()}
                proposed = language.propose(x, v, a, original, k=k)
                expected = [dict(mode=bytes(path).hex(), hamming=h, rank=rank, lower=str(val))
                            for h, pool in sorted(brute.items()) if h >= 1
                            for rank, (val, path) in enumerate(pool[:k]) if val <= 0]
                assert proposed['proposals'] == expected
                counts['dp_comparisons'] += 1
            # Exact constructed forward witnesses, including branch boundary choices.
            biases = [F(float(b)) for b in rng.uniform(-.12, .12, d)]
            for xx in x:
                y = F(float(xx)); path = []
                for b in biases:
                    z = y+b; path.append(sum(z >= t for t in [F(0), F(1, 2), F(1)]))
                    y = max(F(0), 1-abs(2*z-1))
                assert possible(xx, float(y), path)
                counts['feasible_forward_witnesses'] += 1
            counts['cases'] += 1
    return counts


def run(root, out):
    start = time.perf_counter(); census = root/'results/radius_candidate_census/audit_v1'
    io.complete(census)
    hashes = io.read(census/'protocol.json')['source_sha256'].copy()
    for path in [DESIGN, 'work/experiments/support_language_chain_v1.py', 'work/experiments/'+Path(__file__).name,
                 'work/experiments/test_dyadic_branch_search_v1.py', 'work/experiments/branch_image_chain_v1.py']:
        h = io.sha(root/path)
        if path in hashes: assert hashes[path] == h
        hashes[path] = h
    for path, h in hashes.items(): assert io.sha(root/path) == h
    pred = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'
    rows = [r for r in io.read(pred/'rows.json') if r['seed'] == 328000000 and r['method'].startswith('online_')]
    assert len(rows) == 12
    io.save(out/'protocol.json', dict(source_sha256=hashes, census_summary_sha256=io.sha(census/'summary.json'),
        query_targets_accessed=False, posterior_reference_accessed=False, real_state_seed=328000000))
    tested = exhaustive(); io.save(out/'exhaustive.json', tested); print(tested, flush=True)
    records = []; checked = 0
    for row in sorted(rows, key=lambda r: r['method']):
        args, _ = load_case(pred, row)
        result = language.propose(*args, k=8)
        problem = integer.IntegerProblem(*args[:3])
        for p in result['proposals']:
            pattern = np.array(list(bytes.fromhex(p['mode']))).reshape(4, 4)
            assert all(possible(args[0][i], args[1][i], pattern[:, i]) for i in range(4))
            value = reference.fixed_value(*args[:3], pattern)
            assert value == F(p['lower']) <= 0
            checked += 1
        records.append(dict(method=row['method'], source_file=row['file'], source_sha256=row['sha256'], result=result))
        print(dict(real_state=row['method'], proposals=len(result['proposals']), meta=result['meta']), flush=True)
    for path, h in hashes.items(): assert io.sha(root/path) == h
    io.save(out/'calls.json', records)
    summary = dict(passed=True, tests=tested, real_states=len(records), independently_verified_proposals=checked,
        seconds_not_benchmark=time.perf_counter()-start, query_targets_accessed=False, posterior_reference_accessed=False,
        core_research_goal_complete=False, outputs_sha256={n: io.sha(out/n) for n in ['protocol.json', 'exhaustive.json', 'calls.json']})
    io.save(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/support_language_primitive/preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
