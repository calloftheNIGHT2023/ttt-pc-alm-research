"""334 full saved-state equivalence, exhaustive tiny cases and paired timing."""
import hashlib
from itertools import product
import json
from pathlib import Path
import statistics
import time
import traceback
from fractions import Fraction
import numpy as np
import branch_image_chain_v1 as reference
import branch_image_chain_dyadic_v1 as candidate

BASE = 'results/online_credit_fresh_pilot'
DESIGN = 'outputs/ttt-pc-alm-research/334_dyadic_branch_search_protocol_v1.md'
COUNT_FIELDS = ['layer_rows_evaluated', 'empty_transitions', 'retained_transition_rows',
                'pair_outputs', 'max_retained_dp_entries']


def read(path): return json.loads(path.read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)


def semantic(result): return {k: v for k, v in result.items() if k != 'meta'}


def compare(got, expected):
    assert semantic(got) == semantic(expected), (semantic(got), semantic(expected))
    for key in COUNT_FIELDS:
        assert got['meta'][key] == expected['meta'][key], key


def selftest():
    rng = np.random.default_rng(334193); cases = patterns = shells = 0
    for n, depth in [(1, 1), (1, 2), (2, 1), (2, 2)]:
        for case in range(12):
            x, v = rng.uniform(0, 1, (2, n)); credit = rng.normal(size=(depth, n))
            if case == 0: credit[:] = 0
            if case == 1: x[:] = 0; v[:] = 0
            if case == 2: x[:] = .5; v[:] = 1
            if case == 3: credit[:] = float.fromhex('0x0.0000000000001p-1022')
            if case == 4: x[:] = 1; v[:] = -.001
            if case == 5: x[:] = .12; v[:] = .001
            original = rng.integers(0, 4, size=(depth, n))
            problem = candidate.IntegerProblem(x, v, credit); brute = {}
            for flat in product(range(4), repeat=n*depth):
                pattern = np.array(flat).reshape(depth, n)
                expected = reference.fixed_value(x, v, credit, pattern)
                got = problem.fixed(pattern)
                assert (None if got is None else Fraction(got, problem.denominator)) == expected
                if expected is not None:
                    distance = int(np.count_nonzero(pattern != original))
                    brute.setdefault(distance, []).append((expected, flat))
                patterns += 1
            for k in [1, 3, 8]:
                got, _ = candidate.integer_shells(problem, original, k)
                converted = {m: [(Fraction(val, problem.denominator), path) for val, path in pool]
                             for m, pool in got.items()}
                assert converted == {m: sorted(pool)[:k] for m, pool in brute.items()}
                compare(candidate.propose(x, v, credit, original, k), reference.propose(x, v, credit, original, k))
                shells += 1
            cases += 1
    for bad in [float('inf'), float('-inf'), float('nan')]:
        try: candidate.IntegerProblem([bad], [0.], [[0.]])
        except ValueError: pass
        else: raise AssertionError('Nonfinite input accepted')
    return dict(cases=cases, enumerated_patterns=patterns, exact_k_best_comparisons=shells, rejected_nonfinite=3)


def load_case(pred, row):
    ap, mp = pred/row['file'], pred/row['metadata_file']
    assert sha(ap) == row['sha256'] and sha(mp) == row['metadata_sha256']
    meta = read(mp)['metadata']; assert not row['execution_failed'] and not meta['query_targets_accessed']
    with np.load(ap, allow_pickle=False) as z:
        x, v, credit = z['x_observed'], z['v_observed'], z['trigger_credit']
    pattern = np.array(list(bytes.fromhex(meta['selected_state']['original_mode'])), dtype=np.uint8).reshape(4, 4)
    return (x, v, credit, pattern), meta['proposal']


def run(root, out):
    begin = time.perf_counter(); pred = root/BASE/'pilot_predictions_v2'
    summary = read(pred/'summary.json'); assert summary['passed'] and summary['tasks'] == 512
    assert sha(pred/'rows.json') == summary['outputs_sha256']['rows.json']
    rows = [r for r in read(pred/'rows.json') if r['method'].startswith('online_')]
    rows.sort(key=lambda r: (r['seed'], r['method'])); assert len(rows) == 6144
    original_hashes = read(pred/'protocol.json')['source_sha256']
    for name in ['branch_image_chain_v1.py', 'factorized_dual_branch_search_v1.py']:
        assert sha(root/'work/experiments'/name) == original_hashes[name]
    paths = [DESIGN, 'work/experiments/branch_image_chain_dyadic_v1.py',
             'work/experiments/test_dyadic_branch_search_v1.py', 'work/experiments/branch_image_chain_v1.py',
             'work/experiments/factorized_dual_branch_search_v1.py']
    hashes = {p: sha(root/p) for p in paths}
    write(out/'protocol.json', dict(source_sha256=hashes, prediction_summary_sha256=sha(pred/'summary.json'),
        calls=6144, timing_seeds=list(range(328000000, 328000004)), timing_repetitions=2,
        timing_order_seed=334929, query_targets_accessed=False, implementation_only=True))
    tests = selftest(); write(out/'selftest.json', tests); print(tests, flush=True)
    records, timing_cases = [], []
    for index, row in enumerate(rows):
        args, expected = load_case(pred, row); start = time.perf_counter()
        got = candidate.propose(*args, k=8); seconds = time.perf_counter()-start
        compare(got, expected)
        records.append(dict(seed=row['seed'], method=row['method'], equivalent=True, seconds=seconds,
            semantic_sha256=hashlib.sha256(json.dumps(semantic(got), sort_keys=True).encode()).hexdigest(),
            proposals=len(got['proposals']), coordinate_scale_bits=got['meta']['coordinate_scale_bits'],
            credit_scale_bits=got['meta']['credit_scale_bits']))
        if row['seed'] < 328000004:
            timing_cases.append((row['seed'], row['method'], args, expected))
        if (index+1) % 384 == 0:
            print(dict(equivalent_calls=index+1, total=6144, seconds=time.perf_counter()-begin), flush=True)
    write(out/'equivalence.json', records)
    assert len(timing_cases) == 48
    # Same-process new/old interleaved kernel timing, including interface conversion.
    for function in [reference.propose, candidate.propose]: function(*timing_cases[0][2], k=8)
    rng = np.random.default_rng(334929); timings = []
    for seed, name, args, expected in timing_cases:
        for rep in range(2):
            for which in rng.permutation(2):
                function = reference.propose if which == 0 else candidate.propose
                tick = time.perf_counter(); got = function(*args, k=8); seconds = time.perf_counter()-tick
                compare(got, expected)
                timings.append(dict(seed=seed, method=name, repetition=rep,
                    implementation='fraction' if which == 0 else 'integer', seconds=seconds))
    assert len(timings) == 192; write(out/'timings.json', timings)
    groups = []
    for name in sorted({r['method'] for r in timings}):
        means = {impl: statistics.fmean(r['seconds'] for r in timings if r['method'] == name and r['implementation'] == impl)
                 for impl in ['fraction', 'integer']}
        groups.append(dict(method=name, **means, fraction_over_integer=means['fraction']/means['integer']))
    for path, digest in hashes.items(): assert sha(root/path) == digest
    result = dict(passed=True, selftest=tests, saved_state_equivalences=6144, timing_calls=192,
        timing_tasks=4, timing_states=48, timing_methods=groups, seconds=time.perf_counter()-begin,
        query_targets_accessed=False, complete_online_speedup_measured=False, core_research_goal_complete=False,
        outputs_sha256={n: sha(out/n) for n in ['protocol.json', 'selftest.json', 'equivalence.json', 'timings.json']})
    write(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'dyadic_branch_kernel_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        write(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
