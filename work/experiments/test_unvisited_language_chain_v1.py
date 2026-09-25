"""350 exhaustive, independently scored complement-K-best tests."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import branch_image_chain_v1 as reference
import branch_image_chain_dyadic_v1 as integer
import support_language_chain_v1 as old
import unvisited_language_chain_v1 as new
from test_support_language_chain_v1 import possible

DESIGN = 'outputs/ttt-pc-alm-research/350_unvisited_language_protocol_v1.md'
SOURCES = ['unvisited_language_chain_v1.py', 'test_unvisited_language_chain_v1.py', 'census_unvisited_language_v1.py']


def source_hashes(root):
    base = root/'results/strong_pool_online/development_predictions_v1'; io.complete(base)
    hashes = io.read(base/'protocol.json')['source_sha256'].copy()
    for path in [DESIGN]+['work/experiments/'+n for n in SOURCES]:
        assert path not in hashes; hashes[path] = io.sha(root/path)
    for path, digest in hashes.items(): assert io.sha(root/path) == digest
    return hashes


def exhaustive():
    rng = np.random.default_rng(350929)
    counts = dict(cases=0, patterns=0, exact_shell_comparisons=0, empty_set_equivalences=0,
        old_unknown_retention_checks=0, all_forbidden_empty_checks=0, bad_inputs_rejected=0)
    for n, d in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)]:
        for case in range(6):
            x, v = rng.uniform(0, 1, (2, n)); a = rng.normal(size=(d, n)); original = rng.integers(0, 4, (d, n))
            if case == 1: a[:] = 0
            if case == 2: x[:] = .5; v[:] = 1
            if case == 3: a[:] = float.fromhex('0x0.0000000000001p-1022')
            if case == 4: v[:] = -.002
            if case == 5: x[:] = .12; v[:] = .001
            patterns = list(product(range(4), repeat=n*d)); brute = {}
            for flat in patterns:
                pattern = np.array(flat).reshape(d, n); counts['patterns'] += 1
                if not all(possible(x[i], v[i], pattern[:, i]) for i in range(n)): continue
                value = reference.fixed_value(x, v, a, pattern)
                if value is None: continue
                h = int(np.count_nonzero(pattern != original)); brute.setdefault(h, []).append((value, flat))
            brute = {h: sorted(pool) for h, pool in brute.items()}
            sets = [[], [bytes(p).hex() for p in patterns[::2]], [bytes(p).hex() for p in patterns],
                    sorted({bytes(p).hex() for pool in brute.values() for _, p in pool[:8]})]
            problem = integer.IntegerProblem(x, v, a)
            for fi, forbidden in enumerate(sets):
                forbidden_set = set(forbidden)
                for k in [1, 2, 8]:
                    expected = {h: [(val, path) for val, path in pool if bytes(path).hex() not in forbidden_set][:k]
                                for h, pool in brute.items()}
                    expected = {h: pool for h, pool in expected.items() if pool}
                    dp, meta, _ = new.shells(problem, original, k, forbidden)
                    assert {h: [(F(val, problem.denominator), path) for val, path in pool] for h, pool in dp.items()} == expected
                    assert meta['trie_nodes'] <= 1+len(forbidden_set)*d
                    proposed = new.propose(x, v, a, original, k, forbidden)
                    wanted = [dict(mode=bytes(path).hex(), hamming=h, rank=rank, lower=str(val))
                              for h, pool in sorted(expected.items()) if h >= 1
                              for rank, (val, path) in enumerate(pool) if val <= 0]
                    assert proposed['proposals'] == wanted
                    old_result = old.propose(x, v, a, original, k)
                    unknown = {p['mode'] for p in old_result['proposals']} - forbidden_set
                    assert unknown <= {p['mode'] for p in proposed['proposals']}
                    counts['old_unknown_retention_checks'] += 1
                    if fi == 0:
                        assert {key: val for key, val in proposed.items() if key not in ['meta', 'minimum_unvisited_hamming']} == {
                            key: val for key, val in old_result.items() if key != 'meta'}
                        counts['empty_set_equivalences'] += 1
                    if fi == 2:
                        assert not dp and not proposed['proposals']; counts['all_forbidden_empty_checks'] += 1
                    counts['exact_shell_comparisons'] += 1
            counts['cases'] += 1
    for key in ['0', 'ff00', '00', '000000', 'GG00', '00 00', '0A00', 1]:
        try: new.propose([.1], [.2], [[1.], [-1.]], [[1], [1]], forbidden=[key])
        except ValueError: counts['bad_inputs_rejected'] += 1
        else: raise AssertionError(('Bad forbidden mode accepted', key))
    for k in [0, -1, True, 1.5]:
        try: new.propose([.1], [.2], [[1.], [-1.]], [[1], [1]], k=k)
        except ValueError: counts['bad_inputs_rejected'] += 1
        else: raise AssertionError(('Bad k accepted', k))
    return counts


def run(root, out):
    start = time.perf_counter(); hashes = source_hashes(root)
    io.save(out/'protocol.json', dict(source_sha256=hashes, query_targets_accessed=False, posterior_reference_accessed=False))
    counts = exhaustive()
    for p, digest in hashes.items(): assert io.sha(root/p) == digest
    io.save(out/'checks.json', counts)
    result = dict(passed=True, counts=counts, seconds_not_benchmark=time.perf_counter()-start,
        query_targets_accessed=False, posterior_reference_accessed=False, core_research_goal_complete=False,
        outputs_sha256={n: io.sha(out/n) for n in ['protocol.json', 'checks.json']})
    io.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/unvisited_language/preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
