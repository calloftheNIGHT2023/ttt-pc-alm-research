"""339 exact extraction tests; gated behind complete 337 and positive 338 evidence."""
from fractions import Fraction
from itertools import product
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as suite
import branch_image_chain_v1 as reference
import branch_image_chain_dyadic_v1 as integer
import branch_image_chain_radius_v1 as radius
from test_dyadic_branch_search_v1 import COUNT_FIELDS, load_case, semantic

DESIGN = 'outputs/ttt-pc-alm-research/339_search_radius_primitive_protocol_v1.md'
KS = [1, 2, 4, 8]
FIELDS = ['current_lower', 'current_structurally_infeasible', 'current_certified_infeasible',
          'minimum_nonexcluded_hamming', 'necessary_hamming_lower_bound', 'shell_minima']


def nested(smaller, larger):
    for field in FIELDS: assert smaller[field] == larger[field], field
    for key in COUNT_FIELDS: assert smaller['meta'][key] == larger['meta'][key], key
    assert all(p in larger['proposals'] for p in smaller['proposals'])


def exhaustive():
    rng = np.random.default_rng(339193)
    cases = patterns = comparisons = nested_pairs = rejected = 0
    for n, d in [(1, 1), (1, 2), (2, 1), (2, 2), (1, 4)]:
        for case in range(6):
            x, v = rng.uniform(0, 1, (2, n))
            credit = rng.normal(size=(d, n)); original = rng.integers(0, 4, (d, n))
            if case == 0: credit[:] = 0
            if case == 1: x[:] = 0; v[:] = 0
            if case == 2: x[:] = .5; v[:] = 1
            if case == 3: credit[:] = float.fromhex('0x0.0000000000001p-1022')
            if case == 4: v[:] = -.002
            if case == 5: x[:] = .12; v[:] = .001
            brute = {}
            for flat in product(range(4), repeat=n*d):
                pattern = np.array(flat).reshape(d, n)
                value = reference.fixed_value(x, v, credit, pattern)
                if value is not None:
                    h = int(np.count_nonzero(pattern != original))
                    brute.setdefault(h, []).append((value, flat))
                patterns += 1
            brute = {h: sorted(pool) for h, pool in brute.items()}
            current = reference.fixed_value(x, v, credit, original)
            eligible = sorted(h for h, pool in brute.items() if h >= 1 and pool[0][0] <= 0)
            minimum = eligible[0] if eligible else None
            for k in KS:
                previous = None
                for span in [1, 2, 3, 6, 'all']:
                    got = radius.propose(x, v, credit, original, k=k, span=span)
                    wanted = []
                    maximum = None if minimum is None else n*d if span == 'all' else min(n*d, minimum+span-1)
                    if minimum is not None:
                        for h in range(minimum, maximum+1):
                            for rank, (value, path) in enumerate(brute.get(h, [])[:k]):
                                if value <= 0:
                                    wanted.append(dict(mode=bytes(path).hex(), hamming=h,
                                                       rank=rank, lower=str(value)))
                    expected = dict(current_lower=None if current is None else str(current),
                        current_structurally_infeasible=current is None,
                        current_certified_infeasible=current is None or current > 0,
                        minimum_nonexcluded_hamming=minimum,
                        necessary_hamming_lower_bound=minimum if current is None or current > 0 else None,
                        shell_minima={str(h): str(pool[0][0]) for h, pool in sorted(brute.items())},
                        proposals=wanted)
                    assert semantic(got) == expected
                    if previous is not None: nested(previous, got); nested_pairs += 1
                    previous = got; comparisons += 1
                    if span == 3:
                        baseline = integer.propose(x, v, credit, original, k=k)
                        assert semantic(got) == semantic(baseline)
                        for field in COUNT_FIELDS: assert got['meta'][field] == baseline['meta'][field]
            cases += 1
    for kwargs in [dict(k=0), dict(k=-1), dict(k=True), dict(k=1.5),
                   dict(span=0), dict(span=-1), dict(span=True), dict(span=1.5), dict(span='wide')]:
        try: radius.propose([0.], [0.], [[0.]], [[0]], **kwargs)
        except ValueError: rejected += 1
        else: raise AssertionError(('Invalid selector input accepted', kwargs))
    return dict(cases=cases, enumerated_patterns=patterns, exact_extraction_comparisons=comparisons,
                nested_pairs=nested_pairs, invalid_arguments_rejected=rejected)


def main(root, out):
    begin = time.perf_counter()
    calibration = root/'results/budget_reinvestment/calibration_v1'
    suite.complete(calibration)
    audit = suite.complete(root/'results/budget_reinvestment/audit_v1')
    assert audit['calibration_summary_sha256'] == suite.sha(calibration/'summary.json')
    assert suite.gate(root) == suite.read(calibration/'protocol.json')['source_sha256']
    reach = root/'results/search_radius_reachability/audit_v1'; suite.complete(reach)
    methods = suite.read(reach/'methods.json')
    union = next(m for m in methods if m['method'] == 'strong_union')
    assert union['certified_categories'].get('above_window', 0) > 0, 'No demonstrated radius bottleneck'
    env = suite.resources.old.environment_snapshot(root)
    assert not env['other_research_or_git_pack_processes'], env
    hashes = suite.read(calibration/'protocol.json')['source_sha256'].copy()
    for path in [DESIGN, 'work/experiments/branch_image_chain_radius_v1.py',
                 'work/experiments/test_search_radius_primitive_v1.py',
                 'work/experiments/test_dyadic_branch_search_v1.py']:
        if path in hashes: assert suite.sha(root/path) == hashes[path]
        hashes[path] = suite.sha(root/path)
    pred = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'; suite.complete(pred)
    rows = [r for r in suite.read(pred/'rows.json') if r['seed'] == 328000000 and r['method'].startswith('online_')]
    assert len(rows) == 12
    suite.save(out/'protocol.json', dict(source_sha256=hashes, calibration_summary_sha256=suite.sha(calibration/'summary.json'),
        reachability_summary_sha256=suite.sha(reach/'summary.json'), seed=328000000, ks=KS, spans=[3, 6, 'all'],
        query_targets_accessed=False, complete_online_prediction=False, environment=env))
    tests = exhaustive(); suite.save(out/'exhaustive.json', tests); print(tests, flush=True)
    records = []; source = []; comparisons = pairs = 0
    for row in sorted(rows, key=lambda r: r['method']):
        args, saved = load_case(pred, row)
        source.append(dict(method=row['method'], file=row['file'], sha256=row['sha256'],
                           metadata_file=row['metadata_file'], metadata_sha256=row['metadata_sha256']))
        for k in KS:
            baseline = integer.propose(*args, k=k); previous = None
            for span in [3, 6, 'all']:
                got = radius.propose(*args, k=k, span=span)
                if span == 3:
                    assert semantic(got) == semantic(baseline)
                    if k == 8: assert semantic(got) == semantic(saved)
                for field in COUNT_FIELDS: assert got['meta'][field] == baseline['meta'][field]
                if previous is not None: nested(previous, got); pairs += 1
                previous = got; comparisons += 1
                records.append(dict(method=row['method'], k=k, span=span, result=got))
        print(dict(real_state=row['method'], calls=comparisons, total=144), flush=True)
    for path, digest in hashes.items(): assert suite.sha(root/path) == digest
    suite.save(out/'calls.json', records); suite.save(out/'sources.json', source)
    result = dict(passed=True, tests=tests, real_state_calls=comparisons, real_state_nested_pairs=pairs,
        seconds_not_benchmark=time.perf_counter()-begin, query_targets_accessed=False,
        new_query_benefit_measured=False, core_research_goal_complete=False,
        outputs_sha256={name: suite.sha(out/name) for name in ['protocol.json', 'exhaustive.json', 'calls.json', 'sources.json']})
    suite.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/search_radius_primitive/preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
