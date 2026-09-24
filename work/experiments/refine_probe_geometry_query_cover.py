"""Bounded, label-free refinement of the first geometry's query coverage.

The hard-point pilot uses the previously recorded worst v1 grid location.
The adaptive stage reuses the completed v3 grid's hash-bound independent
checks, then independently checks every new integral. No model or gate changes.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import time

from probe_simplex_readout_intervals_v3 import integrate
from audit_probe_simplex_readout_intervals_v3 import verify
from probe_query_interval_cover import cover, verify_complete, dyadic_inside
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def choose_refinement(record, budget, position, lipschitz):
    """Try to bridge the widest gap, or a positive bounded portion of it."""
    budget, position, lipschitz = map(F, (budget, position, lipschitz))
    assert not record['complete'] and budget > position and lipschitz > 0
    gaps = [(F(a), F(b)) for a, b in record['gaps']]
    left, right = max(gaps, key=lambda ab: (ab[1] - ab[0], -ab[0]))
    query = dyadic_inside(left, right)
    if query is None:
        return None
    # Do not require a negative point target when a gap is too wide.
    # Half of the available budget is reserved for a useful local radius.
    radius = min(max(query - left, right - query),
                 (budget - position) / (2 * lipschitz))
    target = budget - position - lipschitz * radius
    assert left < query < right and radius > 0 and target > 0
    return dict(query=str(query), gap=[str(left), str(right)],
                intended_radius=str(radius), target=str(target))


def checked_inputs(base, source, src):
    directory = base / source
    summary = read(directory / 'summary.json')
    assert summary['diagnosis_complete'] and not summary['query_targets_accessed']
    for name, digest in summary['outputs_sha256'].items():
        assert sha(directory / name) == digest, name
    protocol = read(directory / 'protocol.json')
    for name, digest in protocol['source_sha256'].items():
        assert sha(src / name) == digest, name
    inputs = read(directory / 'inputs.json')
    assert not inputs['query_targets_accessed']
    roots = [tuple(tuple(F(t) for t in v) for v in simplex)
             for simplex in inputs['roots']]
    coefficients = [F(t) for t in inputs['coefficients']]
    assert sum(coefficients) == 0
    budget = F(protocol['exact_nominal_budget'])
    position = F(protocol['exact_position_bound'])
    constant = F(protocol['exact_query_lipschitz'])
    assert budget == F(1, 10**12) and 0 <= position < budget
    assert constant == 16 * sum(abs(t) for t in coefficients) and constant > 0
    return directory, summary, protocol, roots, coefficients, budget, position, constant


def prerequisites(base, src):
    hashes = {}
    records = {}
    for name in ['simplex_interval_selftests_v3', 'deep_simplex_envelope_selftests_v1']:
        directory = base / name
        summary = read(directory / 'summary.json')
        assert summary['passed'] and not summary['query_targets_accessed']
        for filename, digest in summary['outputs_sha256'].items():
            assert sha(directory / filename) == digest
        for filename, digest in summary['source_sha256'].items():
            assert sha(src / filename) == digest
            assert filename not in hashes or hashes[filename] == digest
            hashes[filename] = digest
        records[name] = sha(directory / 'summary.json')
    directory = base / 'query_cover_selftests_v1'
    summary = read(directory / 'summary.json')
    assert summary['passed'] and summary['tests'] == 13
    assert sha(directory / 'tests.json') == summary['tests_sha256']
    assert sha(src / 'probe_query_interval_cover.py') == summary['source_sha256']
    hashes['probe_query_interval_cover.py'] = summary['source_sha256']
    records[directory.name] = sha(directory / 'summary.json')
    hashes[Path(__file__).name] = sha(Path(__file__))
    return hashes, records


def selftest(base):
    out = base / 'adaptive_query_cover_selftests_v1'
    assert not out.exists()
    tests = {}
    for name, points, constant in [
        ('wide_gap', [], F(4)),
        ('small_gap', [(F(0), F(0)), (F(1), F(0))], F(3)),
        ('unequal_radii', [(F(0), F(0)), (F(1), F(1, 2))], F(4)),
    ]:
        initial = cover(points, F(1), F(0), constant)
        choice = choose_refinement(initial, F(1), F(0), constant)
        assert choice is not None
        query, target = F(choice['query']), F(choice['target'])
        updated = cover(points + [(query, target)], F(1), F(0), constant)
        assert F(updated['exact_uncovered_length']) < F(initial['exact_uncovered_length'])
        tests[name] = dict(passed=True, choice=choice)
    points = []
    record = cover(points, F(1), F(0), F(4))
    for iteration in range(16):
        choice = choose_refinement(record, F(1), F(0), F(4))
        points.append((F(choice['query']), F(choice['target'])))
        updated = cover(points, F(1), F(0), F(4))
        assert F(updated['exact_uncovered_length']) < F(record['exact_uncovered_length'])
        record = updated
        if record['complete']:
            break
    assert record['complete']
    tests['bounded_complete_cover'] = verify_complete(record, points, F(1), F(0), F(4))
    out.mkdir()
    exclusive_json(out / 'tests.json', tests)
    summary = dict(passed=True, tests=len(tests), query_targets_accessed=False,
                   audit_gate_passed=False, source_sha256=sha(Path(__file__)),
                   tests_sha256=sha(out / 'tests.json'))
    exclusive_json(out / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--stage', choices=['selftest', 'hard-point', 'adaptive'], required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    src = Path(__file__).parent
    base = root / 'results/probe_credit_confirmation'
    if args.stage == 'selftest':
        selftest(base)
        return
    assert not (base / 'evaluation_v3').exists(), 'Diagnosis must precede query scoring'
    hashes, tests = prerequisites(base, src)
    own_test = base / 'adaptive_query_cover_selftests_v1'
    tested = read(own_test / 'summary.json')
    assert tested['passed'] and tested['tests'] == 4
    assert tested['source_sha256'] == sha(Path(__file__))
    assert sha(own_test / 'tests.json') == tested['tests_sha256']
    tests[own_test.name] = sha(own_test / 'summary.json')
    source = 'simplex_readout_integral_diagnosis_' + ('v1' if args.stage == 'hard-point' else 'v3')
    inp, old, protocol, roots, coefficients, budget, position, constant = checked_inputs(base, source, src)
    for name, digest in protocol['source_sha256'].items():
        assert name not in hashes or hashes[name] == digest
        hashes[name] = digest
    out = base / ('geometry_integral_hardpoint_v1' if args.stage == 'hard-point'
                  else 'geometry_adaptive_query_cover_v1')
    assert not out.exists()
    out.mkdir()
    (out / 'queries').mkdir()
    max_splits, max_queries = 32768, (1 if args.stage == 'hard-point' else 64)
    exclusive_json(out / 'protocol.json', dict(
        stage=args.stage, input_directory=source, input_summary_sha256=sha(inp / 'summary.json'),
        prerequisites=tests, source_sha256=hashes, max_splits_per_query=max_splits,
        max_new_queries=max_queries, exact_budget=str(budget), exact_position_bound=str(position),
        exact_query_lipschitz=str(constant), query_targets_accessed=False, audit_gate_passed=False,
        scope='First geometry nominal-distribution diagnostic only; old arrays, labels and acceptance gates unchanged'))
    points, inherited = [], []
    if args.stage == 'adaptive':
        manifest = read(inp / 'files.json')
        assert len(manifest) == 257 and old['counts']['queries'] == 257
        for index in range(257):
            name = f'queries/{index:03d}.json'
            assert sha(inp / name) == manifest[name]
            row = read(inp / name)
            proof, checked = row['proof'], row['independent_check']
            assert row['query_index'] == index and not row['query_targets_accessed']
            assert checked['passed'] and F(proof['query_coordinate']) == F(index, 256)
            assert [F(t) for t in proof['coefficients']] == coefficients
            for field in ['exact_lower', 'exact_upper', 'exact_absolute_upper_bound']:
                assert F(proof[field]) == F(checked[field])
            bound = F(proof['exact_absolute_upper_bound'])
            assert bound == max(abs(F(proof['exact_lower'])), abs(F(proof['exact_upper'])))
            assert F(row['exact_nominal_absolute_upper_bound']) == bound + position
            points.append((F(index, 256), bound))
            inherited.append(dict(file=name, sha256=manifest[name]))
    exclusive_json(out / 'inherited.json', inherited)
    coverage = cover(points, budget, position, constant)
    exclusive_json(out / 'initial_cover.json', coverage)
    files = {}
    start = time.perf_counter()
    reason = 'query_work_limit'
    violations = []
    for iteration in range(max_queries):
        if args.stage == 'hard-point':
            # Chosen from the completed v1 bound, not from task targets or quality.
            assert old['worst_query_indices'] == [49]
            choice = dict(query=str(F(49, 256)), target=str((budget - position) / 2),
                          selection='Previously saved worst v1 bound, index49')
        else:
            if coverage['complete']:
                reason = 'continuous_cover_complete'
                break
            choice = choose_refinement(coverage, budget, position, constant)
            if choice is None:
                reason = 'dyadic_work_limit'
                break
        query, target = F(choice['query']), F(choice['target'])
        answer = integrate(query, roots, coefficients, target=target, max_splits=max_splits)
        checked = verify(roots, coefficients, query, 4, answer)
        upper = F(answer['exact_absolute_upper_bound'])
        low, high = F(answer['exact_lower']), F(answer['exact_upper'])
        actual_lower = max(F(0), low if low > 0 else -high if high < 0 else F(0)) - position
        if actual_lower > budget:
            violations.append(dict(query=str(query), exact_nominal_lower_bound=str(actual_lower)))
        points.append((query, upper))
        updated = cover(points, budget, position, constant)
        filename = f'queries/{iteration:03d}.json'
        exclusive_json(out / filename, dict(choice=choice, proof=answer, independent_check=checked,
            exact_nominal_absolute_upper_bound=str(upper + position),
            exact_nominal_absolute_lower_bound=str(max(F(0), actual_lower)),
            exact_uncovered_before=coverage['exact_uncovered_length'],
            exact_uncovered_after=updated['exact_uncovered_length'], query_targets_accessed=False))
        files[filename] = sha(out / filename)
        print(json.dumps(dict(iteration=iteration, query=float(query), splits=answer['splits'],
            status=answer['status'], nominal_upper=float(upper + position),
            uncovered=float(F(updated['exact_uncovered_length'])),
            seconds=time.perf_counter() - start, query_targets_accessed=False)), flush=True)
        improved = F(updated['exact_uncovered_length']) < F(coverage['exact_uncovered_length'])
        coverage = updated
        if violations:
            reason = 'proved_nominal_budget_violation'
            break
        if args.stage == 'hard-point':
            reason = 'hard_point_finished'
            break
        if not improved:
            reason = 'bounded_refinement_no_certified_progress'
            break
    if args.stage == 'adaptive' and coverage['complete']:
        reason = 'continuous_cover_complete'
    independent_cover = (verify_complete(coverage, points, budget, position, constant)
                         if coverage['complete'] else None)
    for name, digest in hashes.items():
        assert sha(src / name) == digest
    exclusive_json(out / 'final_cover.json', coverage)
    exclusive_json(out / 'points.json', [[str(q), str(u)] for q, u in points])
    exclusive_json(out / 'files.json', files)
    result = dict(diagnosis_complete=True, reason=reason, new_queries=len(files),
        inherited_queries=len(inherited), continuous_query_bound_certified=coverage['complete'],
        independent_cover_check=independent_cover, confirmed_nominal_gate_violations=violations,
        exact_uncovered_length=coverage['exact_uncovered_length'],
        seconds=time.perf_counter() - start, query_targets_accessed=False, audit_gate_passed=False,
        scope='One geometry only; no full-path acceptance or task-quality result',
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','inherited.json','initial_cover.json',
                                              'final_cover.json','points.json','files.json']})
    exclusive_json(out / 'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
