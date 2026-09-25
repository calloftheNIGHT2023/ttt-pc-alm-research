"""330 sealed-run mechanism check: regions, certificates and particle support.

No candidate is executed, and no 328 scoring code, query truth, or complete
posterior is read. The independent certificate helpers have historical shared
imports, but this program calls only their rational verification functions.
The original scientific experiment and statistical family remain unchanged.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from audit_post_escape_geometry_v1 import certificates
from audit_stasis_escape_geometry_v1 import inequalities

BASE = 'results/online_credit_fresh_pilot'
DESIGN = 'outputs/ttt-pc-alm-research/330_fresh_pool_mechanism_protocol_v1.md'
FREEZE = 'outputs/ttt-pc-alm-research/330_fresh_pool_mechanism_freeze_v1.json'
POLICIES = ['first_fit', 'uniform_state']
CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
NONDUAL = ['residual', 'bp', 'random_sign', 'zero']
CANDIDATES = ['online_first_fit_dual', 'online_uniform_state_dual_plus_residual']
SOURCES = ['audit_online_credit_fresh_pools_v1.py', 'audit_post_escape_geometry_v1.py', 'audit_stasis_escape_geometry_v1.py']
TOLERANCE = 1e-7


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8') as f:
        f.write(text + '\n')


def forward(points, coordinates):
    value = np.broadcast_to(coordinates, (len(points), len(coordinates))).copy()
    for layer in range(4):
        z = value + points[:, layer, None]
        value = np.where((z < 0) | (z > 1), 0., np.where(z <= .5, 2*z, 2-2*z))
    return value


def same_array(left, right):
    return left.shape == right.shape and left.dtype == right.dtype and left.tobytes() == right.tobytes()


def exact_trigger(b, x, v):
    values = [Fraction(float(t)) for t in x]; codes = []
    for bias in b:
        z = [t + Fraction(float(bias)) for t in values]
        codes.extend(int(t >= 0) + int(t >= Fraction(1, 2)) + int(t >= 1) for t in z)
        values = [max(Fraction(0), min(2*t, 2-2*t)) for t in z]
    residual = [max(Fraction(0), abs(a-Fraction(float(y)))-Fraction(.001)) for a, y in zip(values, v)]
    loss = sum((t*t for t in residual), Fraction(0)) / (2*len(x))
    return loss, bytes(codes).hex()


def exclusive_modes(candidate, controls, pools, failures):
    required = [candidate] + list(controls)
    if any(failures[name] for name in required):
        return dict(judged=False, unique_modes=None, failed_required_methods=[n for n in required if failures[n]])
    other = set().union(*(pools[name] for name in controls))
    return dict(judged=True, unique_modes=sorted(pools[candidate] - other), failed_required_methods=[])


def selftest():
    points = np.zeros((1, 4)); q = np.arange(33, dtype=float) / 32
    expected = []
    for x in q:
        value = Fraction(float(x))
        for _ in range(4):
            value = max(Fraction(0), min(2*value, 2-2*value))
        expected.append(float(value))
    assert np.array_equal(forward(points, q)[0], expected)
    pools = {'c': {'a', 'b', 'u'}, 'x': {'a'}, 'y': {'b'}}
    failures = {name: False for name in pools}
    assert exclusive_modes('c', ['x', 'y'], pools, failures)['unique_modes'] == ['u']
    pools['y'].add('u'); assert exclusive_modes('c', ['x', 'y'], pools, failures)['unique_modes'] == []
    failures['x'] = True
    assert exclusive_modes('c', ['x', 'y'], pools, failures) == dict(judged=False, unique_modes=None, failed_required_methods=['x'])
    assert same_array(np.array([1.]), np.array([1.]))
    assert not same_array(np.array([1.]), np.array([1.], dtype=np.float32))
    assert not same_array(np.array([0.]), np.array([-0.]))
    loss, pattern = exact_trigger(np.zeros(4), q[:4], np.array(expected[:4]))
    assert loss == 0 and len(bytes.fromhex(pattern)) == 16
    assert exact_trigger(np.zeros(4), np.array([0.]), np.array([.1]))[0] > 0
    return dict(passed=True, scalar_forward_coordinates=33, exclusive_pool_cases=3, bitwise_cases=3,
        exact_support_cases=2, query_targets_accessed=False, posterior_reference_accessed=False)


def gate(root, stage):
    base = root / BASE
    # No task metadata or arrays are opened until all three successful seals exist.
    folders = [base / f'{stage}_{kind}_v2' for kind in ['predictions', 'evaluation', 'audit']]
    for folder in folders:
        assert (folder / 'summary.json').is_file() and not (folder / 'failure.json').exists(), folder
    ps, es, au = [read(folder / 'summary.json') for folder in folders]
    assert ps['passed'] and es['passed'] and au['passed']
    assert au['prediction_summary_sha256'] == sha(folders[0] / 'summary.json')
    assert au['evaluation_summary_sha256'] == sha(folders[1] / 'summary.json')
    n = 512 if stage == 'pilot' else 3
    assert ps['tasks'] == es['tasks'] == n and au['counts']['predictors'] == 51*n
    assert es['stage'] == au['stage'] == stage
    for name, digest in ps['outputs_sha256'].items():
        assert sha(folders[0] / name) == digest
    protocol = read(folders[0] / 'protocol.json')
    assert protocol['stage'] == stage and len(protocol['seeds']) == n
    assert not protocol['query_targets_accessed']
    assert not read(folders[0] / 'before_query_manifest.json')['query_targets_accessed']
    for name, digest in protocol['source_sha256'].items():
        assert sha(root / 'work/experiments' / name) == digest
    frozen = read(root / FREEZE)
    assert frozen['design_sha256'] == sha(root / DESIGN)
    assert frozen['query_quality_accessed_when_written'] is False
    assert frozen['source_sha256'] == {name: sha(root / 'work/experiments' / name) for name in SOURCES}
    assert selftest()['passed']
    return folders[0], protocol, au


def run(root, out, stage):
    start = time.perf_counter(); folder, protocol, main_audit = gate(root, stage)
    names = [f'online_{p}_{c}' for p in POLICIES for c in CHANNELS]
    rows = read(folder / 'rows.json'); index = {(r['seed'], r['method']): r for r in rows}
    counts = Counter(); maxima = Counter(); aggregates = {name: Counter() for name in names}; task_records = []
    save(out / 'protocol.json', dict(stage=stage, design_sha256=sha(root / DESIGN), freeze_sha256=sha(root / FREEZE),
        source_sha256={name: sha(root / 'work/experiments' / name) for name in SOURCES},
        prediction_summary_sha256=sha(folder / 'summary.json'),
        independent_main_audit_sha256=sha(root / BASE / f'{stage}_audit_v2/summary.json'),
        numeric_sampling_tolerance=TOLERANCE, methods=names, candidates=CANDIDATES,
        query_targets_accessed=False, posterior_reference_accessed=False, new_significance_tests=False))
    for seed in protocol['seeds']:
        pools = {}; failures = {}; logs = {}; arrays = {}; certificates_seen = {}; same_pools = {}; states = {}; original_pool = None
        task = dict(seed=seed, methods={}, candidates={})
        for name in names:
            row = index[seed, name]
            assert sha(folder / row['file']) == row['sha256']
            assert sha(folder / row['metadata_file']) == row['metadata_sha256']
            log = read(folder / row['metadata_file']); meta = log['metadata']; logs[name] = meta
            assert log['seed'] == seed and log['method'] == name and not meta['query_targets_accessed']
            failures[name] = bool(meta['execution_failed']); aggregates[name]['tasks'] += 1
            if failures[name]:
                pools[name] = set(); aggregates[name]['execution_failures'] += 1
                task['methods'][name] = dict(execution_failed=True, positive_modes=None)
                continue
            with np.load(folder / row['file'], allow_pickle=False) as z:
                a = {key: z[key] for key in ['points', 'allocation', 'prediction', 'x_observed', 'v_observed',
                    'trigger_b', 'trigger_h', 'trigger_u', 'trigger_location', 'trigger_credit']}
            arrays[name] = a; x = a['x_observed']; v = a['v_observed']; points = a['points']; allocation = a['allocation']
            keys = meta['positive_modes']; assert keys == sorted(set(keys)); pools[name] = set(keys)
            original = tuple(meta['original_positive_modes'])
            if original_pool is None:
                original_pool = original
            assert original == original_pool and set(original).issubset(keys)
            assert meta['new_positive_modes'] == sorted(set(keys) - set(original))
            policy = meta['policy']; assert name == f"online_{policy}_{meta['channel']}"
            exact_loss, exact_mode = exact_trigger(a['trigger_b'], x, v)
            assert Fraction(meta['selected_state']['exact_support_loss']) == exact_loss
            assert meta['selected_state']['original_mode'] == exact_mode
            assert meta['selected_state']['support_fit'] == (exact_loss == 0)
            counts['exact_trigger_support_checks'] += 1
            if exact_loss == 0:
                aggregates[name]['support_fit_tasks'] += 1
                aggregates[name]['support_fit_with_nonzero_multiplier'] += bool(np.any(a['trigger_u'] != 0))
                if meta['channel'] == 'bp':
                    credit_max = float(np.max(abs(a['trigger_credit'])))
                    maxima['support_fit_bp_credit_max_abs'] = max(maxima['support_fit_bp_credit_max_abs'], credit_max)
                    aggregates[name]['support_fit_bp_credit_exact_zero'] += (credit_max == 0)
                    aggregates[name]['support_fit_bp_credit_at_most_1e_12'] += (credit_max <= 1e-12)
            if policy not in states:
                states[policy] = {key: a[key] for key in ['trigger_b', 'trigger_h', 'trigger_u', 'trigger_location']}
            else:
                for key, value in states[policy].items():
                    assert same_array(value, a[key]), (seed, name, key)
                    counts['same_trigger_bitwise_arrays'] += 1
            assert not meta['trace_enabled'] and not meta['uses_complete_posterior_reference']
            if meta['channel'] != 'bp':
                assert meta['no_global_bp_guard_enabled']
            assert np.isfinite(points).all() and np.max(abs(points)) <= .12 + TOLERANCE
            expected_shape = (2048, 4) if keys else (1, 4)
            assert points.shape == expected_shape
            if keys:
                assert np.issubdtype(allocation.dtype, np.integer) and allocation.shape == (len(keys),)
                assert np.all(allocation >= 0) and int(allocation.sum()) == 2048
                support_error = float(np.max(abs(forward(points, x) - v)))
                assert support_error <= .001 + TOLERANCE, (seed, name, support_error)
                maxima['sample_support_abs_error'] = max(maxima['sample_support_abs_error'], support_error)
                offset = 0
                for key, count in zip(keys, allocation):
                    matrix, rhs = inequalities(x, v, key)
                    if count:
                        residual = points[offset:offset+int(count)] @ np.asarray(matrix, dtype=float).T - np.asarray(rhs, dtype=float)
                        violation = max(0., float(np.max(residual)))
                        assert violation <= TOLERANCE, (seed, name, key, violation)
                        maxima['sample_constraint_violation'] = max(maxima['sample_constraint_violation'], violation)
                    offset += int(count); counts['region_allocation_checks'] += 1
                counts['support_checked_particles'] += len(points)
            else:
                assert allocation.shape == (0,)
                aggregates[name]['empty_pool_fallback_tasks'] += 1
            for key, note in meta['new_mode_classifications_detail'].items():
                if key not in certificates_seen:
                    matrix, rhs = inequalities(x, v, key)
                    counts['unique_exact_certificates'] += certificates(matrix, rhs, note)
                    certificates_seen[key] = note; counts['unique_classifications'] += 1
                    counts['classification_' + note['classification']] += 1
                    if note['positive_volume_certified'] and not note['numerical_volume_available']:
                        counts['positive_certificate_without_numeric_volume'] += 1
                else:
                    prior = certificates_seen[key]
                    for field in ['classification', 'certificates', 'volume', 'positive_volume_certified', 'numerical_volume_available']:
                        assert note[field] == prior[field], (seed, key, field)
                counts['classification_instances'] += 1
                if key in keys:
                    assert note['positive_volume_certified'] and note['numerical_volume_available']
            for key in meta['new_positive_modes']:
                assert key in meta['new_mode_classifications_detail']
            pool_key = tuple(keys)
            if pool_key in same_pools:
                for field in ['points', 'allocation', 'prediction']:
                    assert same_array(a[field], same_pools[pool_key][field]), (seed, name, field)
                    counts['same_pool_bitwise_arrays'] += 1
            else:
                same_pools[pool_key] = {field: a[field] for field in ['points', 'allocation', 'prediction']}
            aggregates[name]['successful_calls'] += 1
            aggregates[name]['positive_task_mode_pairs'] += len(keys)
            aggregates[name]['new_vs_original_pairs'] += len(meta['new_positive_modes'])
            aggregates[name]['tasks_with_new_vs_original'] += bool(meta['new_positive_modes'])
            task['methods'][name] = dict(execution_failed=False, positive_modes=keys,
                new_vs_original=meta['new_positive_modes'], support_fit=meta['selected_state']['support_fit'],
                multiplier_max_abs=float(np.max(abs(a['trigger_u']))), credit_max_abs=float(np.max(abs(a['trigger_credit']))))
            counts['successful_calls'] += 1
        for candidate in CANDIDATES:
            policy = 'first_fit' if candidate == CANDIDATES[0] else 'uniform_state'
            controls = {'same_trigger_four': [f'online_{policy}_{c}' for c in NONDUAL],
                'both_triggers_eight': [f'online_{p}_{c}' for p in POLICIES for c in NONDUAL]}
            result = {}
            for label, methods in controls.items():
                item = exclusive_modes(candidate, methods, pools, failures)
                item['controls'] = methods; item['positive_certificate_sources'] = []
                if item['judged']:
                    for key in item['unique_modes']:
                        proof = logs[candidate]['new_mode_classifications_detail'][key]
                        assert proof['positive_volume_certified']
                        item['positive_certificate_sources'].append(dict(mode=key,
                            metadata_file=index[seed, candidate]['metadata_file'], metadata_sha256=index[seed, candidate]['metadata_sha256']))
                result[label] = item
            task['candidates'][candidate] = result
        counts['tasks'] += 1; task_records.append(task)
        if counts['tasks'] % 16 == 0:
            print(dict(audited_tasks=counts['tasks'], total=len(protocol['seeds']), seconds=time.perf_counter()-start), flush=True)
    credits = []
    for candidate in CANDIDATES:
        for label in ['same_trigger_four', 'both_triggers_eight']:
            rr = [r['candidates'][candidate][label] for r in task_records]
            credits.append(dict(candidate=candidate, comparison=label, judged_tasks=sum(r['judged'] for r in rr),
                unjudged_tasks=sum(not r['judged'] for r in rr),
                tasks_with_unique_positive=sum(bool(r['unique_modes']) for r in rr if r['judged']),
                unique_task_mode_pairs=sum(len(r['unique_modes']) for r in rr if r['judged'])))
    save(out / 'tasks.json', task_records); save(out / 'methods.json', {name: dict(values) for name, values in aggregates.items()})
    save(out / 'credit_exclusivity.json', credits)
    result = dict(passed=True, stage=stage, counts=dict(counts), maxima=dict(maxima), seconds=time.perf_counter()-start,
        credit_exclusivity=credits, numeric_sampling_tolerance=TOLERANCE, query_targets_accessed=False,
        posterior_reference_accessed=False, numerical_volume_accuracy_proved=False, iid_sampling_proved=False,
        new_significance_tests=False, core_research_goal_complete=False,
        outputs_sha256={n: sha(out / n) for n in ['protocol.json', 'tasks.json', 'methods.json', 'credit_exclusivity.json']})
    save(out / 'summary.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['preflight', 'pilot']); ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()
    if args.selftest:
        assert args.stage is None; print(json.dumps(selftest()), flush=True)
    else:
        assert args.stage is not None
        root = Path(__file__).resolve().parents[2]
        out = root / BASE / f'{args.stage}_pool_audit_v1'; out.mkdir(parents=True, exist_ok=False)
        try:
            run(root, out, args.stage)
        except Exception:
            save(out / 'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
            raise
