"""308 post-seal common geometry evaluation. No posterior or query truth.

All proposals are immutable inputs. This script has no search or fitting path.
Numeric volumes, exact feasibility certificates, and unresolved cases are
reported separately. This is old-task mechanism development, not confirmation.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction as F
import gzip
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import complete_credit_mode_geometry_v1 as classifier
from exact_quadratic_events_v1 import encode

SEEDS = list(range(5910000, 5910064))
FAMILIES = ['dual', 'residual', 'random_sign']
CONTROL_NAMES = [f'{family}_{horizon}' for family in
    ['alm_keep', 'alm_reset', 'nodual', 'pc', 'adam', 'adam_perturb_001', 'adam_perturb_004']
    for horizon in ([1, 2, 4, 8, 64] if family.startswith('adam') else [64])]
METHODS = [f'{family}_{suffix}' for family in FAMILIES for suffix in
           ['event_open', 'event_all', 'event_boundary_only', 'grid_exact', 'grid_float']]
METHODS += ['control_' + name for name in CONTROL_NAMES] + ['original_only']
assert len(CONTROL_NAMES) == 19 and len(METHODS) == 35


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save(path, data):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(encode(data), stream, ensure_ascii=False, indent=2, allow_nan=False)


def hexmode(mode):
    return bytes(int(k) for k in mode).hex()


def load_support(root):
    folder = root / 'results/complete_credit_amplitude_events/development_v2'
    assert not (folder / 'failure.json').exists(), 'The full support phase failed'
    summary, manifest, protocol = [read(folder / name) for name in
        ['summary.json', 'before_geometry_manifest.json', 'protocol.json']]
    assert summary['support_execution_passed'] and summary['all_selected_states_processed']
    assert summary['states'] == 131 and summary['directions'] == 393 and not summary['smoke_only']
    assert manifest['all_selected_states_processed'] and not manifest['geometry_or_posterior_accessed']
    assert sha(folder / 'before_geometry_manifest.json') == summary['manifest_sha256']
    for name, digest in manifest['files_sha256'].items():
        assert sha(folder / name) == digest, name
    for name, digest in protocol['source_sha256'].items():
        assert sha(root / 'work/experiments' / name) == digest, name
    for name, digest in protocol['design_sha256'].items():
        assert sha(root / 'outputs/ttt-pc-alm-research' / name) == digest, name
    for name, digest in protocol['input_sha256'].items():
        assert sha(root / name) == digest, name
    rows = read(folder / 'rows.json')
    assert len(rows) == 393
    by_state = defaultdict(dict)
    for row in rows:
        key = (row['seed'], *row['location'])
        assert row['family'] not in by_state[key]
        assert row['sha256'] == manifest['files_sha256'][row['file']]
        by_state[key][row['family']] = row
    expected = {(task['seed'], *loc) for task in protocol['task_manifest'] for loc in task['locations']}
    assert set(by_state) == expected and len(expected) == 131
    assert all(set(rows) == set(FAMILIES) for rows in by_state.values())
    return folder, summary, protocol, by_state


def read_state(folder, grouped):
    methods, controls, common = {}, None, None
    payloads = {}
    for family in FAMILIES:
        row = grouped[family]
        with gzip.open(folder / row['file'], 'rt', encoding='utf-8') as stream:
            payload = json.load(stream)
        assert payload['family'] == family and payload['seed'] == row['seed'] and payload['location'] == row['location']
        state = {k: v for k, v in payload['state'].items() if k != 'direction'}
        if common is None:
            common, controls = state, payload['controls']
        else:
            assert common == state and controls == payload['controls']
        partition = payload['partition']
        assert partition['coverage_complete'] == row['coverage_complete']
        assert len(partition['cells']) == row['cells'] and len(partition['boundaries']) == row['boundaries']
        assert len(partition['pending']) == row['pending_intervals']
        open_modes = {hexmode(c['mode']) for c in partition['cells']}
        boundary = {hexmode(c['mode']) for c in partition['boundaries']}
        grid = {r['exact_mode'] for r in payload['seven_points']}
        float_grid = {r['float_mode'] for r in payload['seven_points']}
        assert sorted(open_modes) == row['open_modes'] and sorted(boundary) == row['boundary_modes']
        assert sorted(grid) == row['fixed_exact_modes'] and sorted(float_grid) == row['fixed_float_modes']
        if row['coverage_complete']:
            assert grid <= open_modes | boundary
        methods.update({family + '_event_open': open_modes, family + '_event_all': open_modes | boundary,
                        family + '_event_boundary_only': boundary - open_modes,
                        family + '_grid_exact': grid, family + '_grid_float': float_grid})
        payloads[family] = dict(coverage_complete=row['coverage_complete'], search_seconds=row['search_seconds'],
                              comparisons=row['comparisons'], cells=row['cells'],
                              full_diagnostic_seconds=row['full_diagnostic_seconds'],
                              floating_value_discrepancies=row['floating_value_discrepancies'],
                              floating_mode_discrepancies=row['floating_mode_discrepancies'])
    assert set(controls) == set(CONTROL_NAMES)
    methods.update({'control_' + name: set(controls[name]['modes']) for name in CONTROL_NAMES})
    assert set(methods) == set(METHODS) - {'original_only'}
    return common, methods, payloads


def certify_label(a, r, result):
    checks = classifier.verify_certificates(a, r, encode(result))
    label = result['classification']
    certs = result['certificates']
    positive = any(c.get('strict_interior', False) for c in certs)
    impossible = any(c['type'] == 'negative_constant_row' or c.get('conclusion') == 'infeasible' for c in certs)
    zero = any(c.get('conclusion') == 'zero_volume_or_empty' for c in certs)
    assert not (positive and (impossible or zero))
    assert result['positive_volume_certified'] == positive
    if label == 'positive_volume':
        assert positive
    elif label == 'infeasible':
        assert impossible and result['closed_region_feasible'] is False
    elif label == 'zero_volume_or_empty':
        assert zero and not impossible
    else:
        assert label == 'unresolved' and not (positive or impossible or zero)
    if result['numerical_volume_available']:
        assert positive and result['volume'] > 0
    return checks


def describe(modes, classified, old):
    modes = set(modes)
    positives = {m for m in modes if classified[m]['positive_volume_certified']}
    new_positive = positives - old
    counts = Counter(classified[m]['classification'] for m in modes)
    return dict(proposal_modes=sorted(modes), positive_modes=sorted(positives),
        new_positive_modes=sorted(new_positive), classifications=dict(counts),
        closed_feasible_certified=sum(classified[m]['closed_region_feasible'] is True for m in modes),
        unknown_modes=sorted(m for m in modes if classified[m]['classification'] == 'unresolved'),
        positive_modes_without_numerical_volume=sorted(m for m in positives if not classified[m]['numerical_volume_available']),
        known_positive_numeric_volume_sum=sum(classified[m]['volume'] for m in positives if classified[m]['volume'] is not None),
        known_new_positive_numeric_volume_sum=sum(classified[m]['volume'] for m in new_positive if classified[m]['volume'] is not None))


def comparisons(methods, classified, old):
    positive = lambda name: {m for m in methods[name] if classified[m]['positive_volume_certified']}
    du = positive('dual_event_all') - old
    other_directions = positive('residual_event_all') | positive('random_sign_event_all')
    strong = set().union(*(positive('control_' + n) for n in CONTROL_NAMES))
    return dict(dual_new_positive=sorted(du),
        dual_new_positive_missed_by_own_seven=sorted(du - positive('dual_grid_exact')),
        dual_new_positive_exclusive_vs_other_directions=sorted(du - other_directions),
        dual_new_positive_exclusive_vs_all_controls=sorted(du - strong),
        dual_new_positive_exclusive_vs_directions_and_controls=sorted(du - other_directions - strong),
        extra_positive_by_family={f: sorted(positive(f + '_event_all') - positive(f + '_grid_exact') - old) for f in FAMILIES},
        boundary_added_positive_by_family={f: sorted(positive(f + '_event_all') - positive(f + '_event_open') - old) for f in FAMILIES})


def run(root, out):
    begin = time.perf_counter()
    folder, support_summary, support_protocol, by_state = load_support(root)
    testing = root / 'results/complete_credit_amplitude_events/geometry_tests_v1'
    tested, test_protocol = read(testing / 'summary.json'), read(testing / 'protocol.json')
    assert tested['passed'] and not tested['real_task_geometry_accessed']
    for name, digest in tested['outputs_sha256'].items():
        assert sha(testing / name) == digest
    for name, digest in test_protocol['source_sha256'].items():
        assert sha(root / 'work/experiments' / name) == digest
    raw = root / 'results/certificate_activity_attribution/development'
    raw_manifest = read(raw / 'before_evaluation_manifest.json')
    assert sha(raw / 'rows.json') == raw_manifest['rows_sha256']
    oldrows = {r['seed']: r for r in read(raw / 'rows.json') if r['method'] == 'credit_control_probe33'}
    assert set(oldrows) == set(SEEDS)
    docs = root / 'outputs/ttt-pc-alm-research'
    names = list(test_protocol['source_sha256']) + [Path(__file__).name]
    save(out / 'protocol.json', dict(source_sha256={n: sha(root / 'work/experiments' / n) for n in names},
         support_summary_sha256=sha(folder / 'summary.json'), support_manifest_sha256=sha(folder / 'before_geometry_manifest.json'),
         geometry_test_summary_sha256=sha(testing / 'summary.json'), raw_manifest_sha256=sha(raw / 'before_evaluation_manifest.json'),
         design_sha256=sha(docs / '308_geometry_evaluation_protocol_v1.md'),
         seeds=SEEDS, selected_states=131, directions=393, methods=METHODS,
         query_targets_accessed=False, posterior_moments_accessed=False, new_confirmation=False,
         resources_matched=False, candidate_code_changed=False,
         global_lp_scope='offline common evaluator only, not candidate credit or initialization'))
    files, tasks, states, counts = {}, [], [], Counter()
    for seed in SEEDS:
        old = oldrows[seed]
        source = raw / old['file']
        assert sha(source) == old['sha256']
        with np.load(source, allow_pickle=False) as z:
            x, v = z['x_observed'], z['v_observed']
        old_positive = set(old['metadata']['positive_modes'])
        grouped_states, methods = [], {n: set() for n in METHODS}
        for key in sorted(k for k in by_state if k[0] == seed):
            common, state_methods, resources = read_state(folder, by_state[key])
            assert [F(float(t)) for t in x] == [F(t) for t in common['x']]
            assert [F(float(t)) for t in v] == [F(t) for t in common['v']]
            assert F(common['bound']) == classifier.BOUND and F(common['eps']) == classifier.EPS
            for name, modes in state_methods.items():
                methods[name].update(modes)
            grouped_states.append((key, state_methods, resources))
        all_modes = set().union(old_positive, *methods.values())
        classified = {}
        for mode in sorted(all_modes):
            result = classifier.classify_mode(x, v, mode)
            _, a, r, _, _ = classifier.matrices(x, v, mode)
            checks = certify_label(a, r, result)
            counts['certificate_rechecks'] += checks
            counts['mode_classifications'] += 1
            counts[result['classification']] += 1
            counts['LP_calls'] += result['lp_calls']
            counts['geometry_repairs'] += len(result['geometry_repairs'])
            classified[mode] = result
        # Prior positives are retained even if a new numerical check is unresolved.
        # A contradictory exact rejection is an audit error, not a pool deletion.
        for mode in old_positive:
            assert classified[mode]['classification'] not in ['infeasible', 'zero_volume_or_empty'], (seed, mode)
        original_unresolved = sorted(m for m in old_positive if not classified[m]['positive_volume_certified'])
        task_methods = {}
        for name in METHODS:
            task_methods[name] = describe(methods[name], classified, old_positive)
            task_methods[name]['retained_pool'] = sorted(old_positive | set(task_methods[name]['positive_modes']))
        for key, state_methods, resources in grouped_states:
            states.append(dict(seed=seed, location=list(key[1:]), resources=resources,
                               methods={name: describe(modes, classified, old_positive) for name, modes in state_methods.items()},
                               comparisons=comparisons(state_methods, classified, old_positive)))
        record = dict(seed=seed, selected_states=len(grouped_states), original_positive_modes=sorted(old_positive),
            original_positive_recheck_unresolved=original_unresolved, methods=task_methods,
            comparisons=comparisons(methods, classified, old_positive),
            geometry=classified, x_observed=x.tolist(), v_observed=v.tolist(), source_sha256=old['sha256'])
        filename = f'{seed}_geometry.json'
        save(out / filename, record)
        files[filename] = sha(out / filename)
        tasks.append(dict(seed=seed, selected_states=len(grouped_states), file=filename, sha256=files[filename],
            comparisons=record['comparisons'], original_positive_recheck_unresolved=original_unresolved,
            methods=task_methods, classifications=dict(Counter(v['classification'] for v in classified.values()))))
        if (seed - SEEDS[0] + 1) % 8 == 0:
            print(dict(phase='geometry', tasks=seed - SEEDS[0] + 1, mode_classifications=counts['mode_classifications'],
                       classifications={k: counts[k] for k in ['positive_volume', 'infeasible', 'zero_volume_or_empty', 'unresolved']}), flush=True)
    assert len(tasks) == 64 and len(states) == 131
    aggregate = {}
    for method in METHODS:
        rows = [t['methods'][method] for t in tasks]
        aggregate[method] = dict(tasks_with_new_positive=sum(bool(r['new_positive_modes']) for r in rows),
            task_positive_pairs=sum(len(r['new_positive_modes']) for r in rows),
            known_new_positive_numeric_volume_sum=sum(r['known_new_positive_numeric_volume_sum'] for r in rows),
            task_unknown_mode_pairs=sum(len(r['unknown_modes']) for r in rows),
            positive_volume_missing_pairs=sum(len(r['positive_modes_without_numerical_volume']) for r in rows))
    save(out / 'tasks.json', tasks)
    save(out / 'states.json', states)
    save(out / 'aggregate.json', aggregate)
    for name in ['protocol.json', 'tasks.json', 'states.json', 'aggregate.json']:
        files[name] = sha(out / name)
    summary = dict(execution_passed=True, tasks=64, states=131, methods=35, counts=dict(counts),
        support_complete_directions=support_summary['counts']['complete_directions'],
        support_incomplete_directions=support_summary['counts']['incomplete_directions'],
        geometry_classification_complete=counts['unresolved'] == 0,
        original_positive_recheck_unresolved=sum(len(t['original_positive_recheck_unresolved']) for t in tasks),
        dual_tasks_with_extra_positive_over_grid=sum(bool(t['comparisons']['dual_new_positive_missed_by_own_seven']) for t in tasks),
        dual_tasks_exclusive_vs_directions_and_controls=sum(bool(t['comparisons']['dual_new_positive_exclusive_vs_directions_and_controls']) for t in tasks),
        query_targets_accessed=False, posterior_moments_accessed=False, resources_matched=False,
        independent_task_gain_established=False, seconds=time.perf_counter() - begin, outputs_sha256=files)
    save(out / 'summary.json', summary)
    print({k: v for k, v in summary.items() if k != 'outputs_sha256'}, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(root, args.out)
    except Exception:
        save(args.out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
