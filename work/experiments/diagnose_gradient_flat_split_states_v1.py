"""289: old64 support-only census; no query arrays, no new adaptation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import batched_bp_discovery as bp


EPS = .001
CHANGE = 1e-10
METHODS = {
    'alm': ('certificate_activity_attribution', 'credit_control_probe33'),
    'archive': ('probe_continuation_credit', 'probe_archive_only'),
    'nodual': ('probe_continuation_credit', 'probe_then_nodual33'),
    'bp': ('probe_continuation_credit', 'probe_then_adam240_33'),
}
COMMON = ['x_observed', 'v_observed', 'initial_b', 'initial_h', 'initial_u',
          'effective_initial_u', 'initial_best', 'atomic_trial_b',
          'atomic_trial_origins', 'origins', 'assigned_actions']
LOCAL = ['prefix_b', 'prefix_h', 'prefix_u', 'anchor_b', 'anchor_h', 'anchor_u']


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def tent(z):
    # Same floating-point operation order as the frozen model, not a new model.
    return np.maximum(0., 1. - np.abs(2. * z - 1.))


def independent_reverse(bank, x, v):
    h = np.broadcast_to(x, (len(bank), len(x)))
    slopes = []
    for j in range(bank.shape[1]):
        z = h + bank[:, j, None]
        slopes.append(np.where((z > 0) & (z < .5), 2.,
                      np.where((z > .5) & (z < 1), -2., 0.)))
        h = tent(z)
    raw = h - v
    residual = np.sign(raw) * np.maximum(np.abs(raw) - EPS, 0.)
    credit = residual.copy()
    gradient = np.empty_like(bank)
    for j in reversed(range(bank.shape[1])):
        credit = credit * slopes[j]
        gradient[:, j] = credit.mean(axis=1)
    return gradient, raw


def rational_state(b, x, v):
    """Exact-real evaluation of the stored binary64 inputs, not binary64 replay."""
    biases = [F(float(t)) for t in b]
    epsilon = F(EPS)
    gradient = [F(0) for _ in biases]
    loss = F(0)
    max_raw = F(0)
    knot_count = 0
    feasible = True
    for xx, vv in zip(x, v):
        h = F(float(xx))
        slopes = []
        for bias in biases:
            z = h + bias
            knot_count += int(z in (F(0), F(1, 2), F(1)))
            slopes.append(2 if 0 < z < F(1, 2) else -2 if F(1, 2) < z < 1 else 0)
            h = max(F(0), min(2 * z, 2 - 2 * z))
        raw = h - F(float(vv))
        max_raw = max(max_raw, abs(raw))
        excess = max(abs(raw) - epsilon, F(0))
        feasible = feasible and excess == 0
        residual = excess if raw >= 0 else -excess
        loss += residual * residual / (2 * len(x))
        credit = residual
        for j in reversed(range(len(biases))):
            credit *= slopes[j]
            gradient[j] += credit / len(x)
    return dict(exact_band_feasible=bool(feasible), exact_gradient_zero=all(t == 0 for t in gradient),
                exact_gradient=[float(t) for t in gradient], exact_loss=float(loss),
                exact_max_raw=float(max_raw), exact_knot_count=knot_count)


def residuals(b, h, x):
    # b (..., restart, depth); h (..., depth, restart, support)
    return np.stack([h[..., j, :, :] - tent((x if j == 0 else h[..., j-1, :, :])
                    + b[..., :, j, None]) for j in range(b.shape[-1])], axis=-3)


def mode_list(bank, x):
    bank = np.asarray(bank).reshape(-1, 4)
    h = np.broadcast_to(x, (len(bank), len(x)))
    parts = []
    for j in range(4):
        z = h + bank[:, j, None]
        parts.append((z >= 0).astype(np.uint8) + (z >= .5).astype(np.uint8)
                     + (z >= 1).astype(np.uint8))
        h = tent(z)
    return [row.tobytes().hex() for row in np.concatenate(parts, axis=1)]


def first_over(values):
    indices = np.flatnonzero(np.asarray(values) > CHANGE)
    return int(indices[0]) if len(indices) else None


def selftests():
    # Interior nonzero gradient, flat saturated loss, fold-convention zero, and
    # exactly feasible loss. No old or new task query targets are involved.
    cases = [
        (np.zeros(4), np.array([.1]), np.array([.41]), False, False),
        (np.array([-.12, 0, 0, 0]), np.array([.01]), np.array([.5]), True, False),
        (np.zeros(4), np.array([0.]), np.array([.5]), True, False),
        (np.zeros(4), np.array([.125]), np.array([0.]), True, True),
    ]
    maximum = 0.
    for b, x, v, zero, feasible in cases:
        exact = rational_state(b, x, v)
        assert exact['exact_gradient_zero'] == zero
        assert exact['exact_band_feasible'] == feasible
        _, error, jac, _ = bp.evaluate(b[None], x, v)
        gradient = np.einsum('rni,rn->ri', jac, error, optimize=False) / len(x)
        reverse, _ = independent_reverse(b[None], x, v)
        np.testing.assert_allclose(gradient, reverse, rtol=0, atol=1e-12)
        maximum = max(maximum, float(np.max(abs(gradient[0] - exact['exact_gradient']))))
        if zero:
            m = np.zeros_like(b)
            second = np.zeros_like(b)
            current = b.copy()
            for it in range(1, 481):
                grad, _ = independent_reverse(current[None], x, v)
                m = .9*m + .1*grad[0]
                second = .999*second + .001*grad[0]**2
                current = np.clip(current - .003*(m/(1-.9**it)) /
                                  (np.sqrt(second/(1-.999**it))+1e-8), -.12, .12)
                assert np.array_equal(current, b)
    assert maximum < 1e-12
    assert mode_list(np.zeros((1, 4)), np.array([0., .5, 1.]))[0][:6] == '010203'
    return dict(passed=True, analytic_cases=len(cases), stationary_adam_checks=3*480,
                maximum_exact_float_gradient_gap=maximum, query_arrays_decoded=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    out = root / 'results/gradient_flat_split_states/development_v1'
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    tests = selftests()
    save(out/'selftests.json', tests)
    sources = ['work/experiments/diagnose_gradient_flat_split_states_v1.py',
               'work/experiments/batched_bp_discovery.py',
               'work/experiments/local_branch_memory.py',
               'work/experiments/cold_stagnation_switch.py',
               'work/experiments/probe_continuation_credit.py',
               'work/experiments/certificate_activity_attribution.py',
               'outputs/ttt-pc-alm-research/289_gradient_flat_split_state_protocol.md']
    index = {}
    parents = {}
    for group in sorted({p[0] for p in METHODS.values()}):
        directory = root/'results'/group/'development'
        summary = read(directory/'summary.json')
        manifest = read(directory/'before_evaluation_manifest.json')
        assert summary['passed'] and summary['tasks'] == 64 and summary['failures'] == 0
        assert summary['before_evaluation_manifest_sha256'] == sha(directory/'before_evaluation_manifest.json')
        assert manifest['rows_sha256'] == sha(directory/'rows.json')
        assert manifest['protocol_sha256'] == sha(directory/'protocol.json')
        assert manifest['phase_accesses_query_targets'] is False
        rows = read(directory/'rows.json')
        lookup = {(r['seed'], r['method']): r for r in rows}
        assert len(lookup) == len(rows)
        index[group] = (directory, lookup, manifest)
        parents[group] = {name: sha(directory/name) for name in
                          ['summary.json', 'protocol.json', 'rows.json', 'before_evaluation_manifest.json']}
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope='OLD64 support-only mechanism census',
                    seeds=list(range(5910000, 5910064)), methods=METHODS,
                    query_arrays_decoded=False, query_targets_accessed=False,
                    prior_query_results_exist=True, significant_state_change_threshold=CHANGE,
                    exact_reference='Fractions of stored binary64 inputs; zero derivative convention at knots',
                    parents=parents, source_sha256={name: sha(root/name) for name in sources})
    save(out/'protocol.json', protocol)
    totals = Counter()
    max_gaps = Counter()
    tasks = []
    origins = []
    input_hashes = {}
    for seed in protocol['seeds']:
        arrays = {}
        metadata = {}
        for role, (group, name) in METHODS.items():
            directory, lookup, manifest = index[group]
            row = lookup[seed, name]
            path = (directory/row['file']).resolve()
            assert path.is_relative_to(directory.resolve())
            digest = sha(path)
            assert digest == row['sha256'] == manifest['prediction_files'][row['file']]
            input_hashes[str(path.relative_to(root)).replace('\\', '/')] = digest
            fields = COMMON + (LOCAL if role in ['alm', 'nodual'] else ['bp_b', 'bp_roles'] if role == 'bp' else [])
            with np.load(path, allow_pickle=False) as z:
                arrays[role] = {key: z[key].copy() for key in fields}
            metadata[role] = row['metadata']
            assert metadata[role]['trace_enabled']
        a, d, b, p = [arrays[k] for k in ['alm', 'nodual', 'bp', 'archive']]
        for role in ['nodual', 'bp', 'archive']:
            for key in COMMON:
                assert np.array_equal(a[key], arrays[role][key]), (seed, role, key)
                totals['shared_arrays'] += 1
        x, v = a['x_observed'], a['v_observed']
        initial = a['initial_b']
        assert initial.shape == (33, 4) and np.max(abs(initial)) <= .12
        assert not a['initial_u'].any() and not a['effective_initial_u'].any()
        assert np.array_equal(b['bp_b'][0], initial) and b['bp_b'].shape == (241, 33, 4)
        assert b['bp_roles'].tolist() == [True]*240 + [False]
        for state in [a, d]:
            for key in ['b', 'h', 'u']:
                assert np.array_equal(state['prefix_'+key][0], state['initial_'+key])
                expected = state['prefix_'+key][-1, :1] if key == 'b' else state['prefix_'+key][-1, :, :1]
                assert np.array_equal(state['anchor_'+key][0], expected)
            assert state['prefix_b'].shape == (33, 33, 4)
            assert state['anchor_b'].shape == (33, 1, 4)
        for key in ['b', 'h']:
            assert np.array_equal(a['prefix_'+key][1], d['prefix_'+key][1]), (seed, key)
            totals['first_primal_step_identities'] += 1
        assert not d['prefix_u'].any() and not d['anchor_u'].any()
        for phase in ['prefix', 'anchor']:
            rr = residuals(a[phase+'_b'][1:], a[phase+'_h'][1:], x)
            gap = float(np.max(abs(a[phase+'_u'][1:] - (a[phase+'_u'][:-1] + .5*rr))))
            assert gap <= 1e-14, (seed, phase, gap)
            max_gaps['dual_recurrence'] = max(max_gaps['dual_recurrence'], gap)
            totals['dual_recurrence_scalar_checks'] += rr.size
        _, error, jac, raw = bp.evaluate(initial, x, v)
        gradient = np.einsum('rni,rn->ri', jac, error, optimize=False) / len(x)
        reverse, reverse_raw = independent_reverse(initial, x, v)
        np.testing.assert_array_equal(raw, reverse_raw)
        np.testing.assert_allclose(gradient, reverse, rtol=0, atol=1e-12)
        max_gaps['forward_reverse_gradient'] = max(max_gaps['forward_reverse_gradient'], float(np.max(abs(gradient-reverse))))
        split = np.max(abs(residuals(initial, a['initial_h'], x)), axis=(0, 2))
        positive = {role: set(m['positive_modes']) for role, m in metadata.items()}
        for role, state in arrays.items():
            banks = [state['atomic_trial_b'], state['initial_b']]
            if role in ['alm', 'nodual']:
                banks += [state['prefix_b'].reshape(-1, 4), state['anchor_b'].reshape(-1, 4)]
            elif role == 'bp':
                banks += [state['bp_b'].reshape(-1, 4)]
            visited = set(mode_list(np.concatenate(banks), x))
            assert visited == set(metadata[role]['visited_modes']), (seed, role, 'visited')
            assert positive[role] <= visited
            assert positive['archive'] <= positive[role]
            totals['complete_visited_mode_reconstructions'] += 1
        sets = {role+'_new_vs_archive': positive[role]-positive['archive'] for role in ['alm', 'nodual', 'bp']}
        sets.update(alm_only_vs_bp=positive['alm']-positive['bp'],
                    bp_only_vs_alm=positive['bp']-positive['alm'],
                    alm_only_vs_nodual=positive['alm']-positive['nodual'],
                    alm_only_vs_both=positive['alm']-(positive['bp']|positive['nodual']))
        reachable_zero = {role: set() for role in ['alm', 'nodual', 'bp']}
        task_counts = Counter()
        for origin in range(33):
            rational = rational_state(initial[origin], x, v)
            exact_grad = np.array(rational['exact_gradient'])
            max_gaps['rational_float_gradient'] = max(max_gaps['rational_float_gradient'], float(np.max(abs(exact_grad-gradient[origin]))))
            zero = bool(np.all(gradient[origin] == 0))
            float_feasible = bool(np.all(abs(raw[origin]) <= EPS))
            traces = {}
            for role, state in [('alm', a), ('nodual', d)]:
                traces[role] = state['prefix_b'][:, origin]
                if origin == 0:
                    traces[role] = np.concatenate([traces[role], state['anchor_b'][1:, 0]])
            traces['bp'] = b['bp_b'][:, origin]
            movements = {role: np.max(abs(trace-initial[origin]), axis=1) for role, trace in traces.items()}
            stationary = np.array_equal(traces['bp'], np.broadcast_to(initial[origin], traces['bp'].shape))
            if zero:
                assert stationary, (seed, origin, 'zero gradient but BP moved')
            reachable = {role: set(mode_list(trace, x)) & positive[role] for role, trace in traces.items()}
            if zero:
                for role in reachable_zero:
                    reachable_zero[role].update(reachable[role])
            divergence = np.max(abs(traces['alm']-traces['nodual']), axis=1)
            record = dict(seed=seed, origin=origin, **rational,
                          float_gradient_zero=zero, float_band_feasible=float_feasible,
                          float_gradient_linf=float(np.max(abs(gradient[origin]))),
                          split_residual_linf=float(split[origin]), bp_trace_bitwise_stationary=bool(stationary),
                          maximum_parameter_change={role: float(vals.max()) for role, vals in movements.items()},
                          first_parameter_change={role: first_over(vals) for role, vals in movements.items()},
                          first_alm_nodual_parameter_divergence=first_over(divergence),
                          maximum_alm_nodual_parameter_divergence=float(divergence.max()),
                          positive_modes_reachable={role: sorted(keys) for role, keys in reachable.items()})
            origins.append(record)
            flags = dict(float_gradient_zero=zero, exact_gradient_zero=rational['exact_gradient_zero'],
                         float_band_feasible=float_feasible, exact_band_feasible=rational['exact_band_feasible'],
                         exact_knot_present=rational['exact_knot_count'] > 0,
                         float_exact_zero_disagreement=zero != rational['exact_gradient_zero'],
                         float_exact_band_disagreement=float_feasible != rational['exact_band_feasible'],
                         zero_gradient_split_residual=zero and split[origin] > CHANGE,
                         zero_gradient_alm_moved=zero and movements['alm'].max() > CHANGE,
                         zero_gradient_nodual_moved=zero and movements['nodual'].max() > CHANGE,
                         zero_gradient_alm_nodual_diverged=zero and divergence.max() > CHANGE,
                         zero_gradient_reached_alm_new=zero and bool(reachable['alm'] & sets['alm_new_vs_archive']),
                         zero_gradient_reached_alm_only_vs_both=zero and bool(reachable['alm'] & sets['alm_only_vs_both']))
            for name, flag in flags.items():
                task_counts[name] += int(flag)
        for role in reachable_zero:
            sets[role+'_new_reachable_from_zero'] = reachable_zero[role] - positive['archive']
        sets['alm_only_vs_both_reachable_from_zero'] = reachable_zero['alm'] & sets['alm_only_vs_both']
        task = dict(seed=seed, counts=dict(task_counts), positive_modes={role: sorted(value) for role, value in positive.items()},
                    sets={key: sorted(value) for key, value in sets.items()})
        tasks.append(task)
        totals.update(task_counts)
        totals.update({'task_has_'+name: int(bool(value)) for name, value in sets.items()})
        totals.update({'task_mode_sum_'+name: len(value) for name, value in sets.items()})
        if (seed-5910000+1) % 8 == 0:
            print(json.dumps(dict(tasks=len(tasks), total=64, seconds=time.perf_counter()-started)), flush=True)
    save(out/'origins.json', origins)
    save(out/'tasks.json', tasks)
    save(out/'input_hashes.json', input_hashes)
    assert len(origins) == 2112 and len(tasks) == 64 and len(input_hashes) == 256
    summary = dict(passed=True, meaning='Census/invariants passed, NOT an advantage claim',
                   tasks=64, origins=2112, predictor_trace_files=256, counts=dict(totals),
                   maximum_gaps=dict(max_gaps), query_arrays_decoded=False, query_targets_accessed=False,
                   adaptation_rerun=False, resource_scope='offline diagnostic cost; not charged as online method cost',
                   seconds=time.perf_counter()-started, protocol_sha256=sha(out/'protocol.json'),
                   outputs_sha256={name: sha(out/name) for name in ['selftests.json', 'origins.json', 'tasks.json', 'input_hashes.json']})
    save(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
