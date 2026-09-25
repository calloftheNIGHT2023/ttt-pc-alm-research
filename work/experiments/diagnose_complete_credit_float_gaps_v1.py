"""308 preserve/explain fixed-threshold float discrepancies; no tolerance change."""
from fractions import Fraction as F
import argparse
import gzip
import json
from pathlib import Path
import traceback

import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from dual_amplitude_local_audit_v1 import scalar_step
from complete_credit_rational_reference_v1 import direct_step, tent
from exact_quadratic_events_v1 import Poly
from evaluate_complete_credit_mode_geometry_v1 import load_support, read, save, sha


def bias_candidates(previous, target, old):
    """Read-only replay of the frozen float bias solver, including all costs."""
    previous, target = previous[None], target[None]
    old = np.array([old])
    base = cold.base
    n = previous.shape[1]
    reg = np.searchsorted(base.KNOTS, previous - base.BOUND, side='right')
    s0 = base.SLOPES[reg]
    c0 = s0 * previous + base.INTERCEPTS[reg]
    ini = [np.sum(s0 * s0, axis=1), np.sum(s0 * (target - c0), axis=1), np.sum((c0 - target)**2, axis=1)]
    events = (base.KNOTS[None, :, None] - previous[:, None, :]).reshape(1, -1)
    sb, sa = base.SLOPES[:-1][None, :, None], base.SLOPES[1:][None, :, None]
    cb = sb * previous[:, None, :] + base.INTERCEPTS[:-1][None, :, None]
    ca = sa * previous[:, None, :] + base.INTERCEPTS[1:][None, :, None]
    delta = [np.broadcast_to(sa**2 - sb**2, (1, 3, n)).reshape(1, -1),
             (sa * (target[:, None, :] - ca) - sb * (target[:, None, :] - cb)).reshape(1, -1),
             ((ca - target[:, None, :])**2 - (cb - target[:, None, :])**2).reshape(1, -1)]
    valid = (events > -base.BOUND) & (events < base.BOUND)
    positions = np.clip(events, -base.BOUND, base.BOUND)
    order = np.argsort(positions, axis=1, kind='stable')
    positions = np.take_along_axis(positions, order, axis=1)
    coefs = []
    for initial, change in zip(ini, delta):
        ordered = np.take_along_axis(np.where(valid, change, 0.), order, axis=1)
        coefs.append(np.concatenate([initial[:, None], initial[:, None] + np.cumsum(ordered, axis=1)], axis=1) / n)
    aa, bb, cc = coefs
    lo = np.concatenate([np.full((1, 1), -base.BOUND), positions], axis=1)
    hi = np.concatenate([positions, np.full((1, 1), base.BOUND)], axis=1)
    candidates = np.clip((bb + .01 * old[:, None]) / (np.maximum(aa, 0) + .01), lo, hi)
    costs = aa * candidates**2 - 2 * bb * candidates + cc
    costs += .01 * (candidates - old[:, None])**2
    winner = int(np.argmin(costs[0]))
    exact_costs = []
    for value in candidates[0]:
        q = F(float(value))
        cost = sum((tent(F(float(p)) + q) - F(float(t)))**2 for p, t in zip(previous[0], target[0])) / n
        cost += F(.01) * (q - F(float(old[0])))**2
        exact_costs.append(cost)
    exact_winner = min(range(len(exact_costs)), key=lambda i: exact_costs[i])
    return dict(candidates=candidates[0].tolist(), float_costs=costs[0].tolist(), exact_costs=exact_costs,
                float_winner=winner, exact_winner_among_float_candidates=exact_winner,
                exact_cost_excess=exact_costs[winner] - exact_costs[exact_winner],
                candidate_gap=abs(float(candidates[0, winner] - candidates[0, exact_winner])),
                equal_float_cost_at_two_winners=bool(costs[0, winner] == costs[0, exact_winner]))


def run(root, out):
    folder, summary, protocol, groups = load_support(root)
    selected = [r for r in read(folder / 'rows.json') if r['floating_value_discrepancies']]
    names = [Path(__file__).name, 'cold_stagnation_switch.py', 'streaming_branch_projection.py',
             'complete_credit_rational_reference_v1.py', 'dual_amplitude_local_audit_v1.py']
    save(out / 'protocol.json', dict(source_sha256={n: sha(root / 'work/experiments' / n) for n in names},
         support_summary_sha256=sha(folder / 'summary.json'), fixed_threshold=1e-10,
         selection='all saved seven-point value discrepancies above the unchanged fixed threshold',
         query_targets_accessed=False, update_or_tolerance_changed=False))
    records = []
    with discovery_box(.12):
        for row in selected:
            with gzip.open(folder / row['file'], 'rt', encoding='utf-8') as stream:
                payload = json.load(stream)
            data = payload['state']
            state = {k: [F(q) for q in data[k]] for k in ['b', 'x', 'v']}
            state.update({k: [[F(q) for q in layer] for layer in data[k]] for k in ['h', 'direction']})
            state.update({k: F(data[k]) for k in ['bound', 'trust', 'eps']})
            b0, h0, u, x, v = [np.array(state[k], dtype=float) for k in ['b', 'h', 'direction', 'x', 'v']]
            for saved in payload['seven_points']:
                if not saved['float_value_discrepancy']:
                    continue
                alpha = F(saved['alpha'])
                exact = direct_step(state, alpha)
                exact_b, exact_h = np.array(exact['b'], dtype=float), np.array(exact['h'], dtype=float)
                production = cold.Local(b0[None], x, v, 'alm')
                production.h = h0[:, None].copy()
                scaled_u = float(alpha) * u
                production.u = scaled_u[:, None].copy()
                production.step()
                pb, ph = production.b[0], production.h[:, 0]
                assert np.array_equal(pb, np.array(saved['float_b']))
                assert np.array_equal(ph, np.array(saved['float_h']))
                assert [Poly(q).at(alpha) for q in saved['exact_b']] == exact['b']
                scalar = scalar_step(b0, h0, u, x, v, float(alpha))
                diagnostics = []
                for j in np.flatnonzero(abs(pb - exact_b) > 1e-10):
                    previous = x if j == 0 else ph[j - 1]
                    target = ph[j] + scaled_u[j]
                    candidates = bias_candidates(previous, target, b0[j])
                    assert candidates['candidates'][candidates['float_winner']] == pb[j]
                    diagnostics.append(dict(layer=int(j), **candidates))
                records.append(dict(seed=row['seed'], location=row['location'], family=row['family'], alpha=alpha,
                    source_sha256=row['sha256'], exact_b=exact['b'], exact_h=exact['h'],
                    production_b=pb.tolist(), production_h=ph.tolist(),
                    scalar_b=scalar['b'].tolist(), scalar_h=scalar['h'].tolist(),
                    max_bias_gap=float(np.max(abs(pb - exact_b))), max_activity_gap=float(np.max(abs(ph - exact_h))),
                    max_scalar_gap=max(float(np.max(abs(scalar['b'] - exact_b))), float(np.max(abs(scalar['h'] - exact_h)))),
                    bias_cost_diagnostics=diagnostics, saved_modes_equal=not saved['float_mode_discrepancy']))
    assert len(records) == summary['counts']['floating_value_discrepancies']
    save(out / 'cases.json', records)
    result = dict(passed=True, cases=len(records), production_arrays_replayed_exactly=True,
        max_production_gap=max(max(r['max_bias_gap'], r['max_activity_gap']) for r in records),
        max_scalar_gap=max(r['max_scalar_gap'] for r in records),
        bias_candidate_cost_misorderings=sum(d['exact_cost_excess'] > 0 for r in records for d in r['bias_cost_diagnostics']),
        float_cost_ties_with_distinct_better_exact_candidate=sum(d['exact_cost_excess'] > 0 and d['equal_float_cost_at_two_winners'] for r in records for d in r['bias_cost_diagnostics']),
        activity_discrepancies_above_threshold=sum(r['max_activity_gap'] > 1e-10 for r in records),
        modes_unchanged=all(r['saved_modes_equal'] for r in records), threshold_changed=False,
        query_targets_accessed=False, outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    save(out / 'summary.json', result)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        run(Path(__file__).resolve().parents[2], args.out)
    except Exception:
        save(args.out / 'failure.json', dict(traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
