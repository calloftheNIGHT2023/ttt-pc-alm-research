"""Independent scalar reconstruction of 322/323 for the noon delivery.

Does not import the candidate selector or either evaluation script. Reuses the
already audited per-region Monte Carlo moments, not query answers. Numerical
agreement is not an interval-certified risk bound or a fresh test experiment.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run(root, out):
    start = time.perf_counter()
    base = root / 'results/branch_image_chain'
    inputs = {}
    counts = Counter()
    max_gap = 0.

    def checked(p):
        inputs[str(p.relative_to(root))] = sha(p)
        return read(p)

    def close(a, b):
        nonlocal max_gap
        gap = abs(a - b)
        max_gap = max(max_gap, gap)
        assert gap < 2e-13, (a, b, gap)
        counts['numeric_fields'] += 1

    for folder in ['development_v1', 'geometry_v1', 'audit_v1',
                   'geometry_audit_v1', 'conditional_risk_v1', 'opportunity_v1']:
        summary = checked(base / folder / 'summary.json')
        assert summary['passed']
        for name, digest in summary.get('outputs_sha256', {}).items():
            assert sha(base / folder / name) == digest
            counts['sealed_file_hashes'] += 1
    moments = root / 'results/confirmation_conditional_risk/moments'
    index = {t['seed']: t for t in checked(moments / 'tasks.json')}
    records = defaultdict(list)
    for row in checked(base / 'development_v1/rows.json'):
        p = base / 'development_v1' / row['file']
        assert sha(p) == row['sha256']
        records[row['seed']].append(checked(p))
    scores = checked(base / 'conditional_risk_v1/scores.json')
    scoremap = {(s['seed'], s['channel'], s['grid']): s for s in scores}
    targets = {(t['seed'], t['channel'], t['mode']): t
               for t in checked(base / 'opportunity_v1/targets.json')}
    tasks = {t['seed']: t for t in checked(base / 'opportunity_v1/tasks.json')}
    channels = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
    categories = ['no_trigger', 'outside_window', 'ranked_out', 'selected']
    tally = {(n, c): [] for n in channels for c in categories}
    risk_rows = []
    for row in checked(base / 'geometry_v1/tasks.json'):
        seed = row['seed']
        g = checked(base / 'geometry_v1' / row['file'])
        mt = index[seed]
        p = moments / mt['file']
        assert sha(p) == mt['sha256']
        inputs[str(p.relative_to(root))] = mt['sha256']
        with np.load(p, allow_pickle=False) as z:
            vol, means = z['volumes'], z['means']
        keys = mt['keys']
        total = math.fsum(float(v) for v in vol)
        weights = {k: float(v) / total for k, v in zip(keys, vol)}
        prior = set(g['prior_positive_modes'])

        def mixture(pool):
            ids = [i for i, k in enumerate(keys) if k in pool]
            mass = math.fsum(weights[keys[i]] for i in ids)
            arr = np.array([[math.fsum(weights[keys[i]] * float(means[i, b, q])
                            for i in ids) / mass for q in range(257)] for b in range(4)])
            return mass, arr

        _, full = mixture(set(keys))
        oldmass, old = mixture(prior)
        close(oldmass, tasks[seed]['prior_mass'])
        assert tasks[seed]['triggers'] == len(records[seed])
        baseline = {}
        for grid, indices in [(257, list(range(257))), (129, list(range(0, 257, 2)))]:
            baseline[grid] = [math.fsum(float(old[a, q] - full[a, q]) *
                              float(old[b, q] - full[b, q]) for q in indices) / grid
                              for a, b in [(0, 1), (2, 3)]]
            for got, expected in zip(baseline[grid], tasks[seed]['pool_excess_pairs'][str(grid)]):
                close(got, expected)
        risk_rows.append((bool(records[seed]), baseline))
        for name in channels:
            positive = set(g['methods'][name]['positive_modes'])
            added = positive - prior
            newmass, new = mixture(prior | positive) if added else (oldmass, old)
            for mode in set(keys) - prior:
                selected = mode in positive
                in_window = False
                for rec in records[seed]:
                    distance = sum(a != b for a, b in zip(bytes.fromhex(mode), bytes.fromhex(rec['original_mode'])))
                    lower = rec['results'][name]['minimum_nonexcluded_hamming']
                    if lower is not None and lower <= distance <= lower + 2:
                        in_window = True
                category = ('selected' if selected else 'no_trigger' if not records[seed]
                            else 'ranked_out' if in_window else 'outside_window')
                target = targets[seed, name, mode]
                assert target['category'] == category
                close(weights[mode], target['posterior_mass'])
                tally[name, category].append(weights[mode])
                counts['coverage_labels'] += 1
            for grid, indices in [(257, list(range(257))), (129, list(range(0, 257, 2)))]:
                s = scoremap[seed, name, grid]
                assert sorted(added) == s['new_modes']
                close(newmass, s['new_mass'])
                deltas = []
                for j, (a, b) in enumerate([(0, 1), (2, 3)]):
                    after = math.fsum(float(new[a, q] - full[a, q]) *
                                     float(new[b, q] - full[b, q]) for q in indices) / grid
                    delta = after - baseline[grid][j]
                    close(after, s['pairs'][j]['new_excess'])
                    close(delta, s['pairs'][j]['delta'])
                    deltas.append(delta)
                close(math.fsum(deltas) / 2, s['delta'])
                counts['risk_rows'] += 1
        counts['tasks'] += 1
    agg = checked(base / 'opportunity_v1/aggregate.json')
    for (name, category), ws in tally.items():
        assert len(ws) == agg[name][category]['mode_pairs']
        close(math.fsum(ws) / 64, agg[name][category]['mean_mass'])
    opportunity = checked(base / 'opportunity_v1/risk_opportunity.json')
    for grid in [257, 129]:
        for label, flag in [('all', None), ('no_trigger', False), ('with_trigger', True)]:
            chosen = [v[grid] for triggered, v in risk_rows if flag is None or triggered == flag]
            entry = opportunity[str(grid)][label]
            assert len(chosen) == entry['tasks']
            pairs = [math.fsum(r[j] for r in chosen) / 64 for j in range(2)]
            for a, b in zip(pairs, entry['pair_contributions']):
                close(a, b)
            close(math.fsum(pairs) / 2, entry['mean_contribution'])
    for w in checked(base / 'opportunity_v1/ranking_witnesses.json'):
        assert Fraction(w['lower']) <= 0
        if not w['selected_here']:
            assert (Fraction(w['lower']), w['mode']) > (Fraction(w['last_retained_lower']), w['last_retained_mode'])
        counts['exact_rank_witness_inequalities'] += 1
    ratios = {grid: opportunity[grid]['no_trigger']['mean_contribution'] /
              opportunity[grid]['all']['mean_contribution'] for grid in ['257', '129']}
    result = dict(passed=True, counts=dict(counts), max_numeric_gap=max_gap,
                  no_trigger_risk_fraction=ratios, seconds=time.perf_counter()-start,
                  source_sha256=sha(Path(__file__)), input_sha256=inputs,
                  query_targets_accessed=False, independent_task_gain_established=False,
                  scope='Independent scalar moment mixtures, risk rows, coverage categories and aggregate arithmetic; existing exact K-order and geometry audits reused.')
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print({k: v for k, v in result.items() if k != 'input_sha256'}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run(Path(__file__).resolve().parents[2], args.out)
