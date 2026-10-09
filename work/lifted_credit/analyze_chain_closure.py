"""Chain closure summary (2026-10): selected self-iterating learner vs every comparator on the rebuilt NLP/Graph bases,
two seeds (449003: r19 + r17_b baselines; 449004: r20), identical evaluation sequences within a seed.
Writes results/lifted_credit/chain_closure_summary.json and prints a markdown table."""
import json
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[2] / 'results' / 'lifted_credit'
rng = np.random.default_rng(0)
SEEDS = {'449003': ('r19_{dom}_select_s3', 'r17_{dom}_rebuilt_s3_b'), '449004': ('r20_{dom}_select_s4', None)}
METHODS = ['select_cv', 'ttt_pcalm_prox', 'ttt_gd_soft', 'ttt_gd_soft_k45', 'ttt_gd_full', 'ttt_gd_full_k45', 'ttt_gd_official', 'ttt_quad_ridge', 'transformer']


def load(run):
    p = R / run / 'results.json'
    return json.loads(p.read_text())['results'] if p.exists() else {}


def per_seq(J, m, k):
    return np.concatenate([np.array(a['per_seq']) for a in J[m]['eval_test'][k]['per_attribute'].values()])


out = {}
for dom in ('nlp', 'graph'):
    for seed, (main, extra) in SEEDS.items():
        J = load(main.format(dom=dom))
        if extra:
            J = {**load(extra.format(dom=dom)), **J}
        if 'select_cv' not in J:
            continue
        cell = {}
        for k in ('2.0', '4.0', '8.0'):
            sel = per_seq(J, 'select_cv', k); row = {'select_cv': dict(mean=float(sel.mean()), median=float(np.median(sel)))}
            for m in METHODS[1:]:
                if m not in J:
                    continue
                b = per_seq(J, m, k); dl = sel - b
                bs = np.array([dl[rng.integers(0, len(dl), len(dl))].mean() for _ in range(4000)])
                row[m] = dict(mean=float(b.mean()), median=float(np.median(b)), diff=float(dl.mean()),
                              ci95=[float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], select_better_frac=float(np.mean(dl < 0)))
            cell[k] = row
        cell['fraction_choosing_pcalm'] = J['select_cv'].get('fraction_choosing_pcalm')
        out[f'{dom}_{seed}'] = cell

(R / 'chain_closure_summary.json').write_text(json.dumps(out, indent=1))
for key, cell in out.items():
    print(f'\n### {key}  (chose PC-ALM: {cell["fraction_choosing_pcalm"]})')
    print('| method | 2d | 4d | 8d |\n|---|---|---|---|')
    for m in METHODS:
        if m not in cell['2.0']:
            continue
        cells = []
        for k in ('2.0', '4.0', '8.0'):
            r = cell[k][m]
            cells.append(f"{r['mean']:.3f}" if m == 'select_cv' else f"{r['mean']:.3f} ({r['diff']:+.3f} [{r['ci95'][0]:+.3f},{r['ci95'][1]:+.3f}])")
        print(f'| {m} | ' + ' | '.join(cells) + ' |')
