"""Tables for report 450: noise-calibrated TTT x PC-ALM vs the 449 baselines (identical eval sequences, same seeds)."""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else 'results/lifted_credit')


def load(run):
    p = R / run / 'results.json'
    return json.loads(p.read_text()) if p.exists() else None


def attr_table(title, runs):
    print(f'\n### {title}\n\n| model (run) | n=2d | n=4d | n=8d |\n|---|---|---|---|')
    for run in runs:
        J = load(run)
        if J is None:
            continue
        for name, r in J['results'].items():
            ev = r['eval_test']
            cells = ' | '.join(f"{ev[k]['pooled_nmse_mean']:.3f} ({ev[k]['pooled_nmse_median']:.3f})" for k in ('2.0', '4.0', '8.0') if k in ev)
            ls = r.get('learned_scalars')
            extra = '' if not ls else '  ' + ', '.join(f'{k[4:]}={v:.3g}' for k, v in ls.items() if k in ('log_c', 'log_g', 'log_sr', 'log_mu_pc'))
            print(f'| {name} ({run}){extra} | {cells} |')


def cv_table(runs):
    print('\n### CV (CIFAR-10 test images, mean (median) NMSE)\n\n| model (run) | 2m | 3m | 4m | 6m |\n|---|---|---|---|---|')
    for run in runs:
        J = load(run)
        if J is None:
            continue
        for name, r in J.items():
            if not isinstance(r, dict) or 'eval_test_images' not in r:
                continue
            ev = r['eval_test_images']
            print(f'| {name} ({run}) | ' + ' | '.join(f"{ev[k]['nmse_mean']:.3f} ({ev[k]['nmse_median']:.3f})" for k in ('2.0', '3.0', '4.0', '6.0') if k in ev) + ' |')


attr_table('NLP (MRC + GloVe, held-out attributes)', ['r3_nlp_glove', 'r2_nlp_glove', 'r_nlp_glove'])
attr_table('Graph (QM9 + frozen GINE, held-out attributes)', ['r3_graph_qm9', 'r2_graph_qm9', 'r_graph_qm9'])
for s in ('0', '0.5', '1'):
    attr_table(f'Synthetic, noise inside |.| sigma={s}', [f'r3_synth_s{s}'])
cv_table(['r3_cv_pca_fixed', 'r3_cv_pcainit_a', 'r3_cv_pcainit_b', 'r_cv_pr', 'r2_cv_pca_inner'])
