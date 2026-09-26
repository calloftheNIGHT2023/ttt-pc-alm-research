"""Paired comparison on identical eval sequences (report 450). usage: paired_450.py runA:modelA runB:modelB [...]
Per-sequence NMSE pooled over held-out attributes; paired bootstrap CI of mean difference (A - B), share of sequences
where A is better, and the Wilcoxon signed-rank p-value."""
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

R = Path('results/lifted_credit')


def per_seq(spec):
    run, model = spec.split(':')
    J = json.loads((R / run / 'results.json').read_text())
    if model.startswith('ensemble/'):                       # CV multi-chain evaluation (report 450)
        r = dict(eval_test_images=J['ensemble'][model.split('/', 1)[1]])
    else:
        r = J['results'][model] if 'results' in J else J[model]
    ev = r.get('eval_test') or r.get('eval_test_images')
    out = {}
    for k, v in ev.items():
        if 'per_attribute' in v:
            out[k] = np.concatenate([np.array(a['per_seq']) for a in v['per_attribute'].values()])
        else:
            out[k] = np.array(v['per_seq'])
    return out


pairs = sys.argv[1:]
rng = np.random.default_rng(0)
for i in range(0, len(pairs), 2):
    A, B = per_seq(pairs[i]), per_seq(pairs[i + 1])
    print(f'\n{pairs[i]}  vs  {pairs[i + 1]}')
    for k in A:
        a, b = A[k], B[k]; dlt = a - b
        bs = np.array([dlt[rng.integers(0, len(dlt), len(dlt))].mean() for _ in range(4000)])
        p = wilcoxon(a, b).pvalue
        print(f'  n={k}: mean {a.mean():.3f} vs {b.mean():.3f}  diff {dlt.mean():+.4f} CI95 [{np.quantile(bs, .025):+.4f}, {np.quantile(bs, .975):+.4f}]'
              f'  A better {np.mean(dlt < 0):.2f}  wilcoxon p={p:.2g}  (N={len(a)})')
