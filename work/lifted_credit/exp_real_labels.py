"""Exp C2 (report 440, P3 on real labels): standard few-shot binary tasks with REAL labels.

Each task: two random classes of the real dataset, labels +-0.9, n = alpha*d support, real queries.
The relation is (near-)monotone in a feature direction, so the theory predicts NO advantage for
lifted credit and a strong closed-form head. Model link for model-based methods: tanh(2z).
Reported: normalised query MSE against the real +-0.9 labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import common as C
import suite as S

ap = argparse.ArgumentParser()
ap.add_argument('--features', required=True)
ap.add_argument('--dims', type=int, nargs='+', default=[32, 128])
ap.add_argument('--alpha', type=float, default=1.0)
ap.add_argument('--episodes', type=int, default=64)
ap.add_argument('--nq', type=int, default=400)
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--seed', type=int, default=440501)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.Tanh
Z = np.load(args.features); F, lab = Z['X'], Z['label']
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(F)); half = len(F) // 2
fit, pool, plab = F[perm[:half]], F[perm[half:]], lab[perm[half:]]
mu = fit.mean(0); evals, evecs = np.linalg.eigh(np.cov(fit - mu, rowvar=False)); order = np.argsort(evals)[::-1]
classes = np.unique(plab)
rows = []
for d in args.dims:
    Zp = (pool - mu) @ (evecs[:, order[:d]] / np.sqrt(evals[order[:d]]))
    n = int(round(args.alpha * d)); E = args.episodes
    Xs, ys, Xqs, yqs = [], [], [], []
    for e in range(E):
        a, b = rng.choice(classes, 2, replace=False)
        ia, ib = np.flatnonzero(plab == a), np.flatnonzero(plab == b)
        sel = np.concatenate([rng.choice(ia, (n + args.nq) // 2 + 1, replace=False), rng.choice(ib, (n + args.nq) // 2 + 1, replace=False)])
        rng.shuffle(sel); sel = sel[:n + args.nq]
        yy = np.where(plab[sel] == a, 0.9, -0.9)
        Xs.append(Zp[sel[:n]]); ys.append(yy[:n]); Xqs.append(Zp[sel[n:]]); yqs.append(yy[n:])
    T = lambda L: torch.tensor(np.stack(L), dtype=dt, device=args.device)
    X, y, Xq, yq = T(Xs), T(ys), T(Xqs), T(yqs)
    res, tm = S.run_suite(link, X, y, Xq, yq, None, T=args.T, R=8, bp_big=64, sigma=0.1, seed=args.seed + d)
    rs = S.summarise(res, dict(features=Path(args.features).stem, d=d, n=n, task='real_label_binary'))
    rows += rs
    print(f'{Path(args.features).stem} real-label binary d={d} n={n}', flush=True)
    for r in rs:
        print(f"  {r['method']:24s} nmse={r['nmse']:.4f}±{r['nmse_se']:.4f}", flush=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
