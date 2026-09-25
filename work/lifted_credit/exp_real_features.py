"""Exp C1 (report 440): the lifted-credit predictions on real representations from real small models.

Inputs x are real embeddings (NLP / CV / Graph), PCA-whitened to d dims on a disjoint half of the pool.
Per task: random direction u, y = g(u.x) + sigma*eps. The input distribution is real (non-Gaussian,
anisotropic tails); the Gaussian assumption behind the theory and behind the hybrid transform is dropped.
Normalised MSE = query MSE / query-target variance.
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
ap.add_argument('--link', default='he3')
ap.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128])
ap.add_argument('--alpha', type=float, default=4.0)
ap.add_argument('--episodes', type=int, default=64)
ap.add_argument('--nq', type=int, default=1000)
ap.add_argument('--sigma', type=float, default=0.1)
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--restarts', type=int, default=8)
ap.add_argument('--bp_big', type=int, default=64)
ap.add_argument('--seed', type=int, default=440301)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--skip', nargs='*', default=[])
ap.add_argument('--auto_grid', action='store_true', help='activity search range from support targets')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64
link = C.LINKS[args.link]
C.GRID_LIM = 8.0   # whitened real features have heavier tails than N(0,1)

F = np.load(args.features)['X']
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(F))
fitpart, pool = F[perm[:len(F) // 2]], F[perm[len(F) // 2:]]
mu = fitpart.mean(0)
cov = np.cov(fitpart - mu, rowvar=False)
evals, evecs = np.linalg.eigh(cov)
order = np.argsort(evals)[::-1]
rows = []
for d in args.dims:
    Wh = evecs[:, order[:d]] / np.sqrt(evals[order[:d]])          # PCA whitening fitted on disjoint half
    Z = torch.tensor((pool - mu) @ Wh, dtype=dt)
    kurt = float(((Z ** 4).mean(0) / (Z ** 2).mean(0) ** 2).mean())
    n = int(round(args.alpha * d)); E = args.episodes
    g = torch.Generator().manual_seed(args.seed + d)
    idx = torch.stack([torch.randperm(len(Z), generator=g)[:n + args.nq] for _ in range(E)])
    Xall = Z[idx]                                                  # (E, n+nq, d)
    u = torch.randn(E, d, generator=g, dtype=dt); u = u / u.norm(dim=1, keepdim=True)
    z = torch.einsum('end,ed->en', Xall, u)
    yall = link.f(z)
    X, Xq = Xall[:, :n].to(args.device), Xall[:, n:].to(args.device)
    y = (yall[:, :n] + args.sigma * torch.randn(E, n, generator=g, dtype=dt)).to(args.device)
    yq = yall[:, n:].to(args.device)
    res, tm = S.run_suite(link, X, y, Xq, yq, u.to(args.device), T=args.T, R=args.restarts, bp_big=args.bp_big,
                          sigma=args.sigma, seed=args.seed + d, skip=args.skip,
                          grid_lim='auto' if args.auto_grid else None)
    rs = S.summarise(res, dict(features=Path(args.features).stem, d=d, n=n, input_kurtosis=kurt))
    rows += rs + [dict(features=Path(args.features).stem, d=d, method='__time__', seconds={k: round(v, 3) for k, v in tm.t.items()})]
    print(f'{Path(args.features).stem} d={d} n={n} mean-kurtosis={kurt:.2f}', flush=True)
    for r in rs:
        print(f"  {r['method']:24s} nmse={r['nmse']:.4f}±{r['nmse_se']:.4f} med={r['nmse_median']:.4f} "
              f"succ={r['success']:.2f} ov={r['overlap'] if r['overlap'] is None else round(r['overlap'], 3)}", flush=True)
    print('  time:', {k: round(v, 2) for k, v in tm.t.items()}, flush=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
