"""Exp H (report 444): GAMP, a Bayes message-passing baseline, on the He3 single-index tasks.

Prediction written before running: on i.i.d. Gaussian designs GAMP matches or beats PC-ALM (AMP is near
Bayes-optimal for Gaussian GLMs); on real, non-Gaussian / heavy-tailed representations (C1 features) AMP may lose
its guarantees, and that is where PC-ALM could keep an advantage. Both outcomes are reported.
Lifted methods here use the fast grid+Newton activity solver (certified-vs-grid discrepancy is audited in G).
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--mode', choices=['gauss', 'real'], required=True)
ap.add_argument('--features', default=None)
ap.add_argument('--dims', type=int, nargs='+', default=[64, 128, 256])
ap.add_argument('--alphas', type=float, nargs='+', default=[2, 2.5, 3, 4])
ap.add_argument('--episodes', type=int, default=64)
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--seed', type=int, default=444701)
ap.add_argument('--gamp_damps', type=float, nargs='+', default=[.3, .6])
ap.add_argument('--gamp_T', type=int, default=None)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.He3; dev = args.device; sigma = 0.1


def real_tasks(d, n, E, nq, seed):
    F = np.load(args.features)['X']; rng = np.random.default_rng(seed)
    perm = rng.permutation(len(F)); fit, pool = F[perm[:len(F) // 2]], F[perm[len(F) // 2:]]
    mu = fit.mean(0); ev, evec = np.linalg.eigh(np.cov(fit - mu, rowvar=False)); o = np.argsort(ev)[::-1]
    Z = torch.tensor((pool - mu) @ (evec[:, o[:d]] / np.sqrt(ev[o[:d]])), dtype=dt)
    g = torch.Generator().manual_seed(seed + d)
    idx = torch.stack([torch.randperm(len(Z), generator=g)[:n + nq] for _ in range(E)])
    Xall = Z[idx]; u = torch.randn(E, d, generator=g, dtype=dt); u = u / u.norm(dim=1, keepdim=True)
    yall = link.f(torch.einsum('end,ed->en', Xall, u))
    y = yall[:, :n] + sigma * torch.randn(E, n, generator=g, dtype=dt)
    return dict(X=Xall[:, :n].to(dev), y=y.to(dev), Xq=Xall[:, n:].to(dev), yq=yall[:, n:].to(dev), u=u.to(dev))


rows = []
for d in args.dims:
    for alpha in args.alphas:
        n = int(alpha * d); E = args.episodes
        if args.mode == 'gauss':
            t = C.make_tasks(E, d, n, 1000, link, sigma, args.seed + 1000 * d + int(10 * alpha), dev, dt)
            C.GRID_LIM = 5.0; glim = None; zlim = 8.0
        else:
            t = real_tasks(d, n, E, 1000, args.seed + int(10 * alpha))
            glim = 'auto'; zlim = float(max(8.0, (6 ** .5 * float(t['y'].abs().max())) ** (1 / 3) + 2))
        X, y, Xq, yq, u = t['X'], t['y'], t['Xq'], t['yq'], t['u']
        W0 = C.init_w(E, d, 64, args.seed + 7 + d, dev, dt)
        res, tm = {}, {}

        def run(name, fn):
            torch.cuda.synchronize(); t0 = time.perf_counter(); res[name] = fn(); torch.cuda.synchronize()
            tm[name] = time.perf_counter() - t0

        gT = args.gamp_T or args.T; nd = len(args.gamp_damps)
        run(f'gamp_8x{nd}', lambda: C.pick_best(link, X, y, torch.cat(
            [C.gamp_glm(link, X, y, 0.1 * W0[:8], gT, sigma, dm, zlim=zlim) for dm in args.gamp_damps])))
        run(f'gamp_1x{nd}', lambda: C.pick_best(link, X, y, torch.cat(
            [C.gamp_glm(link, X, y, 0.1 * W0[:1], gT, sigma, dm, zlim=zlim) for dm in args.gamp_damps])))
        run('pc_alm_1x2', lambda: C.pick_best(link, X, y, torch.cat([C.lifted(link, X, y, W0[:1], args.T, r, grid_lim=glim) for r in [.1, 1.]])))
        run('pc_alm_8x2', lambda: C.pick_best(link, X, y, torch.cat([C.lifted(link, X, y, W0[:8], args.T, r, grid_lim=glim) for r in [.1, 1.]])))
        run('bayes_transform+LM', lambda: C.bp_lm(link, X, y, C.hybrid_transform_init(link, X, y, sigma, zlim=zlim).unsqueeze(0), args.T)[0])
        run('bp_adam_64x3', lambda: C.pick_best(link, X, y, torch.cat([C.bp_adam(link, X, y, W0, args.T, lr) for lr in [.01, .03, .1]])))
        vq = yq.var(-1)
        for name, w in res.items():
            m = C.query_mse_w(link, Xq, yq, w) / vq
            rows.append(dict(mode=args.mode, features=None if args.features is None else Path(args.features).stem,
                             d=d, alpha=alpha, n=n, method=name, success=float((m < .05).float().mean()),
                             nmse_mean=float(m.mean()), nmse_median=float(m.median()),
                             overlap=float(C.overlap(w, u).mean()), seconds=tm[name]))
        cur = [r for r in rows if r['d'] == d and r['alpha'] == alpha]
        print(f'{args.mode} d={d} alpha={alpha}: ' + '  '.join(f"{r['method']}={r['success']:.2f}/{r['overlap']:.3f}({r['seconds']:.1f}s)" for r in cur), flush=True)
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
