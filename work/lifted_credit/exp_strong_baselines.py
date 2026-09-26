"""Exp G (report 444): the controls requested in review of 442.

Single-index He3 tasks, n = alpha*d. All lifted methods use the CERTIFIED activity solver (all real roots of the
stationarity quintic). Controls:
  pc_alm_cert_{1,8}x2        PC-ALM, rho in {0.1, 1} chosen by support loss
  nomult_cert_{1,8}x2        identical solver and rho selection, multipliers off  (same-rho control)
  altmin_cert_{1,8}          rho = 1e-3, multipliers off (Gerchberg-Saxton limit), certified solver
  y3_transform(+LM)          E[x y^3] = 9 sqrt6 u : closed-form non-CSQ direction, then LM refinement
  bayes_transform(+LM)       E[z|y] label transform (442 'hybrid')
  drsgd_{sq,corr}_R          data-reuse spherical minibatch SGD (each minibatch used for 2 consecutive steps),
                             adapted from Lee-Oko-Suzuki-Wu (NeurIPS 2024) to the known-link neuron; step grid
  bp_adam_64x3               reference from 442
Queries only in the evaluator. Every selection among restarts / hyper-parameters uses support loss only.
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128, 256])
ap.add_argument('--alphas', type=float, nargs='+', default=[3, 4, 6])
ap.add_argument('--episodes', type=int, default=64)
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--drsgd_epochs', type=int, default=100)
ap.add_argument('--seed', type=int, default=444001)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.He3; dev = args.device


def normalize(W): return W / W.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def ls_dir(X, t):
    A = torch.einsum('end,enk->edk', X, X) + 1e-2 * torch.eye(X.shape[-1], dtype=dt, device=dev)
    return normalize(torch.linalg.solve(A, torch.einsum('end,en->ed', X, t).unsqueeze(-1)).squeeze(-1))


def drsgd(X, y, W0, loss, c, epochs, b=8, g=None):
    """Spherical minibatch SGD; every minibatch is used for two consecutive steps (data reuse)."""
    R, E, d = W0.shape; n = X.shape[1]; W = normalize(W0.clone()); eta = c / math.sqrt(d)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(dev)
        for k in range(0, n - b + 1, b):
            Xb, yb = X[:, perm[k:k + b]], y[:, perm[k:k + b]]
            for _rep in range(2):
                s = torch.einsum('ebd,red->reb', Xb, W)
                coef = (link.f(s) - yb) * link.df(s) if loss == 'sq' else -yb * link.df(s)
                G = torch.einsum('reb,ebd->red', coef, Xb) / b
                G = G - (G * W).sum(-1, keepdim=True) * W            # spherical (tangent) gradient
                W = normalize(torch.nan_to_num(W - eta * G))
    return W


rows = []
for d in args.dims:
    for alpha in args.alphas:
        n = int(alpha * d); E = args.episodes
        task = C.make_tasks(E, d, n, 1000, link, 0.1, args.seed + 1000 * d + int(10 * alpha), dev, dt)
        X, y, Xq, yq, u = task['X'], task['y'], task['Xq'], task['yq'], task['u']
        W0 = C.init_w(E, d, 64, args.seed + 7 + d, dev, dt)
        gen = torch.Generator().manual_seed(args.seed + 11 + d)
        res, tm, cert = {}, {}, {}

        def run(name, fn):
            torch.cuda.synchronize(); C.CERT_LOG.clear(); t0 = time.perf_counter()
            res[name] = fn(); torch.cuda.synchronize(); tm[name] = time.perf_counter() - t0
            if C.CERT_LOG:
                cert[name] = dict(calls=len(C.CERT_LOG), solves=sum(s['n'] for s in C.CERT_LOG),
                                  aberth_converged_min=min(s['aberth_converged'] for s in C.CERT_LOG),
                                  fallbacks=sum(s['fallback'] for s in C.CERT_LOG),
                                  max_residual=max(s['max_abs_residual'] for s in C.CERT_LOG))

        lift = lambda W, rho, mult: C.lifted(link, X, y, W, args.T, rho, mult, activity='certified')
        for k in [1, 8]:
            run(f'pc_alm_cert_{k}x2', lambda: C.pick_best(link, X, y, torch.cat([lift(W0[:k], r, True) for r in [.1, 1.]])))
            run(f'nomult_cert_{k}x2', lambda: C.pick_best(link, X, y, torch.cat([lift(W0[:k], r, False) for r in [.1, 1.]])))
            run(f'altmin_cert_{k}', lambda: C.pick_best(link, X, y, lift(W0[:k], 1e-3, False)))
        run('y3_transform', lambda: ls_dir(X, y ** 3))
        run('y3_transform+LM', lambda: C.bp_lm(link, X, y, ls_dir(X, y ** 3).unsqueeze(0), args.T)[0])
        run('bayes_transform+LM', lambda: C.bp_lm(link, X, y, C.hybrid_transform_init(link, X, y, 0.1).unsqueeze(0), args.T)[0])
        for loss in ['sq', 'corr']:
            run(f'drsgd_{loss}_8x4', lambda: C.pick_best(link, X, y, torch.cat(
                [drsgd(X, y, W0[:8], loss, c, args.drsgd_epochs, g=gen) for c in [.1, .3, 1., 3.]])))
        run('bp_adam_64x3', lambda: C.pick_best(link, X, y, torch.cat([C.bp_adam(link, X, y, W0, args.T, lr) for lr in [.01, .03, .1]])))
        for name, w in res.items():
            m = C.query_mse_w(link, Xq, yq, w)
            rows.append(dict(d=d, alpha=alpha, n=n, method=name, success=float((m < .05).float().mean()),
                             mse=float(m.mean()), mse_median=float(m.median()), overlap=float(C.overlap(w, u).mean()),
                             seconds=tm[name], certificate=cert.get(name)))
        cur = [r for r in rows if r['d'] == d and r['alpha'] == alpha]
        print(f'd={d} alpha={alpha} n={n}: ' + '  '.join(f"{r['method']}={r['success']:.2f}({r['seconds']:.1f}s)" for r in cur), flush=True)
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
