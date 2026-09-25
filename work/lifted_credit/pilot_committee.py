"""Exp B (report 440, P4): two-layer committee memory, r hidden units, unknown first layer.

Task: U in R^{r x d} orthonormal rows, y = (1/sqrt r) sum_j g(u_j.x) + sigma*eps.
Model for every model-based method: f_W(x) = (1/sqrt r) sum_j g(w_j.x).
The analytic per-example transform E[z|y] no longer identifies individual units.
"""
import argparse
import json
import math
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--link', default='he3')
ap.add_argument('--r', type=int, default=2)
ap.add_argument('--dims', type=int, nargs='+', default=[16, 32, 64, 128])
ap.add_argument('--alpha', type=float, default=4.0, help='n = alpha * r * d')
ap.add_argument('--episodes', type=int, default=32)
ap.add_argument('--nq', type=int, default=2000)
ap.add_argument('--sigma', type=float, default=0.1)
ap.add_argument('--T', type=int, default=300)
ap.add_argument('--restarts', type=int, default=8)
ap.add_argument('--seed', type=int, default=440101)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--skip', nargs='*', default=[])
ap.add_argument('--bp_big', type=int, default=0, help='extra compute-matched BP restarts')
ap.add_argument('--out', required=True)
args = ap.parse_args()
dt = torch.float64
link = C.LINKS[args.link]
r = args.r
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))


def make(E, d, n, nq, seed, dev):
    g = torch.Generator().manual_seed(seed)
    U = torch.linalg.qr(torch.randn(E, d, r, generator=g, dtype=dt))[0].transpose(1, 2)  # (E,r,d)
    X = torch.randn(E, n, d, generator=g, dtype=dt); Xq = torch.randn(E, nq, d, generator=g, dtype=dt)
    f = lambda Z: link.f(torch.einsum('end,erd->enr', Z, U)).sum(-1) / math.sqrt(r)
    y = f(X) + args.sigma * torch.randn(E, n, generator=g, dtype=dt)
    return [t.to(dev) for t in (U, X, y, Xq, f(Xq))]


def fW(X, W):  # W (...,E,r,d) -> (...,E,n)
    return link.f(torch.einsum('end,...erd->...enr', X, W)).sum(-1) / math.sqrt(r)


def sup_loss(X, y, W):
    return torch.nan_to_num(((fW(X, W) - y) ** 2).mean(-1), nan=float('inf'))


def pick(X, y, W):
    idx = sup_loss(X, y, W).argmin(0)
    return W[idx, torch.arange(W.shape[1])]


def grad(X, y, W):
    S = torch.einsum('end,rejd->renj', X, W)
    res = fW(X, W) - y                                           # (R,E,n)
    return 2 / X.shape[1] / math.sqrt(r) * torch.einsum('ren,renj,end->rejd', res, link.df(S), X)


def gd(X, y, W0, T, lr):
    W = W0.clone()
    for _ in range(T):
        W = torch.nan_to_num(W - lr * grad(X, y, W)).clamp(-50, 50)
    return W


def adam(X, y, W0, T, lr):
    W = W0.clone(); m = torch.zeros_like(W); v = torch.zeros_like(W)
    for t in range(1, T + 1):
        gr = grad(X, y, W); m = .9 * m + .1 * gr; v = .999 * v + .001 * gr ** 2
        W = torch.nan_to_num(W - lr * (m / (1 - .9 ** t)) / ((v / (1 - .999 ** t)).sqrt() + 1e-8)).clamp(-50, 50)
    return W


def lm(X, y, W0, T):
    W = W0.clone(); R, E, rr, d = W.shape; P = rr * d
    mu = torch.ones(R, E, dtype=dt, device=X.device); eye = torch.eye(P, dtype=dt, device=X.device)
    L = sup_loss(X, y, W)
    for _ in range(T):
        S = torch.einsum('end,rejd->renj', X, W)
        J = (link.df(S).unsqueeze(-1) * X[None, :, :, None, :]).reshape(R, E, X.shape[1], P) / math.sqrt(r)
        res = y - fW(X, W)
        A = torch.einsum('renp,renq->repq', J, J) + mu[..., None, None] * eye
        step = torch.linalg.solve(A, torch.einsum('renp,ren->rep', J, res).unsqueeze(-1)).squeeze(-1)
        Wn = W + step.reshape(R, E, rr, d); Ln = sup_loss(X, y, Wn); ok = Ln < L
        W = torch.where(ok[..., None, None], Wn, W); L = torch.where(ok, Ln, L)
        mu = torch.where(ok, mu / 3, mu * 3).clamp(1e-8, 1e8)
    return W


def lifted(X, y, W0, T, rho, use_mult=True, growth=1.0, mu=1e-2, sweeps=1):
    """PC-ALM on the committee: per-unit activities s_ij, constraints s_ij = w_j.x_i.

    Activity block: Gauss-Seidel over units, each an exact global 1-D minimisation.
    Parameter block: exact local least squares per unit (shared Gram matrix).
    """
    R, E, rr, d = W0.shape
    grid = torch.linspace(-5, 5, 241, dtype=dt, device=X.device)
    A = torch.einsum('end,enk->edk', X, X) + mu * torch.eye(d, dtype=dt, device=X.device)
    Achol = torch.linalg.cholesky(A)
    W = W0.clone(); lam = torch.zeros(R, E, X.shape[1], rr, dtype=dt, device=X.device)
    S = torch.einsum('end,rejd->renj', X, W); rt = math.sqrt(r); p = rho
    for _ in range(T):
        pred = torch.einsum('end,rejd->renj', X, W)
        for _s in range(sweeps):
            for j in range(rr):
                c = link.f(S).sum(-1) - link.f(S[..., j])
                S[..., j] = C._activity_step(link, rt * y - c, pred[..., j] - lam[..., j] / p, rr * p, grid)
        if use_mult:
            lam = lam + p * (S - pred)
        B = torch.einsum('end,renj->redj', X, S + lam / p)          # (R,E,d,r)
        Wt = torch.cholesky_solve(B, Achol.unsqueeze(0).expand(R, -1, -1, -1))
        W = Wt.transpose(-1, -2)
        p = p * growth
    return W


def hybrid(X, y, W0, T, dev):
    """Closed-form non-CSQ init (MC estimate of E[z_1|y]) + per-unit random symmetry breaking + LM."""
    g = torch.Generator().manual_seed(7)
    Z = torch.randn(400000, r, generator=g, dtype=dt)
    ys = link.f(Z).sum(-1) / math.sqrt(r) + args.sigma * torch.randn(400000, generator=g, dtype=dt)
    edges = torch.quantile(ys, torch.linspace(0, 1, 201, dtype=dt))
    b = torch.bucketize(ys, edges[1:-1])
    tbin = torch.zeros(200, dtype=dt).index_add_(0, b, Z[:, 0]) / torch.bincount(b, minlength=200).clamp_min(1)
    tb = tbin.to(dev)[torch.bucketize(y, edges[1:-1].to(dev))]
    d = X.shape[-1]
    A = torch.einsum('end,enk->edk', X, X) + 1e-2 * torch.eye(d, dtype=dt, device=dev)
    w = torch.linalg.solve(A, torch.einsum('end,en->ed', X, tb).unsqueeze(-1)).squeeze(-1)
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    Winit = w[None, :, None, :] + W0                          # symmetric direction + per-unit random break
    return lm(X, y, Winit, T), Winit


rows = []
for d in args.dims:
    n = int(round(args.alpha * r * d))
    U, X, y, Xq, yq = make(args.episodes, d, n, args.nq, args.seed + d, args.device)
    tm = C.Timer(); res = {}
    qm = lambda W: ((fW(Xq, W) - yq) ** 2).mean(-1)
    def sub_ov(W):  # mean over units of best |cos| to true subspace alignment: ||P_U w_j|| / ||w_j||
        Pw = torch.einsum('erd,ejd->ejr', U, W)
        return (Pw.norm(dim=-1) / W.norm(dim=-1).clamp_min(1e-12)).mean(-1)
    def rec(name, W): res[name] = (qm(W), sub_ov(W))
    def recp(name, P): res[name] = (((P - yq) ** 2).mean(-1), None)
    rec('oracle', U); res['zero'] = ((yq ** 2).mean(-1), None)
    with tm('closed'):
        recp('linear_ridge', C.linear_ridge(X, y, Xq))
        if 'rf' not in args.skip:
            for kind in ['relu', 'link']:
                F, Fq = C.random_features(X, Xq, 4096, kind, link, args.seed + 7); recp(f'rf4096_{kind}_ridge', C.features_ridge(F, Fq, y))
        if 'rbf' not in args.skip:
            recp('rbf_krr', C.rbf_krr(X, y, Xq))
        recp('poly3_krr', C.poly3_krr(X, y, Xq))
    if 'mlp' not in args.skip:
        with tm('mlp'): recp('mlp_extra_layer', C.mlp_extra_layer(X, y, Xq, T=2 * args.T))
    R = args.restarts
    g0 = torch.Generator().manual_seed(args.seed + 999 + d)
    W0 = (torch.randn(max(R, args.bp_big), args.episodes, r, d, generator=g0, dtype=dt) / math.sqrt(d)).to(args.device)
    if args.bp_big:
        Rb = args.bp_big
        with tm(f'bp_adam_{Rb}x3'): rec(f'bp_adam_{Rb}x3', pick(X, y, torch.cat([adam(X, y, W0[:Rb], args.T, lr) for lr in [.01, .03, .1]])))
        with tm(f'bp_gd_{Rb}x3'): rec(f'bp_gd_{Rb}x3', pick(X, y, torch.cat([gd(X, y, W0[:Rb], args.T, lr) for lr in [.01, .03, .1]])))
    W0 = W0[:R]
    with tm('bp_gd'): rec(f'bp_gd_{R}x3', pick(X, y, torch.cat([gd(X, y, W0, args.T, lr) for lr in [.01, .03, .1]])))
    with tm('bp_adam'): rec(f'bp_adam_{R}x3', pick(X, y, torch.cat([adam(X, y, W0, args.T, lr) for lr in [.01, .03, .1]])))
    with tm('bp_lm'): rec(f'bp_lm_{R}', pick(X, y, lm(X, y, W0, args.T)))
    W1 = W0[:1]
    with tm('pc_alm'): rec('pc_alm_1x2', pick(X, y, torch.cat([lifted(X, y, W1, args.T, rho) for rho in [.1, 1.]])))
    with tm('pc_alm_R'): rec(f'pc_alm_{R}x2', pick(X, y, torch.cat([lifted(X, y, W0, args.T, rho) for rho in [.1, 1.]])))
    with tm('pc_pen'): rec('pc_penalty_cont', lifted(X, y, W1, args.T, .1, False, 1000 ** (1 / args.T))[0])
    with tm('altmin'): rec('altmin', lifted(X, y, W1, args.T, 1e-3, False)[0])
    with tm('hybrid'):
        Wh, Wi = hybrid(X, y, W0, args.T, args.device)
        rec('hybrid_transform_only', Wi[0]); rec(f'hybrid_transform_lm_{R}', pick(X, y, Wh))
    print(f'd={d} n={n} r={r}', flush=True)
    for name, (m, ov) in res.items():
        row = dict(d=d, n=n, r=r, method=name, mse=float(m.mean()), mse_se=float(m.std() / len(m) ** .5),
                   median=float(m.median()), success=float((m < 0.05).float().mean()),
                   overlap=None if ov is None else float(ov.mean()))
        rows.append(row)
        print(f"  {name:24s} mse={row['mse']:.4f}±{row['mse_se']:.4f} med={row['median']:.4f} succ={row['success']:.2f} "
              f"ov={None if ov is None else round(row['overlap'], 3)}", flush=True)
    rows.append(dict(d=d, n=n, r=r, method='__time__', seconds={k: round(v, 3) for k, v in tm.t.items()}))
    print('  time:', {k: round(v, 2) for k, v in tm.t.items()}, flush=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
