"""Exp I (report 445): deep multi-branch memory, PC-ALM vs GAMP.

y = g2(a1 g(w1.x) + a2 g(w2.x) + b) + sigma*eps, g = He3/sqrt6 (layer 1), g2 = |.| (default) or He3 (--outer he3).
Unknown W (2 x d, orthonormal rows), a (unit), b. Both layers are multi-branch.
The He3∘He3 variant (degree 9) was unlearnable for every method incl. oracle GAMP in the smoke test; |.| keeps
two layers of unknown multi-branch structure without the degree-9 blow-up (change recorded in report 445).
Methods (queries only in the evaluator; selections by support loss):
  pcalm_deep_{1,8}x2   two-layer lifting: layer-1 pre-activations S and layer-2 pre-activation T are free
                       variables with multipliers L1, L2; T-step and per-unit S-steps are 1-D global solves
                       (grid + Newton), W and (a,b) by local least squares, rho in {0.1, 1}
  nomult_deep_{1,8}x2  identical, multipliers off
  bp_adam_64x3         Adam on (W, a, b)
  gamp_oracle_8x2      committee GAMP given the TRUE (a, b) (more information than any other method)
  gamp_grid(+adam)     committee GAMP over a 12 x 5 grid of (angle(a), b), best by support loss; then Adam on all
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--dims', type=int, nargs='+', default=[16, 32])
ap.add_argument('--alphas', type=float, nargs='+', default=[8, 16, 32])
ap.add_argument('--episodes', type=int, default=16)
ap.add_argument('--T', type=int, default=500)
ap.add_argument('--seed', type=int, default=445001)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--outer', default='abs', choices=['abs', 'he3'])
ap.add_argument('--matched', action='store_true', help='v3: compute-matched arms only (two budget levels)')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; dev = args.device; g = C.He3; sigma = 0.1; r = 2


class Abs:
    @staticmethod
    def f(z): return z.abs()
    @staticmethod
    def df(z): return torch.sign(z)
    @staticmethod
    def d2f(z): return torch.zeros_like(z)


g2 = Abs if args.outer == 'abs' else C.He3


def make(E, d, n, nq, seed):
    gen = torch.Generator().manual_seed(seed)
    W = torch.linalg.qr(torch.randn(E, d, r, generator=gen, dtype=dt))[0].transpose(1, 2)
    th = torch.rand(E, generator=gen, dtype=dt) * 2 * math.pi
    a = torch.stack([th.cos(), th.sin()], -1)
    b = torch.rand(E, generator=gen, dtype=dt) * 0.6 - 0.3
    X = torch.randn(E, n, d, generator=gen, dtype=dt); Xq = torch.randn(E, nq, d, generator=gen, dtype=dt)
    f = lambda Z: g2.f((a[:, None, :] * g.f(torch.einsum('end,erd->enr', Z, W))).sum(-1) + b[:, None])
    y = f(X) + sigma * torch.randn(E, n, generator=gen, dtype=dt)
    return [t.to(dev) for t in (W, a, b, X, y, Xq, f(Xq))]


def model(X, W, a, b):   # W (R,E,r,d) a (R,E,r) b (R,E) -> (R,E,n)
    S = torch.einsum('end,rejd->renj', X, W)
    return g2.f((a.unsqueeze(2) * g.f(S)).sum(-1) + b.unsqueeze(-1))


def sup_loss(X, y, W, a, b):
    return torch.nan_to_num(((model(X, W, a, b) - y) ** 2).mean(-1), nan=float('inf'))


def pick(X, y, P):
    W, a, b = P
    i = sup_loss(X, y, W, a, b).argmin(0); e = torch.arange(W.shape[1])
    return W[i, e].unsqueeze(0), a[i, e].unsqueeze(0), b[i, e].unsqueeze(0)


def cat(Ps):
    return tuple(torch.cat([p[k] for p in Ps]) for k in range(3))


def solve1d(yv, anchor, rho_t, grid, lk=None):
    """Global 1-D minimiser of (lk(s)-yv)^2 + rho_t/2 (s-anchor)^2 by grid + safeguarded Newton (numerical)."""
    lk = lk or g
    G = grid.view(*([1] * yv.dim()), -1)
    F = (lk.f(G) - yv.unsqueeze(-1)) ** 2 + 0.5 * rho_t.unsqueeze(-1) * (G - anchor.unsqueeze(-1)) ** 2
    s = grid[F.argmin(-1)]
    for _ in range(6):
        gs, dg, d2g = lk.f(s), lk.df(s), lk.d2f(s)
        F1 = 2 * (gs - yv) * dg + rho_t * (s - anchor)
        F2 = 2 * (dg ** 2 + (gs - yv) * d2g) + rho_t
        s = s - torch.where(F2 > 1e-9, F1 / F2, 0.1 * F1).clamp(-0.25, 0.25)
    return s


def pcalm_deep(X, y, P0, T, rho, mult, rho2=None):
    """rho: layer-1 constraint penalty; rho2: layer-2 (bilinear) constraint penalty (nonconvex ALM needs it large)."""
    rho2 = rho if rho2 is None else rho2
    W, a, b = (p.clone() for p in P0)
    R, E, _, d = W.shape
    grid = torch.linspace(-5, 5, 241, dtype=dt, device=dev)
    chol = torch.linalg.cholesky(torch.einsum('end,enk->edk', X, X) + 1e-2 * torch.eye(d, dtype=dt, device=dev))
    S = torch.einsum('end,rejd->renj', X, W)
    L1 = torch.zeros_like(S); L2 = torch.zeros(R, E, X.shape[1], dtype=dt, device=dev)
    full = lambda ref, v: torch.full_like(ref, v)
    for _ in range(T):
        P = torch.einsum('end,rejd->renj', X, W)
        M = (a.unsqueeze(2) * g.f(S)).sum(-1) + b.unsqueeze(-1)
        Tt = solve1d(y.unsqueeze(0).expand_as(M), M - L2 / rho2, full(M, rho2), grid, lk=g2)
        for j in range(r):
            aj = a[:, :, j].unsqueeze(-1)
            aj = torch.where(aj.abs() < 1e-3, torch.full_like(aj, 1e-3), aj)
            rest = M - aj * g.f(S[..., j])
            c = Tt + L2 / rho2 - rest
            # rho/2 (s-q)^2 + rho2/2 (aj g(s) - c)^2  ==  (rho2 aj^2/2) [ (g(s)-c/aj)^2 + (rho/(rho2 aj^2)) (s-q)^2 ]
            S[..., j] = solve1d(c / aj, P[..., j] - L1[..., j] / rho, (2 * rho / (rho2 * aj ** 2)).expand_as(c), grid)
            M = rest + aj * g.f(S[..., j])
        if mult:
            L1 = L1 + rho * (S - P); L2 = L2 + rho2 * (Tt - M)
        Wt = torch.cholesky_solve(torch.einsum('end,renj->redj', X, S + L1 / rho), chol.unsqueeze(0).expand(R, -1, -1, -1))
        W = Wt.transpose(-1, -2)
        D = torch.cat([g.f(S), torch.ones_like(S[..., :1])], -1)
        rhs = torch.einsum('renk,ren->rek', D, Tt + L2 / rho2)
        coef = torch.linalg.solve(torch.einsum('renk,renl->rekl', D, D) + 1e-6 * torch.eye(r + 1, dtype=dt, device=dev), rhs)
        a, b = coef[..., :r], coef[..., r]
    return W, a, b


def pcalm_deep_joint(X, y, P0, T, rho, mult, rho2, G=61, lim=4.0):
    """Two-layer PC-ALM whose per-example local subproblem is solved GLOBALLY and JOINTLY over (t, s1, s2):
    2-D grid over s = (s1, s2) (the same per-example 2-D grid GAMP uses for its output posterior), with the
    |.|-output t-subproblem solved in closed form for each grid point, then 3 projected coordinate-Newton
    polishing sweeps. Requires --outer abs. Parameters by local least squares, multipliers L1, L2."""
    assert g2 is Abs
    W, a, b = (p.clone() for p in P0)
    R, E, _, d = W.shape
    chol = torch.linalg.cholesky(torch.einsum('end,enk->edk', X, X) + 1e-2 * torch.eye(d, dtype=dt, device=dev))
    sg = torch.linspace(-lim, lim, G, dtype=dt, device=dev); gs = g.f(sg)
    S = torch.einsum('end,rejd->renj', X, W)
    L1 = torch.zeros_like(S); L2 = torch.zeros(R, E, X.shape[1], dtype=dt, device=dev)
    yy = y

    def tsolve(c):   # argmin_t (|t|-y)^2 + rho2/2 (t-c)^2, closed form over both sign branches
        tp = ((2 * yy + rho2 * c) / (2 + rho2)).clamp_min(0); tn = ((-2 * yy + rho2 * c) / (2 + rho2)).clamp_max(0)
        F = lambda t: (t.abs() - yy) ** 2 + 0.5 * rho2 * (t - c) ** 2
        Fp, Fn = F(tp), F(tn)
        return torch.where(Fp <= Fn, tp, tn), torch.minimum(Fp, Fn)

    for _ in range(T):
        P = torch.einsum('end,rejd->renj', X, W)
        Q = P - L1 / rho
        Snew = torch.empty_like(S); Tt = torch.empty_like(L2)
        for q in range(R):
            m = a[q, :, None, None, 0] * gs[None, :, None] + a[q, :, None, None, 1] * gs[None, None, :] + b[q, :, None, None]   # (E,G,G)
            c = m[:, None] - (L2[q] / rho2)[:, :, None, None]                                                            # (E,n,G,G)
            yy = y[:, :, None, None]
            _, V = tsolve(c)
            J = V + 0.5 * rho * ((sg[None, None, :, None] - Q[q, :, :, 0, None, None]) ** 2 +
                                 (sg[None, None, None, :] - Q[q, :, :, 1, None, None]) ** 2)
            idx = J.flatten(-2).argmin(-1)
            Snew[q, ..., 0] = sg[idx // G]; Snew[q, ..., 1] = sg[idx % G]
        yy = y.unsqueeze(0)
        for _p in range(3):                                  # coordinate Newton polish on the smooth part
            for j in range(r):
                other = a[..., 1 - j].unsqueeze(-1) * g.f(Snew[..., 1 - j]) + b.unsqueeze(-1)
                aj = a[..., j].unsqueeze(-1)
                m = other + aj * g.f(Snew[..., j]); c = m - L2 / rho2
                t, _ = tsolve(c)
                # d/ds [rho2/2 (t - m + L2/rho2)^2 + rho/2 (s - Q)^2]
                e = m - L2 / rho2 - t
                grad = rho2 * e * aj * g.df(Snew[..., j]) + rho * (Snew[..., j] - Q[..., j])
                hess = rho2 * (aj * g.df(Snew[..., j])) ** 2 + rho2 * e * aj * g.d2f(Snew[..., j]) + rho
                step = torch.where(hess > 1e-9, grad / hess, 0.1 * grad).clamp(-0.1, 0.1)
                Snew[..., j] = (Snew[..., j] - step).clamp(-lim - 1, lim + 1)
        S = Snew
        M = (a.unsqueeze(2) * g.f(S)).sum(-1) + b.unsqueeze(-1)
        Tt, _ = tsolve(M - L2 / rho2)
        if mult:
            L1 = L1 + rho * (S - P); L2 = L2 + rho2 * (Tt - M)
        Wt = torch.cholesky_solve(torch.einsum('end,renj->redj', X, S + L1 / rho), chol.unsqueeze(0).expand(R, -1, -1, -1))
        W = Wt.transpose(-1, -2)
        D = torch.cat([g.f(S), torch.ones_like(S[..., :1])], -1)
        rhs = torch.einsum('renk,ren->rek', D, Tt + L2 / rho2)
        coef = torch.linalg.solve(torch.einsum('renk,renl->rekl', D, D) + 1e-6 * torch.eye(r + 1, dtype=dt, device=dev), rhs)
        a, b = coef[..., :r], coef[..., r]
    return W, a, b


def adam(X, y, P0, steps, lr):
    W, a, b = (p.clone().requires_grad_() for p in P0)
    opt = torch.optim.Adam([W, a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); sup_loss(X, y, W, a, b).sum().backward(); opt.step()
    return W.detach(), a.detach(), b.detach()


def gamp_committee(X, y, a_true, b_true, W0, T, damp, G=41, zlim=5.0):
    """Committee GAMP (r=2) with known second layer (a, b); 2-D grid posterior, diagonal unit variances."""
    R, E, _, d = W0.shape
    X2 = X ** 2; rn = X2.sum(-1); v0 = 1.0 / d
    zg = torch.linspace(-zlim, zlim, G, dtype=dt, device=dev); gz = g.f(zg)
    inner = a_true[:, 0, None, None] * gz[None, :, None] + a_true[:, 1, None, None] * gz[None, None, :] + b_true[:, None, None]
    loglik = -(y[:, :, None, None] - g2.f(inner)[:, None]) ** 2 / (2 * sigma ** 2)          # (E,n,G,G)
    xh = W0.clone(); tx = torch.full((R, E, r), 0.5 * v0, dtype=dt, device=dev)
    sh = torch.zeros(R, E, X.shape[1], r, dtype=dt, device=dev)
    for _ in range(T):
        tp = (tx.unsqueeze(2) * rn[None, :, :, None]).clamp_min(1e-12)                   # (R,E,n,r)
        ph = torch.einsum('end,rejd->renj', X, xh) - tp * sh
        zh = torch.empty_like(ph); tz = torch.empty_like(ph)
        for q in range(R):
            lp = loglik - (zg[None, None, :, None] - ph[q, :, :, 0, None, None]) ** 2 / (2 * tp[q, :, :, 0, None, None]) \
                        - (zg[None, None, None, :] - ph[q, :, :, 1, None, None]) ** 2 / (2 * tp[q, :, :, 1, None, None])
            w = torch.softmax(lp.flatten(-2), -1).view_as(lp)
            m1 = (w.sum(-1) * zg).sum(-1); m2 = (w.sum(-2) * zg).sum(-1)
            zh[q, ..., 0] = m1; zh[q, ..., 1] = m2
            tz[q, ..., 0] = (w.sum(-1) * zg ** 2).sum(-1) - m1 ** 2
            tz[q, ..., 1] = (w.sum(-2) * zg ** 2).sum(-1) - m2 ** 2
        sn = (zh - ph) / tp
        ts = ((1 - tz / tp) / tp).clamp_min(1e-8)
        sh = damp * sn + (1 - damp) * sh
        tr = 1.0 / torch.einsum('end,renj->rejd', X2, ts).clamp_min(1e-12)
        rhat = xh + tr * torch.einsum('end,renj->rejd', X, sh)
        xn = rhat * v0 / (v0 + tr)
        tx = damp * (v0 * tr / (v0 + tr)).mean(-1) + (1 - damp) * tx
        xh = torch.nan_to_num(damp * xn + (1 - damp) * xh)
    return xh


rows = []
for d in args.dims:
    for alpha in args.alphas:
        n = int(alpha * d); E = args.episodes
        Wt, at, bt, X, y, Xq, yq = make(E, d, n, 1000, args.seed + 1000 * d + int(alpha))
        gen = torch.Generator().manual_seed(args.seed + 7 + d)
        R0 = 64
        W0 = (torch.randn(R0, E, r, d, generator=gen, dtype=dt) / math.sqrt(d)).to(dev)
        th0 = torch.rand(R0, E, generator=gen, dtype=dt).to(dev) * 2 * math.pi
        a0 = torch.stack([th0.cos(), th0.sin()], -1); b0 = torch.zeros(R0, E, dtype=dt, device=dev)
        init = lambda k: (W0[:k], a0[:k], b0[:k])
        res, tm = {}, {}

        def run(name, fn):
            torch.cuda.synchronize(); t0 = time.perf_counter(); res[name] = fn(); torch.cuda.synchronize()
            tm[name] = time.perf_counter() - t0

        if args.matched:
            # v3 (report 446): two budget levels. Low ~ GAMP-grid60+Adam; high ~ PC-ALM 8x2+Adam.
            run('pcalm_deepjoint_2x2+adam', lambda: pick(X, y, cat([adam(X, y, pick(X, y, cat([pcalm_deep_joint(X, y, init(2), args.T, .1, True, r2) for r2 in [10., 30.]])), 300, lr) for lr in [.003, .01]])))
            run('nomult_deepjoint_2x2+adam', lambda: pick(X, y, cat([adam(X, y, pick(X, y, cat([pcalm_deep_joint(X, y, init(2), args.T, .1, False, r2) for r2 in [10., 30.]])), 300, lr) for lr in [.003, .01]])))
            run('pcalm_deepjoint_8x2+adam', lambda: pick(X, y, cat([adam(X, y, pick(X, y, cat([pcalm_deep_joint(X, y, init(8), args.T, .1, True, r2) for r2 in [10., 30.]])), 300, lr) for lr in [.003, .01]])))

            def grid_gamp_n(na, nb):
                cands = []
                for thv in torch.linspace(0, math.pi / 2, na):
                    for bv in torch.linspace(-0.3, 0.3, nb):
                        ac = torch.stack([thv.cos(), thv.sin()]).to(dt).to(dev).expand(E, 2).contiguous()
                        bc = torch.full((E,), float(bv), dtype=dt, device=dev)
                        cands.append((gamp_committee(X, y, ac, bc, 0.1 * W0[:1], 150, 0.5), ac.unsqueeze(0), bc.unsqueeze(0)))
                return pick(X, y, cat(cands))
            run('gamp_grid60+adam', lambda: pick(X, y, cat([adam(X, y, grid_gamp_n(12, 5), 300, lr) for lr in [.003, .01]])))
            run('gamp_grid240+adam', lambda: pick(X, y, cat([adam(X, y, grid_gamp_n(24, 10), 300, lr) for lr in [.003, .01]])))

            def oracle_n(k):
                Wc = torch.cat([gamp_committee(X, y, at, bt, 0.1 * W0[:k], 200, dm) for dm in [.3, .6]]); Rr = Wc.shape[0]
                return pick(X, y, (Wc, at.unsqueeze(0).expand(Rr, -1, -1), bt.unsqueeze(0).expand(Rr, -1)))
            run('gamp_oracle_8x2+adam', lambda: pick(X, y, cat([adam(X, y, oracle_n(8), 300, lr) for lr in [.003, .01]])))
            run('bp_adam_64x3', lambda: pick(X, y, cat([adam(X, y, init(64), 1000, lr) for lr in [.003, .01, .03]])))
        else:
            # v2 (after smoke/debug, recorded in 445): joint 2-D local solve, rho=0.1 for layer 1, rho2 in {10, 30}
            # for the bilinear layer-2 constraint, selected by support loss; same grid for the no-multiplier control.
            for k in [1, 8]:
                run(f'pcalm_deepjoint_{k}x2', lambda: pick(X, y, cat([pcalm_deep_joint(X, y, init(k), args.T, .1, True, r2) for r2 in [10., 30.]])))
                run(f'nomult_deepjoint_{k}x2', lambda: pick(X, y, cat([pcalm_deep_joint(X, y, init(k), args.T, .1, False, r2) for r2 in [10., 30.]])))
            run('pcalm_deepjoint_8x2+adam', lambda: pick(X, y, cat([adam(X, y, res['pcalm_deepjoint_8x2'], 300, lr) for lr in [.003, .01]])))
            tm['pcalm_deepjoint_8x2+adam'] += tm['pcalm_deepjoint_8x2']
            run('bp_adam_64x3', lambda: pick(X, y, cat([adam(X, y, init(64), 1000, lr) for lr in [.003, .01, .03]])))

            def oracle():
                Ws = [gamp_committee(X, y, at, bt, 0.1 * W0[:8], 200, dm) for dm in [.3, .6]]
                Wc = torch.cat(Ws); Rr = Wc.shape[0]
                return pick(X, y, (Wc, at.unsqueeze(0).expand(Rr, -1, -1), bt.unsqueeze(0).expand(Rr, -1)))
            run('gamp_oracle_8x2', oracle)
            run('gamp_oracle_8x2+adam', lambda: pick(X, y, cat([adam(X, y, res['gamp_oracle_8x2'], 300, lr) for lr in [.003, .01]])))
            tm['gamp_oracle_8x2+adam'] += tm['gamp_oracle_8x2']

            def grid_gamp():
                cands = []
                for thv in torch.linspace(0, math.pi / 2, 12):
                    for bv in torch.linspace(-0.3, 0.3, 5):
                        ac = torch.stack([thv.cos(), thv.sin()]).to(dt).to(dev).expand(E, 2).contiguous()
                        bc = torch.full((E,), float(bv), dtype=dt, device=dev)
                        Wc = gamp_committee(X, y, ac, bc, 0.1 * W0[:1], 150, 0.5)
                        cands.append((Wc, ac.unsqueeze(0), bc.unsqueeze(0)))
                return pick(X, y, cat(cands))
            run('gamp_grid60', grid_gamp)
            run('gamp_grid60+adam', lambda: pick(X, y, cat([adam(X, y, res['gamp_grid60'], 300, lr) for lr in [.003, .01]])))
            tm['gamp_grid60+adam'] += tm['gamp_grid60']
        vq = yq.var(-1)
        for name, (W, a, b) in res.items():
            m = ((model(Xq, W, a, b)[0] - yq) ** 2).mean(-1) / vq
            Pw = torch.einsum('erd,ejd->ejr', Wt, W[0]); ov = (Pw.norm(dim=-1) / W[0].norm(dim=-1).clamp_min(1e-12)).mean(-1)
            rows.append(dict(d=d, alpha=alpha, n=n, method=name, success=float((m < .05).float().mean()),
                             nmse_mean=float(m.mean()), nmse_median=float(m.median()),
                             subspace_overlap=float(ov.mean()), seconds=tm[name]))
        cur = [x for x in rows if x['d'] == d and x['alpha'] == alpha]
        print(f'd={d} alpha={alpha} n={n}: ' + '  '.join(f"{x['method']}={x['success']:.2f}/{x['subspace_overlap']:.2f}({x['seconds']:.0f}s)" for x in cur), flush=True)
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
