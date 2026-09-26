"""Exp K (report 447): deep multi-branch memory, PC-ALM vs FULL-COVARIANCE committee AMP (oracle and EM).

Same task family and seeds as exp_deep.py --matched (report 446 v3), so PC-ALM rows should reproduce v3.
y = |a1 g(w1.x) + a2 g(w2.x) + b| + sigma*eps, g = He3/sqrt6, W (2 x d), a (unit), b unknown.
Per-task NMSE is saved for paired comparisons. Query data only in the evaluator; selections by support loss.
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, required=True)
ap.add_argument('--alpha', type=float, default=16)
ap.add_argument('--episodes', type=int, default=32)
ap.add_argument('--T', type=int, default=500)
ap.add_argument('--amp_T', type=int, default=300)
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--only', nargs='*', default=None)
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; dev = args.device; g = C.He3; sigma = 0.1; r = 2


class Abs:
    @staticmethod
    def f(z): return z.abs()


g2 = Abs


# ---------------------------------------------------------------- task, model, helpers (identical to exp_deep.py)
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


def model(X, W, a, b):
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


def adam(X, y, P0, steps, lr):
    W, a, b = (p.clone().requires_grad_() for p in P0)
    opt = torch.optim.Adam([W, a, b], lr=lr)
    for _ in range(steps):
        opt.zero_grad(); sup_loss(X, y, W, a, b).sum().backward(); opt.step()
    return W.detach(), a.detach(), b.detach()


def polish(X, y, P):
    return pick(X, y, cat([adam(X, y, P, 300, lr) for lr in [.003, .01]]))


def pcalm_deep_joint(X, y, P0, T, rho, mult, rho2, G=61, lim=4.0):
    """Verbatim from exp_deep.py (report 446)."""
    W, a, b = (p.clone() for p in P0)
    R, E, _, d = W.shape
    chol = torch.linalg.cholesky(torch.einsum('end,enk->edk', X, X) + 1e-2 * torch.eye(d, dtype=dt, device=dev))
    sg = torch.linspace(-lim, lim, G, dtype=dt, device=dev); gs = g.f(sg)
    S = torch.einsum('end,rejd->renj', X, W)
    L1 = torch.zeros_like(S); L2 = torch.zeros(R, E, X.shape[1], dtype=dt, device=dev)
    yy = y

    def tsolve(c):
        tp = ((2 * yy + rho2 * c) / (2 + rho2)).clamp_min(0); tn = ((-2 * yy + rho2 * c) / (2 + rho2)).clamp_max(0)
        F = lambda t: (t.abs() - yy) ** 2 + 0.5 * rho2 * (t - c) ** 2
        Fp, Fn = F(tp), F(tn)
        return torch.where(Fp <= Fn, tp, tn), torch.minimum(Fp, Fn)

    for _ in range(T):
        P = torch.einsum('end,rejd->renj', X, W)
        Q = P - L1 / rho
        Snew = torch.empty_like(S)
        for q in range(R):
            m = a[q, :, None, None, 0] * gs[None, :, None] + a[q, :, None, None, 1] * gs[None, None, :] + b[q, :, None, None]
            c = m[:, None] - (L2[q] / rho2)[:, :, None, None]
            yy = y[:, :, None, None]
            _, V = tsolve(c)
            J = V + 0.5 * rho * ((sg[None, None, :, None] - Q[q, :, :, 0, None, None]) ** 2 +
                                 (sg[None, None, None, :] - Q[q, :, :, 1, None, None]) ** 2)
            idx = J.flatten(-2).argmin(-1)
            Snew[q, ..., 0] = sg[idx // G]; Snew[q, ..., 1] = sg[idx % G]
        yy = y.unsqueeze(0)
        for _p in range(3):
            for j in range(r):
                other = a[..., 1 - j].unsqueeze(-1) * g.f(Snew[..., 1 - j]) + b.unsqueeze(-1)
                aj = a[..., j].unsqueeze(-1)
                m = other + aj * g.f(Snew[..., j]); c = m - L2 / rho2
                t, _ = tsolve(c)
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


def gamp_diag_oracle(X, y, a_true, b_true, W0, T, damp, G=41, zlim=5.0):
    """Old diagonal-variance committee GAMP (verbatim logic from exp_deep.py) for continuity."""
    R, E, _, d = W0.shape
    X2 = X ** 2; rn = X2.sum(-1); v0 = 1.0 / d
    zg = torch.linspace(-zlim, zlim, G, dtype=dt, device=dev); gz = g.f(zg)
    inner = a_true[:, 0, None, None] * gz[None, :, None] + a_true[:, 1, None, None] * gz[None, None, :] + b_true[:, None, None]
    loglik = -(y[:, :, None, None] - g2.f(inner)[:, None]) ** 2 / (2 * sigma ** 2)
    xh = W0.clone(); tx = torch.full((R, E, r), 0.5 * v0, dtype=dt, device=dev)
    sh = torch.zeros(R, E, X.shape[1], r, dtype=dt, device=dev)
    for _ in range(T):
        tp = (tx.unsqueeze(2) * rn[None, :, :, None]).clamp_min(1e-12)
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


# ---------------------------------------------------------------- full-covariance committee AMP
def amp_full(X, y, W0, a0, b0, T, damp, em, em_start=10, em_damp=0.5, G=51, zlim=5.0):
    """Vector GAMP for the K=2 committee (Aubin et al. 2018 form) with FULL 2x2 covariances.

    Output: z_mu ~ N(omega_mu, V_mu), V_mu = sum_i x_mu,i^2 C_i (2x2); posterior moments on a GxG grid;
            g = V^-1 (zhat - omega), dg = V^-1 (Sigma - V) V^-1.
    Input:  A_i = -sum_mu x_mu,i^2 dg_mu (2x2, eigen-clamped PSD), B_i = sum_mu x_mu,i g_mu + A_i W_i,
            Gaussian prior N(0, I/d): C_i = (d I + A_i)^-1, W_i = C_i B_i.
    em=True: from iteration em_start, (a, b) are updated every iteration by the M-step of the |.| channel,
            weighted least squares on the current sign branch (MM step), damped by em_damp.
    """
    R, E, K, d = W0.shape
    n = X.shape[1]; X2 = X ** 2; v0 = 1.0 / d
    I2 = torch.eye(K, dtype=dt, device=dev)
    zg = torch.linspace(-zlim, zlim, G, dtype=dt, device=dev); gz = g.f(zg)
    Z1, Z2 = zg[:, None], zg[None, :]
    W = W0.clone(); a = a0.clone(); b = b0.clone()
    Cc = (0.5 * v0 * I2).expand(R, E, d, K, K).clone()
    gh = torch.zeros(R, E, n, K, dtype=dt, device=dev)
    for t in range(T):
        V = torch.einsum('end,redkl->renkl', X2, Cc) + 1e-10 * I2
        Pm = torch.linalg.inv(V)
        omega = torch.einsum('end,rekd->renk', X, W) - (V @ gh.unsqueeze(-1)).squeeze(-1)
        zhat = torch.empty_like(omega); Sig = torch.empty_like(V)
        for q in range(R):
            inner = a[q, :, 0, None, None] * gz[None, :, None] + a[q, :, 1, None, None] * gz[None, None, :] + b[q, :, None, None]  # (E,G,G)
            ll = -(y[:, :, None, None] - inner.abs()[:, None]) ** 2 / (2 * sigma ** 2)                                         # (E,n,G,G)
            d1 = Z1 - omega[q, :, :, 0, None, None]; d2 = Z2 - omega[q, :, :, 1, None, None]
            P = Pm[q]
            quad = P[..., 0, 0, None, None] * d1 ** 2 + 2 * P[..., 0, 1, None, None] * d1 * d2 + P[..., 1, 1, None, None] * d2 ** 2
            w = torch.softmax((ll - 0.5 * quad).flatten(-2), -1).view(E, n, G, G)
            w1 = w.sum(-1); w2 = w.sum(-2)
            m1 = (w1 * zg).sum(-1); m2 = (w2 * zg).sum(-1)
            s11 = (w1 * zg ** 2).sum(-1) - m1 ** 2; s22 = (w2 * zg ** 2).sum(-1) - m2 ** 2
            s12 = (w * (Z1 * Z2)).sum((-1, -2)) - m1 * m2
            zhat[q] = torch.stack([m1, m2], -1)
            Sig[q] = torch.stack([torch.stack([s11, s12], -1), torch.stack([s12, s22], -1)], -2)
            if em and t >= em_start:
                sgn = torch.sign(inner)[:, None]                                                                   # (E,1,G,G)
                fe = [gz[:, None].expand(G, G), gz[None, :].expand(G, G), torch.ones(G, G, dtype=dt, device=dev)]
                M3 = torch.stack([torch.stack([(w * (fi * fj)).sum((-1, -2)).sum(-1) for fj in fe], -1) for fi in fe], -2)  # (E,3,3)
                v3 = torch.stack([(w * fi * sgn * y[:, :, None, None]).sum((-1, -2)).sum(-1) for fi in fe], -1)         # (E,3)
                th = torch.linalg.solve(M3 + 1e-8 * torch.eye(3, dtype=dt, device=dev), v3.unsqueeze(-1)).squeeze(-1)
                a[q] = em_damp * th[:, :2] + (1 - em_damp) * a[q]
                b[q] = em_damp * th[:, 2] + (1 - em_damp) * b[q]
        gnew = (Pm @ (zhat - omega).unsqueeze(-1)).squeeze(-1)
        dg = Pm @ (Sig - V) @ Pm
        gh = damp * gnew + (1 - damp) * gh
        # per-example PSD projection of -dg BEFORE summation (the matrix analogue of the scalar clamp
        # tau_s = (1 - tau_z/tau_p)/tau_p >= 1e-8 used by the diagonal GAMP); multi-branch posteriors can have
        # posterior covariance larger than the prior, and summing those contributions first cancels information
        Dm = -0.5 * (dg + dg.transpose(-1, -2))
        ev, U = torch.linalg.eigh(Dm)
        Dm = U @ torch.diag_embed(ev.clamp_min(1e-8)) @ U.transpose(-1, -2)
        A = torch.einsum('end,renkl->redkl', X2, Dm)
        Wi = W.transpose(-1, -2)                                                                                     # (R,E,d,K)
        B = torch.einsum('end,renk->redk', X, gh) + (A @ Wi.unsqueeze(-1)).squeeze(-1)
        Cn = torch.linalg.inv(I2 / v0 + A)
        Wn = (Cn @ B.unsqueeze(-1)).squeeze(-1).transpose(-1, -2)
        W = torch.nan_to_num(damp * Wn + (1 - damp) * W)
        Cc = damp * Cn + (1 - damp) * Cc
    return W, a, b


# ---------------------------------------------------------------- experiment
d = args.d; n = int(args.alpha * d); E = args.episodes
Wt, at, bt, X, y, Xq, yq = make(E, d, n, 1000, args.seed + 1000 * d + int(args.alpha))
gen = torch.Generator().manual_seed(args.seed + 7 + d)
W0 = (torch.randn(64, E, r, d, generator=gen, dtype=dt) / math.sqrt(d)).to(dev)
th0 = torch.rand(64, E, generator=gen, dtype=dt).to(dev) * 2 * math.pi
a0 = torch.stack([th0.cos(), th0.sin()], -1); b0 = torch.zeros(64, E, dtype=dt, device=dev)
init = lambda k: (W0[:k], a0[:k], b0[:k])


def em_inits(k):
    """k random W inits x 4 (a,b) inits: angle {pi/8, 3pi/8} x b0 {-0.15, +0.15}; a >= 0 w.l.o.g. (sign -> W)."""
    Ws, As, Bs = [], [], []
    for thv in [math.pi / 8, 3 * math.pi / 8]:
        for bv in [-0.15, 0.15]:
            Ws.append(0.1 * W0[:k]); As.append(torch.tensor([math.cos(thv), math.sin(thv)], dtype=dt, device=dev).expand(k, E, 2).clone())
            Bs.append(torch.full((k, E), bv, dtype=dt, device=dev))
    return torch.cat(Ws), torch.cat(As), torch.cat(Bs)


res, tm = {}, {}


def run(name, fn):
    if args.only is not None and name not in args.only:
        return
    torch.cuda.synchronize(); t0 = time.perf_counter(); res[name] = fn(); torch.cuda.synchronize()
    tm[name] = time.perf_counter() - t0
    print(f'  {name}: {tm[name]:.0f}s', flush=True)


run('pcalm_deepjoint_2x2+adam', lambda: polish(X, y, pick(X, y, cat([pcalm_deep_joint(X, y, init(2), args.T, .1, True, r2) for r2 in [10., 30.]]))))
run('pcalm_deepjoint_8x2+adam', lambda: polish(X, y, pick(X, y, cat([pcalm_deep_joint(X, y, init(8), args.T, .1, True, r2) for r2 in [10., 30.]]))))
run('nomult_deepjoint_2x2+adam', lambda: polish(X, y, pick(X, y, cat([pcalm_deep_joint(X, y, init(2), args.T, .1, False, r2) for r2 in [10., 30.]]))))


def oracle_full(k):
    Ws = [amp_full(X, y, 0.1 * W0[:k], at.unsqueeze(0).expand(k, -1, -1).clone(), bt.unsqueeze(0).expand(k, -1).clone(),
                   args.amp_T, dm, em=False)[0] for dm in [.3, .6]]
    Wc = torch.cat(Ws); Rr = Wc.shape[0]
    return pick(X, y, (Wc, at.unsqueeze(0).expand(Rr, -1, -1), bt.unsqueeze(0).expand(Rr, -1)))


run('amp_full_oracle_8x2+adam', lambda: polish(X, y, oracle_full(8)))


def oracle_diag(k):
    Wc = torch.cat([gamp_diag_oracle(X, y, at, bt, 0.1 * W0[:k], 200, dm) for dm in [.3, .6]]); Rr = Wc.shape[0]
    return pick(X, y, (Wc, at.unsqueeze(0).expand(Rr, -1, -1), bt.unsqueeze(0).expand(Rr, -1)))


run('gamp_diag_oracle_8x2+adam', lambda: polish(X, y, oracle_diag(8)))
for k in [1, 4]:
    run(f'amp_full_em_{4 * k}x2+adam', lambda: polish(X, y, pick(X, y, cat(
        [amp_full(X, y, *em_inits(k), args.amp_T, dm, em=True) for dm in [.3, .6]]))))
run('bp_adam_64x3', lambda: pick(X, y, cat([adam(X, y, init(64), 1000, lr) for lr in [.003, .01, .03]])))

vq = yq.var(-1); rows = []
for name, (W, a, b) in res.items():
    m = ((model(Xq, W, a, b)[0] - yq) ** 2).mean(-1) / vq
    Pw = torch.einsum('erd,ejd->ejr', Wt, W[0]); ov = (Pw.norm(dim=-1) / W[0].norm(dim=-1).clamp_min(1e-12)).mean(-1)
    rows.append(dict(d=d, alpha=args.alpha, n=n, method=name, success=float((m < .05).float().mean()),
                     nmse_mean=float(m.mean()), nmse_median=float(m.median()), subspace_overlap=float(ov.mean()),
                     seconds=tm[name], per_task_nmse=[float(v) for v in m], per_task_overlap=[float(v) for v in ov]))
print(f'd={d} alpha={args.alpha} n={n}: ' + '  '.join(f"{x['method']}={x['success']:.2f}/{x['subspace_overlap']:.2f}({x['seconds']:.0f}s)" for x in rows), flush=True)
(out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
