"""Report 451 / paper Sec. 6: controlled verification of requirements R1-R5 for inference-time self-iterating models.

Oracle keys (no outer loop): k ~ N(0, I_d), w* uniform on the sphere, y = |k.w* + sigma*eps| (noise INSIDE the |.|).
Inner loops (vectorised over a batch of B tasks):
  alm   PC-ALM: two-branch local inversion of (|s|-y)^2 + rho/2 (s-a)^2 (hard, or Gibbs posterior at temperature tau),
        leaky multiplier lam <- (lam + rho r)/(1 + rho v), ridge LS parameter step.  v = 0 is the original PC-ALM.
  em    posterior inversion s = y tanh(y (k.w)/tau) + ridge LS (tau = sigma^2 is the Bayes posterior).
  gd    full-batch gradient descent on the ERM sum (|k.w| - y)^2 (the TTT gradient update).
Experiments:
  r1  start AT w*, run to convergence: the hard (ERM-stationary) inner loop drifts to the norm alpha* w* predicted by
      Prop. 1, alpha* = E[|m| F_sigma(m)] / E[m^2]; the posterior loop stays at w*.
  r2  random init, sweep leak v x noise sigma: v = 0 is best in the exact regime, the optimum moves up with sigma.
  r3  in-context noise estimate: raw RSS/n vs df-corrected RSS/(n-df) against the true residual variance.
  r4  near the identifiability threshold n ~ 2d: single chain vs likelihood-weighted mixture of R chains.
  r5  outer-loop derivative of the query loss w.r.t. solver hyper-parameters (log rho, log v): autograd (pathwise)
      vs central finite differences on the same tasks.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, default=32)
ap.add_argument('--B', type=int, default=512)
ap.add_argument('--exps', nargs='+', default=['r1', 'r2', 'r3', 'r4', 'r5'])
ap.add_argument('--seed', type=int, default=451)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
dev = args.device; d = args.d
torch.backends.cuda.matmul.allow_tf32 = False
DT = torch.float64


def sample(B, n, sigma, gen, nq=256):
    w = torch.randn(B, d, generator=gen, device=dev, dtype=DT); w = w / w.norm(dim=-1, keepdim=True)
    K = torch.randn(B, n, d, generator=gen, device=dev, dtype=DT); Kq = torch.randn(B, nq, d, generator=gen, device=dev, dtype=DT)
    y = (torch.einsum('bnd,bd->bn', K, w) + sigma * torch.randn(B, n, generator=gen, device=dev, dtype=DT)).abs()
    return K, y, Kq, w


def fold(m, sd):
    sd = torch.as_tensor(sd, dtype=DT, device=m.device).clamp_min(1e-12)
    return sd * math.sqrt(2 / math.pi) * torch.exp(-m ** 2 / (2 * sd ** 2)) + m * torch.erf(m / (sd * math.sqrt(2)))


def local(y, a, rho, tau=None):
    sp = ((2 * y + rho * a) / (2 + rho)).clamp_min(0); sm = ((-2 * y + rho * a) / (2 + rho)).clamp_max(0)
    F = lambda s: (s.abs() - y) ** 2 + 0.5 * rho * (s - a) ** 2
    if tau is None:
        return torch.where(F(sp) <= F(sm), sp, sm)
    p = torch.sigmoid((F(sm) - F(sp)) / tau)
    return p * sp + (1 - p) * sm


def ridge_chol(K, mu):
    return torch.linalg.cholesky(K.transpose(1, 2) @ K + mu * torch.eye(K.shape[-1], dtype=K.dtype, device=K.device))


def alm(K, y, w, iters, rho=1.0, mu=1e-3, v=0.0, tau=None):
    L = ridge_chol(K, mu); lam = torch.zeros_like(y)
    v = torch.as_tensor(v, dtype=K.dtype, device=K.device)
    for _ in range(iters):
        pred = torch.einsum('bnd,bd->bn', K, w)
        s = local(y, pred - lam / rho, rho, tau)
        lam = (lam + rho * (s - pred)) / (1 + rho * v) if torch.isfinite(v).all() else torch.zeros_like(lam)
        w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L).squeeze(-1)
    return w


def em(K, y, w, iters, tau, mu=1e-3):
    L = ridge_chol(K, mu)
    for _ in range(iters):
        m = torch.einsum('bnd,bd->bn', K, w)
        s = y * torch.tanh(y * m / tau)
        w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s).unsqueeze(-1), L).squeeze(-1)
    return w


def gd(K, y, w, iters, eta=0.1):
    n = K.shape[1]
    for _ in range(iters):
        m = torch.einsum('bnd,bd->bn', K, w)
        w = w - eta * 2 / n * torch.einsum('bn,bnd->bd', (m.abs() - y) * torch.sign(m), K)
    return w


def metrics(w, wt, Kq, sigma, readout_sd=None):
    """parameter error (sign-invariant), norm ratio, overlap, excess query risk over the Bayes predictor F_sigma(k.w*)."""
    err = torch.minimum((w - wt).norm(dim=-1), (w + wt).norm(dim=-1))
    ov = (w * wt).sum(-1).abs() / w.norm(dim=-1).clamp_min(1e-12)
    mq, mt = torch.einsum('bqd,bd->bq', Kq, w), torch.einsum('bqd,bd->bq', Kq, wt)
    bayes = fold(mt, sigma) if sigma > 0 else mt.abs()
    vy = (mt ** 2 + sigma ** 2 - bayes ** 2).mean(-1) + bayes.var(-1)          # Var(y_q) over the query distribution
    pred_abs = mq.abs(); pred_fold = fold(mq, sigma) if sigma > 0 else mq.abs()
    ex = lambda p: ((p - bayes) ** 2).mean(-1) / vy
    return dict(param_err=float(err.mean()), param_err_median=float(err.median()), norm_ratio=float(w.norm(dim=-1).mean()),
                overlap=float(ov.mean()), success=float((ov > 0.95).double().mean()),
                excess_abs_readout=float(ex(pred_abs).mean()), excess_fold_readout=float(ex(pred_fold).mean()))


def alpha_star(sigma, N=4_000_000):
    g = torch.Generator(device=dev).manual_seed(7)
    m = torch.randn(N, generator=g, device=dev, dtype=DT)
    return float((m.abs() * fold(m, sigma)).mean() / (m ** 2).mean())


R = {}
gen = torch.Generator(device=dev).manual_seed(args.seed)
t0 = time.perf_counter()

if 'r1' in args.exps:              # Prop. 1: consistency of the inner fixed point
    res = []
    for sigma in (0.25, 0.5):
        a_star = alpha_star(sigma)
        for nd in (4, 8, 16, 32, 64):
            K, y, Kq, wt = sample(args.B, nd * d, sigma, gen)
            row = dict(sigma=sigma, n_over_d=nd, alpha_star_theory=a_star)
            row['hard_alm_v0'] = metrics(alm(K, y, wt.clone(), 400), wt, Kq, sigma)
            row['gd_erm'] = metrics(gd(K, y, wt.clone(), 2000, eta=0.05), wt, Kq, sigma)
            row['em_posterior_oracle_tau'] = metrics(em(K, y, wt.clone(), 400, sigma ** 2), wt, Kq, sigma)
            res.append(row)
            print(f"r1 sigma={sigma} n={nd}d alpha*={a_star:.4f} | hard norm {row['hard_alm_v0']['norm_ratio']:.4f} err {row['hard_alm_v0']['param_err']:.4f}"
                  f" | GD norm {row['gd_erm']['norm_ratio']:.4f} | EM norm {row['em_posterior_oracle_tau']['norm_ratio']:.4f} err {row['em_posterior_oracle_tau']['param_err']:.4f}", flush=True)
    R['r1'] = res

if 'r2' in args.exps:              # Thm. 2: leak sweep x noise, random init
    res = []
    for sigma in (0.0, 0.1, 0.3, 0.5):
        for nd in (3, 4):
            K, y, Kq, wt = sample(args.B, nd * d, sigma, gen)
            w0 = 0.1 * torch.randn(args.B, d, generator=gen, device=dev, dtype=DT) / math.sqrt(d)
            for v in (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, float('inf')):
                mt = metrics(alm(K, y, w0.clone(), 300, v=v), wt, Kq, sigma)
                res.append(dict(sigma=sigma, n_over_d=nd, v=v, **mt))
                print(f"r2 sigma={sigma} n={nd}d v={v}: success {mt['success']:.3f} overlap {mt['overlap']:.3f} excess(fold) {mt['excess_fold_readout']:.4f}", flush=True)
    R['r2'] = res

if 'r3' in args.exps:              # Prop. 3: in-context noise estimate
    res = []
    for sigma in (0.25, 0.5):
        for nd in (1.5, 2, 3, 4, 8, 16):
            n = int(nd * d)
            K, y, Kq, wt = sample(args.B, n, sigma, gen)
            mu = 1e-3
            w = em(K, y, wt.clone(), 300, sigma ** 2, mu=mu)
            rss = ((y - torch.einsum('bnd,bd->bn', K, w).abs()) ** 2).mean(-1)
            G = K.transpose(1, 2) @ K
            df = d - mu * torch.cholesky_inverse(ridge_chol(K, mu)).diagonal(dim1=1, dim2=2).sum(-1)
            Kf = torch.randn(args.B, 4096, d, generator=gen, device=dev, dtype=DT)
            yf = (torch.einsum('bnd,bd->bn', Kf, wt) + sigma * torch.randn(args.B, 4096, generator=gen, device=dev, dtype=DT)).abs()
            true_r2 = ((yf - torch.einsum('bnd,bd->bn', Kf, wt).abs()) ** 2).mean(-1)   # residual variance at the truth
            row = dict(sigma=sigma, n_over_d=nd, df_mean=float(df.mean()),
                       raw_ratio=float((rss / true_r2).mean()), corrected_ratio=float((rss * n / (n - df).clamp_min(1) / true_r2).mean()),
                       theory_raw_ratio=float((1 - df / n).mean()))
            res.append(row)
            print(f"r3 sigma={sigma} n={nd}d: raw {row['raw_ratio']:.3f} (theory 1-df/n {row['theory_raw_ratio']:.3f})  corrected {row['corrected_ratio']:.3f}", flush=True)
    R['r3'] = res

if 'r4' in args.exps:              # Prop. 4: posterior mixture near the identifiability threshold
    res = []
    sigma = 0.1
    for nd in (1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
        n = int(nd * d)
        K, y, Kq, wt = sample(args.B, n, sigma, gen)
        Rn = 16
        W, lik = [], []
        for r in range(Rn):
            w0 = torch.randn(args.B, d, generator=gen, device=dev, dtype=DT) / math.sqrt(d)
            w = em(K, y, alm(K, y, w0, 200), 100, sigma ** 2)          # PC-ALM chain, then posterior refinement
            W.append(w)
            r2 = ((y - torch.einsum('bnd,bd->bn', K, w).abs()) ** 2).sum(-1)
            lik.append(-r2 / (2 * sigma ** 2))
        lik = torch.stack(lik, -1); wts = torch.softmax(lik, -1)
        mt = torch.einsum('bqd,bd->bq', Kq, wt); bayes = fold(mt, sigma)
        vy = (mt ** 2 + sigma ** 2 - bayes ** 2).mean(-1) + bayes.var(-1)
        preds = torch.stack([fold(torch.einsum('bqd,bd->bq', Kq, w), sigma) for w in W], -1)
        single = ((preds[..., 0] - bayes) ** 2).mean(-1) / vy
        best = ((preds.gather(-1, lik.argmax(-1)[:, None, None].expand(-1, preds.shape[1], 1)).squeeze(-1) - bayes) ** 2).mean(-1) / vy
        mix = (((preds * wts[:, None, :]).sum(-1) - bayes) ** 2).mean(-1) / vy
        row = dict(n_over_d=nd, sigma=sigma, single_chain=float(single.mean()), best_of_R=float(best.mean()), posterior_mixture=float(mix.mean()),
                   single_median=float(single.median()), mixture_median=float(mix.median()), R=Rn)
        res.append(row)
        print(f"r4 n={nd}d: single {row['single_chain']:.4f}  best-of-{Rn} {row['best_of_R']:.4f}  mixture {row['posterior_mixture']:.4f}", flush=True)
    R['r4'] = res

if 'r5' in args.exps:              # Prop. 5: pathwise (autograd) vs true derivative of the outer loss w.r.t. solver knobs
    res = []
    for sigma in (0.0, 0.3):
        for knob in ('log_rho', 'log_v'):
            for base in (0.0, math.log(0.3)) if knob == 'log_v' else (0.0, math.log(10.0)):
                g5 = torch.Generator(device=dev).manual_seed(args.seed + 55)
                K, y, Kq, wt = sample(4 * args.B, 4 * d, sigma, g5)
                w0 = 0.1 * torch.randn(4 * args.B, d, generator=g5, device=dev, dtype=DT) / math.sqrt(d)
                yq = (torch.einsum('bqd,bd->bq', Kq, wt) + sigma * torch.randn(*Kq.shape[:2], generator=g5, device=dev, dtype=DT)).abs()

                def loss(phi, tau):
                    rho = phi.exp() if knob == 'log_rho' else torch.tensor(1.0, dtype=DT, device=dev)
                    v = phi.exp() if knob == 'log_v' else torch.tensor(0.0, dtype=DT, device=dev)
                    L = ridge_chol(K, 1e-3); lam = torch.zeros_like(y); w = w0.clone()
                    for _ in range(30):
                        pred = torch.einsum('bnd,bd->bn', K, w)
                        s = local(y, pred - lam / rho, rho, tau)
                        lam = (lam + rho * (s - pred)) / (1 + rho * v)
                        w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L).squeeze(-1)
                    return ((torch.einsum('bqd,bd->bq', Kq, w).abs() - yq) ** 2).mean()
                for tau in (None, 0.05):
                    phi = torch.tensor(base, dtype=DT, device=dev, requires_grad=True)
                    l0 = loss(phi, tau); l0.backward(); auto = float(phi.grad)
                    fd = {}
                    for h in (0.2, 0.05):
                        with torch.no_grad():
                            fd[str(h)] = float((loss(torch.tensor(base + h, dtype=DT, device=dev), tau) - loss(torch.tensor(base - h, dtype=DT, device=dev), tau)) / (2 * h))
                    row = dict(sigma=sigma, knob=knob, at=base, local='hard' if tau is None else f'gibbs tau={tau}', loss=float(l0), autograd=auto, finite_diff=fd)
                    res.append(row)
                    print(f"r5 sigma={sigma} {knob}@{base:.2f} {row['local']}: loss {float(l0):.4f} autograd {auto:+.4e}  FD(0.2) {fd['0.2']:+.4e}  FD(0.05) {fd['0.05']:+.4e}", flush=True)
    R['r5'] = res

R['seconds'] = time.perf_counter() - t0
(out / 'results.json').write_text(json.dumps(R, indent=1))
print('DONE', R['seconds'])
