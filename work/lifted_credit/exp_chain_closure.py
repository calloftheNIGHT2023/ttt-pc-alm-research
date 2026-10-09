"""Chain closure (2026-10): controlled checks requested by the requirements audit (report 456).

  c1  R6 derivative: central finite differences of the outer loss in log(rho) and log(v) over a ladder of steps h, with
      bootstrap CIs over 4096 tasks, against backpropagation through the unrolled solver; hard branches and Gibbs
      temperatures tau in {0.5, 0.2, 0.05} (for tau > 0 the loss is smooth: autograd is exact; we report its variance).
  c2  R2 learner side: at matched chain count R, best-of-R (likelihood argmax) vs likelihood-weighted mixture, overall
      and split by whether the chains found competing modes (second-largest weight > 0.1).
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=4610)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
dev = args.device; DT = torch.float64
g = torch.Generator(device=dev).manual_seed(args.seed)
rn = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=DT)


def fold(m, s):
    s = torch.as_tensor(s, dtype=DT, device=dev).clamp_min(1e-12)
    return s * math.sqrt(2 / math.pi) * torch.exp(-m ** 2 / (2 * s ** 2)) + m * torch.erf(m / (s * math.sqrt(2)))


def local(y, a, rho, tau=None):
    sp = ((2 * y + rho * a) / (2 + rho)).clamp_min(0); sm = ((-2 * y + rho * a) / (2 + rho)).clamp_max(0)
    F = lambda s: (s.abs() - y) ** 2 + 0.5 * rho * (s - a) ** 2
    if tau is None:
        return torch.where(F(sp) <= F(sm), sp, sm)
    p = torch.sigmoid((F(sm) - F(sp)) / tau)
    return p * sp + (1 - p) * sm


def chol(K, mu):
    return torch.linalg.cholesky(K.transpose(1, 2) @ K + mu * torch.eye(K.shape[-1], dtype=DT, device=dev))


R = {}; t0 = time.perf_counter()

# ------------------------------------------------------------------ c1: derivative ladder
d = 32; n = 4 * d; B = 4096; res = []
for sigma in (0.0, 0.3):
    w = rn(B, d); w = w / w.norm(dim=-1, keepdim=True)
    K = rn(B, n, d); Kq = rn(B, 64, d)
    y = (torch.einsum('bnd,bd->bn', K, w) + sigma * rn(B, n)).abs()
    yq = (torch.einsum('bqd,bd->bq', Kq, w) + sigma * rn(B, 64)).abs()
    w0 = 0.1 * rn(B, d) / math.sqrt(d); L0 = chol(K, 1e-3)

    def per_task_loss(knob, phi, tau):
        rho = torch.exp(phi) if knob == 'log_rho' else torch.tensor(1.0, dtype=DT, device=dev)
        v = torch.exp(phi) if knob == 'log_v' else torch.tensor(0.0, dtype=DT, device=dev)
        lam = torch.zeros_like(y); ww = w0.clone()
        for _ in range(30):
            pred = torch.einsum('bnd,bd->bn', K, ww); s = local(y, pred - lam / rho, rho, tau)
            lam = (lam + rho * (s - pred)) / (1 + rho * v)
            ww = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L0).squeeze(-1)
        return ((torch.einsum('bqd,bd->bq', Kq, ww).abs() - yq) ** 2).mean(-1)          # per-task outer loss

    def boot_ci(x, reps=2000):
        gb = torch.Generator(device=dev).manual_seed(7)
        idx = torch.randint(len(x), (reps, len(x)), generator=gb, device=dev)
        m = x[idx].mean(-1); return [float(torch.quantile(m, .025)), float(torch.quantile(m, .975))]

    for knob, at in (('log_rho', 0.0), ('log_rho', math.log(10.0)), ('log_v', math.log(0.3)), ('log_v', 0.0)):
        for tau in (None, 0.5, 0.2, 0.05):
            phi = torch.tensor(at, dtype=DT, device=dev, requires_grad=True)
            lt = per_task_loss(knob, phi, tau)
            gt = torch.autograd.grad(lt.sum(), phi)[0] / B                          # mean pathwise derivative
            # per-task pathwise derivatives for the variance (vector-Jacobian with one-hot weights is too costly; use chunks)
            per = []
            for c in range(0, B, 1024):
                phi_c = torch.tensor(at, dtype=DT, device=dev, requires_grad=True)
                ltc = per_task_loss(knob, phi_c, tau)
                mask = torch.zeros(B, dtype=DT, device=dev); mask[c:c + 1024] = 1
                per.append(float(torch.autograd.grad((ltc * mask).sum(), phi_c)[0]) / 1024)
            per = torch.tensor(per, dtype=DT)
            ladder = {}
            with torch.no_grad():
                for h in (0.4, 0.2, 0.1, 0.05, 0.025, 0.0125):
                    lp = per_task_loss(knob, torch.tensor(at + h, dtype=DT, device=dev), tau)
                    lm = per_task_loss(knob, torch.tensor(at - h, dtype=DT, device=dev), tau)
                    fd = (lp - lm) / (2 * h)
                    ladder[str(h)] = dict(mean=float(fd.mean()), ci95=boot_ci(fd), frac_tasks_nonzero=float((fd.abs() > 1e-12).double().mean()))
            row = dict(sigma=sigma, knob=knob, at=at, local='hard' if tau is None else f'gibbs tau={tau}', loss=float(lt.mean()),
                       autograd_mean=float(gt), autograd_chunk_sd=float(per.std()), finite_difference=ladder)
            res.append(row)
            print('c1', sigma, knob, round(at, 3), row['local'], 'auto', f"{float(gt):+.4e}",
                  ' '.join(f"h{h}:{v['mean']:+.4f}[{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]" for h, v in ladder.items()), flush=True)
R['c1_derivative_ladder'] = res

# ------------------------------------------------------------------ c2: search vs averaging at matched R
res = []; d = 32; B = 1024; sigma = 0.1


def alm(K, y, w, iters, rho=1.0, mu=1e-3):
    L = chol(K, mu); lam = torch.zeros_like(y)
    for _ in range(iters):
        pred = torch.einsum('bnd,bd->bn', K, w); s = local(y, pred - lam / rho, rho)
        lam = lam + rho * (s - pred)
        w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L).squeeze(-1)
    return w


def em(K, y, w, iters, tau, mu=1e-3):
    L = chol(K, mu)
    for _ in range(iters):
        m = torch.einsum('bnd,bd->bn', K, w)
        w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, y * torch.tanh(y * m / tau)).unsqueeze(-1), L).squeeze(-1)
    return w


for nd in (2.0, 2.5, 3.0):
    n = int(nd * d)
    w = rn(B, d); w = w / w.norm(dim=-1, keepdim=True); K = rn(B, n, d); Kq = rn(B, 256, d)
    y = (torch.einsum('bnd,bd->bn', K, w) + sigma * rn(B, n)).abs()
    mt = torch.einsum('bqd,bd->bq', Kq, w); bayes = fold(mt, sigma)
    vy = (mt ** 2 + sigma ** 2 - bayes ** 2).mean(-1) + bayes.var(-1)
    preds, loglik = [], []
    for r in range(16):
        wr = em(K, y, alm(K, y, rn(B, d) / math.sqrt(d), 200), 100, sigma ** 2)
        preds.append(fold(torch.einsum('bqd,bd->bq', Kq, wr), sigma))
        loglik.append(-((y - torch.einsum('bnd,bd->bn', K, wr).abs()) ** 2).sum(-1) / (2 * sigma ** 2))
    P = torch.stack(preds, -1); LL = torch.stack(loglik, -1)
    for Rn in (1, 2, 4, 8, 16):
        Pr, Lr = P[..., :Rn], LL[:, :Rn]
        wts = torch.softmax(Lr, -1)
        best = Pr.gather(-1, Lr.argmax(-1)[:, None, None].expand(-1, Pr.shape[1], 1)).squeeze(-1)
        mix = (Pr * wts[:, None, :]).sum(-1)
        ex_b = ((best - bayes) ** 2).mean(-1) / vy; ex_m = ((mix - bayes) ** 2).mean(-1) / vy
        second = wts.sort(-1, descending=True).values[:, 1] if Rn > 1 else torch.zeros(B, dtype=DT, device=dev)
        comp = second > 0.1
        row = dict(n_over_d=nd, R=Rn, best_of_R=float(ex_b.mean()), mixture=float(ex_m.mean()),
                   frac_competing=float(comp.double().mean()),
                   competing_best=float(ex_b[comp].mean()) if comp.any() else None, competing_mixture=float(ex_m[comp].mean()) if comp.any() else None,
                   unique_best=float(ex_b[~comp].mean()), unique_mixture=float(ex_m[~comp].mean()))
        res.append(row)
        print('c2', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
R['c2_search_vs_averaging'] = res

R['seconds'] = time.perf_counter() - t0
(out / 'results.json').write_text(json.dumps(R, indent=1))
print('DONE', R['seconds'])
