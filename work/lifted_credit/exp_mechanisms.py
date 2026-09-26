"""Paper Sec. 'Mechanism experiments': quick, exact theory-vs-measurement checks of the six requirements.

  m1  R1 floor: bases with known noise ratio nu; the Bayes oracle F_s(u.k) must sit ON floor(nu), and every learner
      (self-iteration, TTT gradient update, quadratic closed form, oracle |u.k| probe) must sit above it.
  m2  R2 identifiability: noiseless context, u ~ N(0, I/d). Even with all signs revealed, NMSE >= 1 - n/d for n < d.
  m3  R3 bias: ERM fixed point norm alpha*(sigma) = 1 + c_sigma vs measurement; posterior inversion stays at 1.
  m4  R5 fixed points: the leaky ALM fixed point equals ridge LS with mu_v = (rho mu / 2)(1 + 2v/(1+rho v)).
  m5  Theorem 1: fixed-feature floor (d-1)/(d+2) (1 - (M-d)_+/N_3) vs linear and random-ReLU features, oracle fit.
"""
import argparse
import json
import math
import time
from math import comb
from pathlib import Path
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
ap.add_argument('--seed', type=int, default=452)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
dev = args.device; DT = torch.float64
torch.backends.cuda.matmul.allow_tf32 = False
g = torch.Generator(device=dev).manual_seed(args.seed)
rn = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=DT)


def fold(m, s):
    s = torch.as_tensor(s, dtype=DT, device=dev).clamp_min(1e-12)
    return s * math.sqrt(2 / math.pi) * torch.exp(-m ** 2 / (2 * s ** 2)) + m * torch.erf(m / (s * math.sqrt(2)))


def local(y, a, rho):
    sp = ((2 * y + rho * a) / (2 + rho)).clamp_min(0); sm = ((-2 * y + rho * a) / (2 + rho)).clamp_max(0)
    F = lambda s: (s.abs() - y) ** 2 + 0.5 * rho * (s - a) ** 2
    return torch.where(F(sp) <= F(sm), sp, sm)


def chol(K, mu):
    return torch.linalg.cholesky(K.transpose(1, 2) @ K + mu * torch.eye(K.shape[-1], dtype=DT, device=dev))


def alm(K, y, w, iters, rho=1.0, mu=1e-3, v=0.0):
    L = chol(K, mu); lam = torch.zeros_like(y)
    for _ in range(iters):
        pred = torch.einsum('bnd,bd->bn', K, w); s = local(y, pred - lam / rho, rho)
        lam = (lam + rho * (s - pred)) / (1 + rho * v)
        w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L).squeeze(-1)
    return w, s


def em(K, y, w, iters, mu=1e-3, tau=None):
    """posterior inversion; tau=None: df-corrected in-context noise estimate (R4)."""
    n, d = K.shape[1], K.shape[2]; L = chol(K, mu)
    dfv = d - mu * torch.cholesky_inverse(L).diagonal(dim1=1, dim2=2).sum(-1)
    for _ in range(iters):
        m = torch.einsum('bnd,bd->bn', K, w)
        t = tau if tau is not None else (((y - m.abs()) ** 2).sum(-1) / (n - dfv).clamp_min(1)).clamp_min(1e-4)[:, None]
        w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, y * torch.tanh(y * m / t)).unsqueeze(-1), L).squeeze(-1)
    m = torch.einsum('bnd,bd->bn', K, w)
    r2 = (((y - m.abs()) ** 2).sum(-1) / (n - dfv).clamp_min(1)).clamp_min(1e-6)
    return w, r2


def self_iteration(K, y, R=8, iters=150):
    """Algorithm-1 style: R chains (lifted ALM then posterior refinement), likelihood-weighted mixture (R2)."""
    B, n, d = K.shape; W, R2 = [], []
    for _ in range(R):
        w0 = rn(B, d) / math.sqrt(d)
        w, _ = alm(K, y, w0, iters)
        w, r2 = em(K, y, w, iters)
        W.append(w); R2.append(r2)
    R2 = torch.stack(R2, -1)
    wts = torch.softmax(-0.5 * n * (R2 / R2.min(-1, keepdim=True).values - 1), -1)
    return W, R2, wts


def gd(K, y, w, iters, eta=0.05):
    n = K.shape[1]
    for _ in range(iters):
        m = torch.einsum('bnd,bd->bn', K, w)
        w = w - eta * 2 / n * torch.einsum('bn,bnd->bd', (m.abs() - y) * torch.sign(m), K)
    return w


def quad_head(K, y, Kq, mu=1e-3):
    d = K.shape[-1]; iu = torch.triu_indices(d, d, device=dev)
    f = lambda M: torch.cat([(M.unsqueeze(-1) * M.unsqueeze(-2))[..., iu[0], iu[1]], torch.ones(*M.shape[:2], 1, device=dev, dtype=DT)], -1)
    F, Fq = f(K), f(Kq); sc = F.pow(2).sum(-1).mean(-1).sqrt()[:, None, None]; F, Fq = F / sc, Fq / sc
    a = torch.linalg.solve(F @ F.transpose(1, 2) + mu * torch.eye(K.shape[1], device=dev, dtype=DT), (y ** 2).unsqueeze(-1)).squeeze(-1)
    return torch.einsum('bqp,bnp,bn->bq', Fq, F, a).clamp_min(0).sqrt()


R = {}; t0 = time.perf_counter()

# ------------------------------------------------------------------ m1: R1 floor
res = []; d = 16; n = 4 * d; B = 512; nq = 512
for nu in (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
    u = rn(B, d); u = u / u.norm(dim=-1, keepdim=True) * math.sqrt(1 - nu)
    K = math.sqrt(2) * rn(B, n, d); Kq = math.sqrt(2) * rn(B, nq, d)            # keys = differences of two items
    s = math.sqrt(2 * nu)
    y = (torch.einsum('bnd,bd->bn', K, u) + s * rn(B, n)).abs()
    mq = torch.einsum('bqd,bd->bq', Kq, u); yq = (mq + s * rn(B, nq)).abs()
    vy = 2 - 4 / math.pi                                                        # Var(|z|), z ~ N(0, 2)
    nm = lambda p: float(((p - yq) ** 2).mean() / vy)
    mm = math.sqrt(2 * (1 - nu)) * rn(2_000_000); floor = float((mm ** 2 + s ** 2 - fold(mm, s) ** 2).mean() / vy) if nu > 0 else 0.0
    W, R2, wts = self_iteration(K, y)
    si = sum(wts[:, r:r + 1] * fold(torch.einsum('bqd,bd->bq', Kq, W[r]), R2[:, r:r + 1].sqrt()) for r in range(len(W)))
    wg = gd(K, y, rn(B, d) / math.sqrt(d), 300)
    row = dict(nu=nu, floor_theory=floor, bayes_oracle=nm(fold(mq, s) if nu > 0 else mq.abs()), oracle_probe_abs=nm(mq.abs()),
               self_iteration=nm(si), ttt_gd_full=nm(torch.einsum('bqd,bd->bq', Kq, wg).abs()), quad_closed_form=nm(quad_head(K, y, Kq)))
    res.append(row); print('m1', {k: round(v, 4) for k, v in row.items()}, flush=True)
R['m1_floor'] = res

# ------------------------------------------------------------------ m2: R2 identifiability
res = []; d = 32; B = 256; nq = 256
for nd in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0):
    n = int(nd * d)
    u = rn(B, d) / math.sqrt(d); K = rn(B, n, d); Kq = rn(B, nq, d)
    y = torch.einsum('bnd,bd->bn', K, u).abs(); mq = torch.einsum('bqd,bd->bq', Kq, u); yq = mq.abs()
    vy = float(yq.var())
    # sign-revealed Bayes predictor (strictly more information than any learner has): posterior of mq given K u
    Ku = torch.einsum('bnd,bd->bn', K, u)
    Sig = torch.eye(d, dtype=DT, device=dev) / d
    G = K @ Sig @ K.transpose(1, 2) + 1e-10 * torch.eye(n, dtype=DT, device=dev)
    mean_u = (Sig @ K.transpose(1, 2) @ torch.linalg.solve(G, Ku.unsqueeze(-1))).squeeze(-1)
    Pcov = Sig - Sig @ K.transpose(1, 2) @ torch.linalg.solve(G, K @ Sig)
    mu_q = torch.einsum('bqd,bd->bq', Kq, mean_u); v_q = torch.einsum('bqd,bde,bqe->bq', Kq, Pcov, Kq).clamp_min(1e-14)
    bayes_signs = fold(mu_q, v_q.sqrt())
    W, R2, wts = self_iteration(K, y, R=16, iters=200)
    si = sum(wts[:, r:r + 1] * torch.einsum('bqd,bd->bq', Kq, W[r]).abs() for r in range(len(W)))
    wg = gd(K, y, rn(B, d) / math.sqrt(d), 400)
    nm = lambda p: float(((p - yq) ** 2).mean() / vy)
    row = dict(n_over_d=nd, lower_bound_theory=max(0.0, 1 - nd), bayes_with_signs=nm(bayes_signs), self_iteration=nm(si),
               ttt_gd_full=nm(torch.einsum('bqd,bd->bq', Kq, wg).abs()))
    res.append(row); print('m2', {k: round(v, 4) for k, v in row.items()}, flush=True)
R['m2_identifiability'] = res

# ------------------------------------------------------------------ m3: R3 bias alpha*(sigma)
res = []; d = 32; n = 64 * d; B = 128
for sig in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
    mm = rn(4_000_000); a_star = float((mm.abs() * fold(mm, sig)).mean() / (mm ** 2).mean())
    w = rn(B, d); w = w / w.norm(dim=-1, keepdim=True)
    K = rn(B, n, d); y = (torch.einsum('bnd,bd->bn', K, w) + sig * rn(B, n)).abs()
    we = gd(K, y, w.clone(), 2500, eta=0.05)
    wp, _ = em(K, y, w.clone(), 400, tau=sig ** 2)
    row = dict(sigma=sig, alpha_star_theory=a_star, erm_norm=float(we.norm(dim=-1).mean()), erm_norm_se=float(we.norm(dim=-1).std() / math.sqrt(B)),
               posterior_norm=float(wp.norm(dim=-1).mean()))
    res.append(row); print('m3', {k: round(v, 5) for k, v in row.items()}, flush=True)
R['m3_bias'] = res

# ------------------------------------------------------------------ m4: R5 leaky-ALM fixed points = ridge with mu_v
res = []; d = 16; n = 3 * d; B = 128; rho = 1.0; mu = 2.0
u = rn(B, d); u = u / u.norm(dim=-1, keepdim=True); K = rn(B, n, d); y = torch.einsum('bnd,bd->bn', K, u).abs()
for v in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0):
    w, s = alm(K, y, u.clone(), 4000, rho=rho, mu=mu, v=v)
    b = torch.sign(s); mu_v = rho * mu / 2 * (1 + 2 * v / (1 + rho * v))
    w_pred = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, b * y).unsqueeze(-1), chol(K, mu_v)).squeeze(-1)
    w_naive = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, b * y).unsqueeze(-1), chol(K, rho * mu / 2)).squeeze(-1)
    w_next, _ = alm(K, y, w.clone(), 1, rho=rho, mu=mu, v=v)
    row = dict(v=v, mu_v=mu_v, rel_err_vs_theorem=float(((w - w_pred).norm(dim=-1) / w_pred.norm(dim=-1)).max()),
               rel_diff_vs_unleaked_ridge=float(((w - w_naive).norm(dim=-1) / w_naive.norm(dim=-1)).mean()),
               fixed_point_residual=float(((w_next - w).norm(dim=-1) / w.norm(dim=-1)).max()))
    res.append(row); print('m4', row, flush=True)
R['m4_fixed_points'] = res

# ------------------------------------------------------------------ m5: Theorem 1 fixed-feature floor
res = []
He3 = lambda z: (z ** 3 - 3 * z) / math.sqrt(6)
for d in (8, 16, 32):
    N3 = comb(d + 2, 3) - d
    U = rn(64, d); U = U / U.norm(dim=-1, keepdim=True)                     # 64 tasks
    for kind, Ms in (('linear', [d]), ('random_relu', [d, 4 * d, 16 * d, 64 * d])):
        for M in Ms:
            if M > 2048:
                continue
            nfit = max(20 * M, 4000); X = rn(nfit, d); Xt = rn(8192, d)
            if kind == 'linear':
                P, Pt = X, Xt
            else:
                Wf = rn(d, M) / math.sqrt(d); bf = 0.5 * rn(M)
                P, Pt = torch.relu(X @ Wf + bf), torch.relu(Xt @ Wf + bf)
            Y = He3(X @ U.T); Yt = He3(Xt @ U.T)
            A = torch.linalg.solve(P.T @ P + 1e-6 * torch.eye(P.shape[1], dtype=DT, device=dev), P.T @ Y)
            mse = float(((Pt @ A - Yt) ** 2).mean())
            lb = (d - 1) / (d + 2) * max(0.0, 1 - max(0, M - d) / N3)
            row = dict(d=d, features=kind, M=M, N3=N3, lower_bound_theory=lb, measured_nmse=mse)
            res.append(row); print('m5', row, flush=True)
R['m5_fixed_features'] = res

R['seconds'] = time.perf_counter() - t0
(out / 'results.json').write_text(json.dumps(R, indent=1))
print('DONE', R['seconds'])
