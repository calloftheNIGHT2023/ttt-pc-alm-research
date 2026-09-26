"""Mechanism checks, part b: m4 run to convergence; m5 with the optimal degree-1 features h_i(x) = x_i(|x|^2 - d - 2)."""
import json, math, sys, time
from math import comb
from pathlib import Path
import torch
exec(open('exp_mechanisms.py').read().split("R = {}; t0")[0].replace("args = ap.parse_args()", "args = ap.parse_args(sys.argv[1:])").replace("out.mkdir(parents=True, exist_ok=False)", "out.mkdir(parents=True, exist_ok=True)"))
R = {}
res = []; d = 16; n = 3 * d; B = 128; rho = 1.0; mu = 2.0
u = rn(B, d); u = u / u.norm(dim=-1, keepdim=True); K = rn(B, n, d); y = torch.einsum('bnd,bd->bn', K, u).abs()
for v in (0.0, 0.1, 0.3, 1.0, 3.0, 10.0):
    L = chol(K, mu); lam = torch.zeros_like(y); w = u.clone(); hist = []
    for it in range(60000):
        pred = torch.einsum('bnd,bd->bn', K, w); s = local(y, pred - lam / rho, rho)
        lam = (lam + rho * (s - pred)) / (1 + rho * v)
        w_new = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L).squeeze(-1)
        stept = (w_new - w).norm(dim=-1) / w.norm(dim=-1); step = float(stept.max()); w = w_new
        if step < 1e-14: break
    b = torch.sign(s); mu_v = rho * mu / 2 * (1 + 2 * v / (1 + rho * v))
    w_pred = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, b * y).unsqueeze(-1), chol(K, mu_v)).squeeze(-1)
    w_naive = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, b * y).unsqueeze(-1), chol(K, rho * mu / 2)).squeeze(-1)
    conv = stept < 1e-12; errt = (w - w_pred).norm(dim=-1) / w_pred.norm(dim=-1)
    row = dict(v=v, mu_v=mu_v, iterations=it + 1, frac_converged=float(conv.double().mean()),
               rel_err_converged_max=float(errt[conv].max()) if conv.any() else None, rel_err_vs_theorem=float(errt.max()),
               rel_diff_vs_unleaked_ridge=float(((w - w_naive).norm(dim=-1) / w_naive.norm(dim=-1)).mean()))
    res.append(row); print('m4', row, flush=True)
R['m4_fixed_points'] = res
res = []
He3 = lambda z: (z ** 3 - 3 * z) / math.sqrt(6)
for d in ():
    N3 = comb(d + 2, 3) - d; U = rn(64, d); U = U / U.norm(dim=-1, keepdim=True)
    nfit = 200000 if d <= 32 else 400000; X = rn(nfit, d); Xt = rn(20000, d)
    h = lambda Z: Z * ((Z ** 2).sum(-1, keepdim=True) - d - 2)
    P, Pt = h(X), h(Xt); Y = He3(X @ U.T); Yt = He3(Xt @ U.T)
    A = torch.linalg.solve(P.T @ P, P.T @ Y); mse = float(((Pt @ A - Yt) ** 2).mean() / (Yt ** 2).mean())
    row = dict(d=d, features='optimal_degree1', M=d, lower_bound_theory=(d - 1) / (d + 2), measured_nmse=mse)
    res.append(row); print('m5', row, flush=True)
R['m5_optimal_features'] = res
(out / 'results_b_m4.json').write_text(json.dumps(R, indent=1)); print('DONE')
