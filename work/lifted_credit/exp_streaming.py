"""Exp D (report 440): TTT-style streaming inner loop on the He_3 single-index memory.

Context arrives in chunks of size b. After each chunk every method may spend K inner iterations, then
predicts the held-out queries (evaluator only). Methods:
  ttt_gd_k1        : official TTT inner rule, one GD step per chunk on the new chunk (lr grid)
  ttt_gd_kK        : K GD steps per chunk on the new chunk
  replay_adam_kK   : K Adam steps per chunk on all data seen so far (lr grid, R restarts)
  replay_lm_kK     : K Levenberg-Marquardt steps per chunk on all data seen so far (R restarts)
  pcalm_kK         : persistent activities + multipliers for all seen examples; K lifted sweeps per chunk
Rotational symmetry of the task family makes w=0 (or any isotropic draw) the meta-optimal TTT init,
so an outer loop cannot pre-break the symmetry; all methods start from the same isotropic draw.
"""
import argparse
import json
import math
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, default=64)
ap.add_argument('--chunk', type=int, default=16)
ap.add_argument('--nmax', type=int, default=768)
ap.add_argument('--K', type=int, default=10)
ap.add_argument('--episodes', type=int, default=64)
ap.add_argument('--restarts', type=int, default=8)
ap.add_argument('--nq', type=int, default=1000)
ap.add_argument('--sigma', type=float, default=0.1)
ap.add_argument('--seed', type=int, default=440401)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.He3
E, d, b, K, R = args.episodes, args.d, args.chunk, args.K, args.restarts
task = C.make_tasks(E, d, args.nmax, args.nq, link, args.sigma, args.seed, args.device, dt)
X, y, Xq, yq, u = task['X'], task['y'], task['Xq'], task['yq'], task['u']
W0 = C.init_w(E, d, R, args.seed + 1, args.device, dt)
grid = torch.linspace(-5, 5, 241, dtype=dt, device=args.device)
eye = torch.eye(d, dtype=dt, device=args.device)


def qmse(W):  # W (R,E,d) -> choose by support loss on seen data later; here (E,d)
    return C.query_mse_w(link, Xq, yq, W)


state = {}
state['ttt1'] = {lr: W0.clone() for lr in [0.01, 0.03, 0.1, 0.3]}
state['tttK'] = {lr: W0.clone() for lr in [0.01, 0.03, 0.1, 0.3]}
state['adam'] = {lr: dict(W=W0.clone(), m=torch.zeros_like(W0), v=torch.zeros_like(W0), t=0) for lr in [0.01, 0.03, 0.1]}
state['lm'] = dict(W=W0.clone(), mu=torch.ones(R, E, dtype=dt, device=args.device))
state['pcalm'] = {rho: dict(W=W0[:1].clone(), lam=torch.zeros(1, E, 0, dtype=dt, device=args.device)) for rho in [0.1, 1.0]}
tm = C.Timer(); rows = []


def gd_steps(W, Xc, yc, lr, steps):
    for _ in range(steps):
        s = torch.einsum('end,red->ren', Xc, W)
        W = W - lr * 2 / Xc.shape[1] * torch.einsum('ren,end->red', (link.f(s) - yc) * link.df(s), Xc)
        W = torch.nan_to_num(W).clamp(-50, 50)
    return W


for t in range(1, args.nmax // b + 1):
    n = t * b
    Xc, yc = X[:, n - b:n], y[:, n - b:n]          # new chunk
    Xs, ys = X[:, :n], y[:, :n]                    # everything seen so far
    res = {}
    with tm('ttt_gd_k1'):
        for lr in state['ttt1']: state['ttt1'][lr] = gd_steps(state['ttt1'][lr], Xc, yc, lr, 1)
        res['ttt_gd_k1'] = C.pick_best(link, Xs, ys, torch.cat(list(state['ttt1'].values())))
    with tm(f'ttt_gd_k{K}'):
        for lr in state['tttK']: state['tttK'][lr] = gd_steps(state['tttK'][lr], Xc, yc, lr, K)
        res[f'ttt_gd_k{K}'] = C.pick_best(link, Xs, ys, torch.cat(list(state['tttK'].values())))
    with tm(f'replay_adam_k{K}'):
        for lr, st in state['adam'].items():
            W, m, v = st['W'], st['m'], st['v']
            for _ in range(K):
                st['t'] += 1; tt = st['t']
                s = torch.einsum('end,red->ren', Xs, W)
                gr = 2 / n * torch.einsum('ren,end->red', (link.f(s) - ys) * link.df(s), Xs)
                m = .9 * m + .1 * gr; v = .999 * v + .001 * gr ** 2
                W = torch.nan_to_num(W - lr * (m / (1 - .9 ** tt)) / ((v / (1 - .999 ** tt)).sqrt() + 1e-8)).clamp(-50, 50)
            st.update(W=W, m=m, v=v)
        res[f'replay_adam_{R}x3_k{K}'] = C.pick_best(link, Xs, ys, torch.cat([st['W'] for st in state['adam'].values()]))
    with tm(f'replay_lm_k{K}'):
        st = state['lm']; W, mu = st['W'], st['mu']
        L = C.support_loss(link, Xs, ys, W)
        for _ in range(K):
            s = torch.einsum('end,red->ren', Xs, W); r = ys - link.f(s)
            J = link.df(s).unsqueeze(-1) * Xs.unsqueeze(0)
            A = torch.einsum('rend,renk->redk', J, J) + mu[..., None, None] * eye
            step = torch.linalg.solve(A, torch.einsum('rend,ren->red', J, r).unsqueeze(-1)).squeeze(-1)
            Wn = W + step; Ln = torch.nan_to_num(C.support_loss(link, Xs, ys, Wn), nan=float('inf')); ok = Ln < L
            W = torch.where(ok[..., None], Wn, W); L = torch.where(ok, Ln, L); mu = torch.where(ok, mu / 3, mu * 3).clamp(1e-8, 1e8)
        st.update(W=W, mu=mu)
        res[f'replay_lm_{R}_k{K}'] = C.pick_best(link, Xs, ys, W)
    with tm(f'pcalm_k{K}'):
        A = torch.einsum('end,enk->edk', Xs, Xs) + 1.0 * eye   # unit ridge prior, keeps n<d prefixes well posed
        cands = []
        for rho, st in state['pcalm'].items():
            W = st['W']
            lam = torch.cat([st['lam'], torch.zeros(1, E, b, dtype=dt, device=args.device)], -1)  # new examples: lambda=0
            for _ in range(K):
                pred = torch.einsum('end,red->ren', Xs, W)
                s = C._activity_step(link, ys, pred - lam / rho, rho, grid)
                lam = lam + rho * (s - pred)
                rhs = torch.einsum('end,ren->red', Xs, s + lam / rho).unsqueeze(-1)
                W = torch.linalg.solve(A.unsqueeze(0), rhs).squeeze(-1)
            st.update(W=W, lam=lam); cands.append(W)
        res[f'pcalm_1x2_k{K}'] = C.pick_best(link, Xs, ys, torch.cat(cands))
    for name, w in res.items():
        m = qmse(w)
        rows.append(dict(n=n, method=name, mse=float(m.mean()), se=float(m.std() / E ** .5),
                         success=float((m < 0.05).float().mean()), overlap=float(C.overlap(w, u).mean())))
    if t % 4 == 0 or t == 1:
        print(f'n={n}: ' + '  '.join(f"{r['method']}={r['mse']:.3f}/{r['success']:.2f}" for r in rows[-len(res):]), flush=True)
rows.append(dict(method='__time__', seconds={k: round(v, 3) for k, v in tm.t.items()}))
(out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('time', {k: round(v, 2) for k, v in tm.t.items()})
print('DONE')
