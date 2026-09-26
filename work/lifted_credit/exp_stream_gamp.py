"""Exp J (report 445): streaming TTT inner loops vs GAMP, grouped by state.

He3 single index, chunks of b examples, K inner iterations per chunk, queries evaluated by the evaluator only.
Replay group (keeps all seen data, O(nd) state; hyper-parameters selected by loss on all seen support):
  pcalm_replay      persistent activities + multipliers on all seen data (equivalently proximal DR, see 445)
  gamp_replay       warm-started GAMP on all seen rows (new rows start with s_hat = 0)
  adam_replay       Adam on all seen data, 8 starts x 3 lr
Bounded-state group (selection by accumulated prequential loss on each new chunk BEFORE updating):
  ttt_gd_k1         official TTT rule: one GD step per chunk on that chunk             state d
  mbamp_k{K}        mini-batch streaming AMP: previous Gaussian posterior (mean, var) is the prior   state 2d
  pcalm_rls         lifted PC-ALM iterations on the current chunk only; afterwards the chunk's final LS targets
                    are folded into (Gram, vector) and its activities/multipliers are dropped         state d^2+d
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, default=64)
ap.add_argument('--chunk', type=int, default=16)
ap.add_argument('--nmax_over_d', type=int, default=12)
ap.add_argument('--K', type=int, default=10)
ap.add_argument('--episodes', type=int, default=64)
ap.add_argument('--seed', type=int, default=445101)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.He3; dev = args.device; sigma = 0.1
E, d, b, K = args.episodes, args.d, args.chunk, args.K
nmax = args.nmax_over_d * d
task = C.make_tasks(E, d, nmax, 1000, link, sigma, args.seed, dev, dt)
X, y, Xq, yq, u = task['X'], task['y'], task['Xq'], task['yq'], task['u']
W0 = C.init_w(E, d, 8, args.seed + 1, dev, dt)
grid = torch.linspace(-5, 5, 241, dtype=dt, device=dev)
eye = torch.eye(d, dtype=dt, device=dev)
zg = torch.linspace(-8, 8, 401, dtype=dt, device=dev)
v0 = 1.0 / d


def gamp_step(st, Xs, ys, damp, m, v):
    """One GAMP iteration with per-coordinate Gaussian prior N(m, v); st holds xh, tx (per coordinate), sh."""
    xh, tx, sh = st['xh'], st['tx'], st['sh']
    X2 = Xs ** 2
    tp = torch.einsum('end,red->ren', X2, tx).clamp_min(1e-12)
    ph = torch.einsum('end,red->ren', Xs, xh) - tp * sh
    loglik = -(ys.unsqueeze(-1) - link.f(zg)) ** 2 / (2 * sigma ** 2)
    zh = torch.empty_like(ph); tz = torch.empty_like(ph)
    for q in range(ph.shape[0]):
        w = torch.softmax(loglik - (zg - ph[q].unsqueeze(-1)) ** 2 / (2 * tp[q].unsqueeze(-1)), -1)
        zh[q] = (w * zg).sum(-1); tz[q] = (w * zg ** 2).sum(-1) - zh[q] ** 2
    sn = (zh - ph) / tp
    ts = ((1 - tz / tp) / tp).clamp_min(1e-8)
    sh = damp * sn + (1 - damp) * sh
    tr = 1.0 / torch.einsum('end,ren->red', X2, ts).clamp_min(1e-12)
    rhat = xh + tr * torch.einsum('end,ren->red', Xs, sh)
    xn = (rhat * v + m * tr) / (v + tr); txn = v * tr / (v + tr)
    st.update(xh=torch.nan_to_num(damp * xn + (1 - damp) * xh), tx=damp * txn + (1 - damp) * tx, sh=sh)


def preq(W, Xc, yc):          # prequential loss of each candidate on the NEW chunk before updating
    return torch.nan_to_num(((link.f(torch.einsum('end,red->ren', Xc, W)) - yc) ** 2).mean(-1), nan=1e6).clamp_max(1e6)


# ------------------------------------------------------------------ states
S = {}
S['ttt'] = dict(W=torch.cat([W0[:1]] * 4), lrs=torch.tensor([.01, .03, .1, .3], dtype=dt, device=dev).view(4, 1, 1), pl=torch.zeros(4, E, dtype=dt, device=dev))
S['mbamp'] = dict(xh=torch.cat([0.1 * W0[:1]] * 2), tx=torch.full((2, E, d), 0.5 * v0, dtype=dt, device=dev),
                  m=torch.zeros(2, E, d, dtype=dt, device=dev), v=torch.full((2, E, d), v0, dtype=dt, device=dev),
                  damps=[.3, .6], pl=torch.zeros(2, E, dtype=dt, device=dev))
S['rls'] = dict(W=torch.cat([W0[:1]] * 2), G=torch.zeros(E, d, d, dtype=dt, device=dev), h=torch.zeros(2, E, d, dtype=dt, device=dev),
                rhos=[.1, 1.], pl=torch.zeros(2, E, dtype=dt, device=dev))
S['pcalm'] = {rho: dict(W=W0[:1].clone(), lam=torch.zeros(1, E, 0, dtype=dt, device=dev)) for rho in [.1, 1.]}
S['gamp'] = {dm: dict(xh=0.1 * W0[:1].clone(), tx=torch.full((1, E, d), 0.5 * v0, dtype=dt, device=dev),
                      sh=torch.zeros(1, E, 0, dtype=dt, device=dev)) for dm in [.3, .6]}
S['adam'] = {lr: dict(W=W0.clone(), m=torch.zeros_like(W0), v=torch.zeros_like(W0), t=0) for lr in [.01, .03, .1]}
tm = {k: 0.0 for k in ['ttt_gd_k1', f'mbamp_k{K}', 'pcalm_rls', 'pcalm_replay', 'gamp_replay', 'adam_replay']}
rows = []


def tic(): torch.cuda.synchronize(); return time.perf_counter()
def toc(name, t0): torch.cuda.synchronize(); tm[name] += time.perf_counter() - t0


for t in range(1, nmax // b + 1):
    n = t * b; Xc, yc = X[:, n - b:n], y[:, n - b:n]; Xs, ys = X[:, :n], y[:, :n]
    est = {}
    # --- bounded state -------------------------------------------------------
    t0 = tic(); st = S['ttt']
    st['pl'] += preq(st['W'], Xc, yc)
    s_ = torch.einsum('end,red->ren', Xc, st['W'])
    st['W'] = torch.nan_to_num(st['W'] - st['lrs'] * 2 / b * torch.einsum('ren,end->red', (link.f(s_) - yc) * link.df(s_), Xc)).clamp(-50, 50)
    est['ttt_gd_k1'] = st['W'][st['pl'].argmin(0), torch.arange(E)]; toc('ttt_gd_k1', t0)

    t0 = tic(); st = S['mbamp']
    st['pl'] += preq(st['xh'], Xc, yc)
    for i, dm in enumerate(st['damps']):
        sub = dict(xh=st['xh'][i:i + 1], tx=st['tx'][i:i + 1], sh=torch.zeros(1, E, b, dtype=dt, device=dev))
        for _ in range(K):
            gamp_step(sub, Xc, yc, dm, st['m'][i:i + 1], st['v'][i:i + 1])
        st['xh'][i:i + 1] = sub['xh']; st['tx'][i:i + 1] = sub['tx']
        st['m'][i:i + 1] = sub['xh']; st['v'][i:i + 1] = sub['tx'].clamp_min(1e-10)
    est[f'mbamp_k{K}'] = st['xh'][st['pl'].argmin(0), torch.arange(E)]; toc(f'mbamp_k{K}', t0)

    t0 = tic(); st = S['rls']
    st['pl'] += preq(st['W'], Xc, yc)
    Gc = torch.einsum('end,enk->edk', Xc, Xc)
    A = st['G'] + Gc + 1.0 * eye                    # unit ridge prior as in exp_streaming.py
    for i, rho in enumerate(st['rhos']):
        W = st['W'][i:i + 1]; lam = torch.zeros(1, E, b, dtype=dt, device=dev)
        for _ in range(K):
            pred = torch.einsum('end,red->ren', Xc, W)
            s = C._activity_step(link, yc, pred - lam / rho, rho, grid)
            lam = lam + rho * (s - pred)
            W = torch.linalg.solve(A.unsqueeze(0), (st['h'][i:i + 1] + torch.einsum('end,ren->red', Xc, s + lam / rho)).unsqueeze(-1)).squeeze(-1)
        st['h'][i:i + 1] = st['h'][i:i + 1] + torch.einsum('end,ren->red', Xc, s + lam / rho)
        st['W'][i:i + 1] = W
    st['G'] = st['G'] + Gc
    est['pcalm_rls'] = st['W'][st['pl'].argmin(0), torch.arange(E)]; toc('pcalm_rls', t0)

    # --- replay --------------------------------------------------------------
    t0 = tic(); A = torch.einsum('end,enk->edk', Xs, Xs) + 1.0 * eye; cands = []
    for rho, st in S['pcalm'].items():
        W = st['W']; lam = torch.cat([st['lam'], torch.zeros(1, E, b, dtype=dt, device=dev)], -1)
        for _ in range(K):
            pred = torch.einsum('end,red->ren', Xs, W)
            s = C._activity_step(link, ys, pred - lam / rho, rho, grid)
            lam = lam + rho * (s - pred)
            W = torch.linalg.solve(A.unsqueeze(0), torch.einsum('end,ren->red', Xs, s + lam / rho).unsqueeze(-1)).squeeze(-1)
        st.update(W=W, lam=lam); cands.append(W)
    est['pcalm_replay'] = C.pick_best(link, Xs, ys, torch.cat(cands)); toc('pcalm_replay', t0)

    t0 = tic(); cands = []
    for dm, st in S['gamp'].items():
        st['sh'] = torch.cat([st['sh'], torch.zeros(1, E, b, dtype=dt, device=dev)], -1)
        m0 = torch.zeros(1, E, d, dtype=dt, device=dev); vv = torch.full((1, E, d), v0, dtype=dt, device=dev)
        for _ in range(K):
            gamp_step(st, Xs, ys, dm, m0, vv)
        cands.append(st['xh'])
    est['gamp_replay'] = C.pick_best(link, Xs, ys, torch.cat(cands)); toc('gamp_replay', t0)

    t0 = tic()
    for lr, st in S['adam'].items():
        W, mm, vv = st['W'], st['m'], st['v']
        for _ in range(K):
            st['t'] += 1; tt = st['t']
            s_ = torch.einsum('end,red->ren', Xs, W)
            gr = 2 / n * torch.einsum('ren,end->red', (link.f(s_) - ys) * link.df(s_), Xs)
            mm = .9 * mm + .1 * gr; vv = .999 * vv + .001 * gr ** 2
            W = torch.nan_to_num(W - lr * (mm / (1 - .9 ** tt)) / ((vv / (1 - .999 ** tt)).sqrt() + 1e-8)).clamp(-50, 50)
        st.update(W=W, m=mm, v=vv)
    est['adam_replay'] = C.pick_best(link, Xs, ys, torch.cat([st['W'] for st in S['adam'].values()])); toc('adam_replay', t0)

    if t % max(1, (d // 2) // b) == 0 or t == nmax // b:
        for name, w in est.items():
            m = C.query_mse_w(link, Xq, yq, w)
            rows.append(dict(n=n, n_over_d=n / d, method=name, success=float((m < .05).float().mean()),
                             mse=float(m.mean()), overlap=float(C.overlap(w, u).mean())))
        print(f'n={n} ({n / d:.1f}d): ' + '  '.join(f"{r['method']}={r['success']:.2f}" for r in rows[-len(est):]), flush=True)
        (out / 'rows.json').write_text(json.dumps(dict(rows=rows, seconds=tm), indent=1))
print('seconds', {k: round(v, 1) for k, v in tm.items()})
print('DONE')
