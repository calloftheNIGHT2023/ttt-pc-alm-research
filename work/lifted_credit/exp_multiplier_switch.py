"""Exp E (report 441, Prop. 4): causal test of the multiplier at the SAME stuck state.

Phase 1: lifted PC WITHOUT multipliers (penalty rho fixed) for T1 sweeps -> state (w, s).
Phase 2 from that identical state, T2 more iterations of:
  switch_on_multiplier : same lifted solver, multipliers enabled (lambda starts at 0)
  continue_no_mult     : same lifted solver, multipliers still off
  bp_adam / bp_lm      : BP continuations of the parameter w (lr grid picked by support loss)
Only the multiplier differs between the first two arms.
"""
import argparse
import json
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128])
ap.add_argument('--alpha', type=float, default=4.0)
ap.add_argument('--episodes', type=int, default=128)
ap.add_argument('--rho', type=float, default=0.1)
ap.add_argument('--T1', type=int, default=200)
ap.add_argument('--T2', type=int, default=200)
ap.add_argument('--seed', type=int, default=440601)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.He3; rows = []


def lifted_from(X, y, W, s, T, rho, mult):
    grid = torch.linspace(-5, 5, 241, dtype=dt, device=X.device)
    A = torch.einsum('end,enk->edk', X, X) + 1e-2 * torch.eye(X.shape[-1], dtype=dt, device=X.device)
    lam = torch.zeros_like(s); resid = []
    for _ in range(T):
        pred = torch.einsum('end,red->ren', X, W)
        s = C._activity_step(link, y, pred - lam / rho, rho, grid)
        if mult:
            lam = lam + rho * (s - pred)
        W = torch.linalg.solve(A.unsqueeze(0), torch.einsum('end,ren->red', X, s + lam / rho).unsqueeze(-1)).squeeze(-1)
        resid.append(float((s - pred).pow(2).mean()))
    return W, s, resid


for d in args.dims:
    n = int(args.alpha * d); E = args.episodes
    t = C.make_tasks(E, d, n, 1000, link, 0.1, args.seed + d, args.device, dt)
    X, y, Xq, yq, u = t['X'], t['y'], t['Xq'], t['yq'], t['u']
    W0 = C.init_w(E, d, 1, args.seed + 7 + d, args.device, dt)
    W1, s1, r1 = lifted_from(X, y, W0, torch.einsum('end,red->ren', X, W0), args.T1, args.rho, False)
    stuck = C.query_mse_w(link, Xq, yq, W1[0]) >= 0.05          # tasks not solved after phase 1
    arms = {}
    arms['switch_on_multiplier'] = lifted_from(X, y, W1, s1, args.T2, args.rho, True)[0][0]
    # v2: same rho selection as the main method (rho in {0.1, 1}, chosen by support loss only)
    arms['switch_on_multiplier_rho_sel'] = C.pick_best(link, X, y, torch.cat(
        [lifted_from(X, y, W1, s1, args.T2, rho, True)[0] for rho in [0.1, 1.0]]))
    arms['continue_no_mult'] = lifted_from(X, y, W1, s1, args.T2, args.rho, False)[0][0]
    arms['bp_adam_x3lr'] = C.pick_best(link, X, y, torch.cat([C.bp_adam(link, X, y, W1, args.T2, lr) for lr in [.01, .03, .1]]))
    arms['bp_lm'] = C.bp_lm(link, X, y, W1, args.T2)[0]
    row = dict(d=d, n=n, episodes=E, stuck_after_phase1=int(stuck.sum()),
               phase1_final_residual=r1[-1], phase1_overlap=float(C.overlap(W1[0], u)[stuck].mean()))
    for k, W in arms.items():
        m = C.query_mse_w(link, Xq, yq, W)
        row[k + '_escape_rate_among_stuck'] = float((m[stuck] < 0.05).float().mean())
        row[k + '_mse_among_stuck'] = float(m[stuck].mean())
        row[k + '_overlap_among_stuck'] = float(C.overlap(W, u)[stuck].mean())
    rows.append(row); print(json.dumps(row), flush=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
