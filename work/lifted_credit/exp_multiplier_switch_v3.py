"""Exp E v3 (report 444): multiplier switch with review fixes.

Changes vs v1/v2 (kept unchanged in results/):
  * certified activity solver (all real roots of the stationarity quintic) in both phases;
  * 'stuck' is decided from SUPPORT data only: over the last 10 phase-1 sweeps the relative parameter change is
    below 1e-8, no example changes its activity branch, and support MSE exceeds 0.05 (noise variance 0.01).
    This is a numerical-stationarity criterion, not a proof of an exact fixed point;
  * the no-multiplier continuation gets the SAME rho selection as the multiplier arm (rho in {0.1, 1}, support loss).
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
ap.add_argument('--rho1', type=float, default=0.1)
ap.add_argument('--T1', type=int, default=200)
ap.add_argument('--T2', type=int, default=400)
ap.add_argument('--seed', type=int, default=444601)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.He3; rows = []


def run_lifted(X, y, W, T, rho, mult, track=0):
    A = torch.einsum('end,enk->edk', X, X) + 1e-2 * torch.eye(X.shape[-1], dtype=dt, device=X.device)
    lam = torch.zeros(W.shape[0], X.shape[0], X.shape[1], dtype=dt, device=X.device)
    hist = []
    for t in range(T):
        pred = torch.einsum('end,red->ren', X, W)
        s, _ = C._activity_certified_he3(y.expand_as(pred), pred - lam / rho, rho)
        if mult:
            lam = lam + rho * (s - pred)
        Wn = torch.linalg.solve(A.unsqueeze(0), torch.einsum('end,ren->red', X, s + lam / rho).unsqueeze(-1)).squeeze(-1)
        if track and t >= T - track:
            hist.append(dict(rel=((Wn - W).norm(dim=-1) / W.norm(dim=-1).clamp_min(1e-12))[0], s=s[0]))
        W = Wn
    return W, hist


for d in args.dims:
    n = int(args.alpha * d); E = args.episodes
    t = C.make_tasks(E, d, n, 1000, link, 0.1, args.seed + d, args.device, dt)
    X, y, Xq, yq, u = t['X'], t['y'], t['Xq'], t['yq'], t['u']
    W0 = C.init_w(E, d, 1, args.seed + 7 + d, args.device, dt)
    W1, hist = run_lifted(X, y, W0, args.T1, args.rho1, False, track=11)
    rel = torch.stack([h['rel'] for h in hist[1:]]).max(0).values
    branch_changes = torch.stack([((hist[i + 1]['s'] - hist[i]['s']).abs() > 1e-6).sum(-1) for i in range(len(hist) - 1)]).sum(0)
    sup = C.support_loss(link, X, y, W1)[0]
    stuck = (rel < 1e-8) & (branch_changes == 0) & (sup > 0.05)
    arms = {
        'switch_on_multiplier_rho_sel': C.pick_best(link, X, y, torch.cat([run_lifted(X, y, W1, args.T2, r, True)[0] for r in [.1, 1.]])),
        'no_multiplier_rho_sel': C.pick_best(link, X, y, torch.cat([run_lifted(X, y, W1, args.T2, r, False)[0] for r in [.1, 1.]])),
        'bp_adam_x3lr': C.pick_best(link, X, y, torch.cat([C.bp_adam(link, X, y, W1, args.T2, lr) for lr in [.01, .03, .1]])),
        'bp_lm': C.bp_lm(link, X, y, W1, args.T2)[0],
    }
    row = dict(d=d, n=n, episodes=E, support_stuck=int(stuck.sum()),
               criteria=dict(rel_change_max=float(rel.max()), tasks_with_branch_changes=int((branch_changes > 0).sum()),
                             support_mse_min=float(sup.min())),
               phase1_query_unsolved=int((C.query_mse_w(link, Xq, yq, W1[0]) >= .05).sum()),
               phase1_overlap_stuck=float(C.overlap(W1[0], u)[stuck].mean()) if stuck.any() else None)
    for k, W in arms.items():
        m = C.query_mse_w(link, Xq, yq, W)
        row[k] = dict(escape_rate=float((m[stuck] < .05).float().mean()) if stuck.any() else None,
                      mse=float(m[stuck].mean()) if stuck.any() else None,
                      overlap=float(C.overlap(W, u)[stuck].mean()) if stuck.any() else None)
    rows.append(row); print(json.dumps(row), flush=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
