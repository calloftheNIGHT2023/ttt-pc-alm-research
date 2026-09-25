"""Exp F (report 441, Prop. 2 vs Prop. 3): sample-complexity scaling exponents.

Prediction written before running (after committee r=4 showed BP catching up at n=16d, d=64):
  PC-ALM succeeds once n >= c1 * d        (threshold alpha* = n*/d roughly constant in d)
  BP (strongest: Adam, 64 starts x 3 lr) needs n >= c2 * d^{3/2}  (alpha* grows like sqrt(d))
Grid: d in {32, 64, 128}, alpha = n/d in {2,3,4,6,8,12,16,24,32}. Single-index He3, sigma=0.1.
"""
import argparse
import json
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--dims', type=int, nargs='+', default=[32, 64, 128])
ap.add_argument('--alphas', type=float, nargs='+', default=[2, 3, 4, 6, 8, 12, 16, 24, 32])
ap.add_argument('--episodes', type=int, default=32)
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--bp_restarts', type=int, default=64)
ap.add_argument('--seed', type=int, default=440701)
ap.add_argument('--device', default='cuda:1')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dt = torch.float64; link = C.He3; rows = []
for d in args.dims:
    for a in args.alphas:
        n = int(a * d); E = args.episodes
        t = C.make_tasks(E, d, n, 1000, link, 0.1, args.seed + 1000 * d + int(10 * a), args.device, dt)
        X, y, Xq, yq, u = t['X'], t['y'], t['Xq'], t['yq'], t['u']
        W0 = C.init_w(E, d, args.bp_restarts, args.seed + 7 + d, args.device, dt)
        tm = C.Timer(); res = {}
        with tm('pc_alm_1x2'):
            res['pc_alm_1x2'] = C.pick_best(link, X, y, torch.cat([C.lifted(link, X, y, W0[:1], args.T, r) for r in [.1, 1.]]))
        big = f'bp_adam_{args.bp_restarts}x3'
        with tm(big):
            res[big] = C.pick_best(link, X, y, torch.cat([C.bp_adam(link, X, y, W0, args.T, lr) for lr in [.01, .03, .1]]))
        with tm('bp_adam_1x3'):
            res['bp_adam_1x3'] = C.pick_best(link, X, y, torch.cat([C.bp_adam(link, X, y, W0[:1], args.T, lr) for lr in [.01, .03, .1]]))
        for k, w in res.items():
            m = C.query_mse_w(link, Xq, yq, w)
            rows.append(dict(d=d, alpha=a, n=n, method=k, success=float((m < .05).float().mean()),
                             mse=float(m.mean()), overlap=float(C.overlap(w, u).mean()), seconds=tm.t[k]))
        print(f'd={d} alpha={a} n={n}: ' + '  '.join(f"{r['method']}={r['success']:.2f}" for r in rows[-3:]), flush=True)
        (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
