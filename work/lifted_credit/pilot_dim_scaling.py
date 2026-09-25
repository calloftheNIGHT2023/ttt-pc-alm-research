"""Pilot for report 440 predictions P1-P3: dimension scaling at fixed n = alpha*d.

Usage: python pilot_dim_scaling.py --link he3 --dims 8 16 32 64 128 --alpha 4 --out <dir>
"""
import argparse
import json
from pathlib import Path
import torch
import common as C

ap = argparse.ArgumentParser()
ap.add_argument('--link', default='he3')
ap.add_argument('--dims', type=int, nargs='+', default=[8, 16, 32, 64, 128])
ap.add_argument('--alpha', type=float, default=4.0)
ap.add_argument('--episodes', type=int, default=32)
ap.add_argument('--nq', type=int, default=2000)
ap.add_argument('--sigma', type=float, default=0.1)
ap.add_argument('--T', type=int, default=200)
ap.add_argument('--restarts', type=int, default=8)
ap.add_argument('--seed', type=int, default=440001)
ap.add_argument('--device', default='cpu')
ap.add_argument('--skip', nargs='*', default=[])
ap.add_argument('--bp_big', type=int, default=0, help='extra compute-matched BP restarts (0=off)')
ap.add_argument('--out', required=True)
args = ap.parse_args()

torch.set_num_threads(8)
dt = torch.float64
link = C.LINKS[args.link]
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
rows = []
for d in args.dims:
    n = int(round(args.alpha * d))
    task = C.make_tasks(args.episodes, d, n, args.nq, link, args.sigma, args.seed + d, args.device, dt)
    X, y, Xq, yq, u = task['X'], task['y'], task['Xq'], task['yq'], task['u']
    tm = C.Timer(); res = {}

    def rec_w(name, w):
        res[name] = dict(mse=C.query_mse_w(link, Xq, yq, w), ov=C.overlap(w, u))

    def rec_p(name, p):
        res[name] = dict(mse=((p - yq) ** 2).mean(-1), ov=None)

    rec_w('oracle', u)
    res['zero'] = dict(mse=(yq ** 2).mean(-1), ov=None)
    with tm('linear_ridge'): rec_p('linear_ridge', C.linear_ridge(X, y, Xq))
    if 'rf' not in args.skip:
        for kind in ['relu', 'link']:
            with tm(f'rf4096_{kind}'):
                F, Fq = C.random_features(X, Xq, 4096, kind, link, args.seed + 7)
                rec_p(f'rf4096_{kind}_ridge', C.features_ridge(F, Fq, y))
    with tm('rbf_krr'): rec_p('rbf_krr', C.rbf_krr(X, y, Xq))
    with tm('poly3_krr'): rec_p('poly3_krr', C.poly3_krr(X, y, Xq))
    if 'mlp' not in args.skip:
        with tm('mlp_extra_layer'): rec_p('mlp_extra_layer', C.mlp_extra_layer(X, y, Xq, T=args.T * 2))

    R = args.restarts
    W0 = C.init_w(args.episodes, d, max(R, args.bp_big), args.seed + 1000 + d, args.device, dt)
    if args.bp_big:
        with tm(f'bp_adam_{args.bp_big}x3'):
            cands = [C.bp_adam(link, X, y, W0[:args.bp_big], args.T, lr) for lr in [0.01, 0.03, 0.1]]
            rec_w(f'bp_adam_{args.bp_big}x3', C.pick_best(link, X, y, torch.cat(cands)))
        with tm(f'bp_gd_{args.bp_big}x3'):
            cands = [C.bp_gd(link, X, y, W0[:args.bp_big], args.T, lr) for lr in [0.01, 0.03, 0.1]]
            rec_w(f'bp_gd_{args.bp_big}x3', C.pick_best(link, X, y, torch.cat(cands)))
        with tm(f'bp_lm_{args.bp_big}'):
            rec_w(f'bp_lm_{args.bp_big}', C.pick_best(link, X, y, C.bp_lm(link, X, y, W0[:args.bp_big], args.T)))
        with tm('pc_alm_official_grad'):
            cands = [C.lifted(link, X, y, W0[:1], args.T, rho, True, activity='grad') for rho in [0.1, 1.0]]
            rec_w('pc_alm_gradact_1x2', C.pick_best(link, X, y, torch.cat(cands)))
        with tm('pc_alm_official_grad_R'):
            cands = [C.lifted(link, X, y, W0[:R], args.T, rho, True, activity='grad') for rho in [0.1, 1.0]]
            rec_w(f'pc_alm_gradact_{R}x2', C.pick_best(link, X, y, torch.cat(cands)))
        with tm(f'pc_alm_{R}x2'):
            cands = [C.lifted(link, X, y, W0[:R], args.T, rho, True) for rho in [0.1, 1.0]]
            rec_w(f'pc_alm_{R}x2', C.pick_best(link, X, y, torch.cat(cands)))
    W0 = W0[:R]
    with tm('bp_gd'):
        cands = [C.bp_gd(link, X, y, W0, args.T, lr) for lr in [0.01, 0.03, 0.1]]
        rec_w(f'bp_gd_{R}x3', C.pick_best(link, X, y, torch.cat(cands)))
    with tm('bp_adam'):
        cands = [C.bp_adam(link, X, y, W0, args.T, lr) for lr in [0.01, 0.03, 0.1]]
        rec_w(f'bp_adam_{R}x3', C.pick_best(link, X, y, torch.cat(cands)))
    with tm('bp_lm'):
        rec_w(f'bp_lm_{R}', C.pick_best(link, X, y, C.bp_lm(link, X, y, W0, args.T)))
    with tm('bp_lm_1'):
        rec_w('bp_lm_1', C.bp_lm(link, X, y, W0[:1], args.T)[0])
    W1 = W0[:1]
    with tm('pc_alm'):
        cands = [C.lifted(link, X, y, W1, args.T, rho, True) for rho in [0.1, 1.0]]
        rec_w('pc_alm_1x2', C.pick_best(link, X, y, torch.cat(cands)))
    with tm('pc_alm_rho1'):
        rec_w('pc_alm_rho1_single', C.lifted(link, X, y, W1, args.T, 1.0, True)[0])
    with tm('pc_penalty'):
        growth = 1000.0 ** (1.0 / args.T)
        rec_w('pc_penalty_cont', C.lifted(link, X, y, W1, args.T, 0.1, False, rho_growth=growth)[0])
    with tm('altmin'):
        rec_w('altmin_rho1e-3', C.lifted(link, X, y, W1, args.T, 1e-3, False)[0])
    with tm('hybrid'):
        w0 = C.hybrid_transform_init(link, X, y, args.sigma)
        rec_w('hybrid_transform_only', w0)
        rec_w('hybrid_transform_lm', C.bp_lm(link, X, y, w0.unsqueeze(0), args.T)[0])

    for name, r in res.items():
        m = r['mse']
        row = dict(d=d, n=n, method=name, mse=float(m.mean()), mse_se=float(m.std() / len(m) ** 0.5),
                   mse_median=float(m.median()),
                   success=float((m < 0.05).float().mean()),
                   overlap=None if r['ov'] is None else float(r['ov'].mean()),
                   seconds=None)
        rows.append(row)
    for k, v in tm.t.items():
        rows.append(dict(d=d, n=n, method='__time__' + k, seconds=v))
    print(f'd={d} n={n}', flush=True)
    for row in rows:
        if row['d'] == d and not row['method'].startswith('__'):
            print(f"  {row['method']:24s} mse={row['mse']:.4f}±{row['mse_se']:.4f} med={row['mse_median']:.4f} "
                  f"succ={row['success']:.2f} ov={row['overlap'] if row['overlap'] is None else round(row['overlap'],3)}", flush=True)
    print('  time:', {k: round(v, 2) for k, v in tm.t.items()}, flush=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
print('DONE')
