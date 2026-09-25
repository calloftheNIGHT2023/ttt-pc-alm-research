"""Method suite shared by experiments on given (X, y, Xq, yq) episodes. Queries only enter the evaluator."""
import torch
import common as C


def run_suite(link, X, y, Xq, yq, u, T=200, R=8, bp_big=64, sigma=0.1, seed=0, skip=(), tm=None, grid_lim=None):
    tm = tm or C.Timer()
    E, n, d = X.shape
    res = {}
    vq = yq.var(-1)                                  # per-episode query variance for normalised MSE

    def rec_w(name, w):
        m = C.query_mse_w(link, Xq, yq, w)
        res[name] = dict(mse=m, nmse=m / vq, ov=None if u is None else C.overlap(w, u))

    def rec_p(name, p):
        m = ((p - yq) ** 2).mean(-1)
        res[name] = dict(mse=m, nmse=m / vq, ov=None)

    if u is not None:
        rec_w('oracle', u)
    rec_p('zero', torch.zeros_like(yq))
    with tm('linear_ridge'): rec_p('linear_ridge', C.linear_ridge(X, y, Xq))
    if 'rf' not in skip:
        for kind in ['relu', 'link']:
            with tm(f'rf4096_{kind}'):
                F, Fq = C.random_features(X, Xq, 4096, kind, link, seed + 7)
                rec_p(f'rf4096_{kind}_ridge', C.features_ridge(F, Fq, y))
    with tm('rbf_krr'): rec_p('rbf_krr', C.rbf_krr(X, y, Xq))
    with tm('poly3_krr'): rec_p('poly3_krr', C.poly3_krr(X, y, Xq))
    if 'mlp' not in skip:
        with tm('mlp_extra_layer'): rec_p('mlp_extra_layer', C.mlp_extra_layer(X, y, Xq, T=2 * T))
    Rb = max(R, bp_big)
    W0 = C.init_w(E, d, Rb, seed + 1000, X.device, X.dtype)
    with tm(f'bp_adam_{Rb}x3'):
        rec_w(f'bp_adam_{Rb}x3', C.pick_best(link, X, y, torch.cat([C.bp_adam(link, X, y, W0, T, lr) for lr in [.01, .03, .1]])))
    with tm(f'bp_gd_{Rb}x3'):
        rec_w(f'bp_gd_{Rb}x3', C.pick_best(link, X, y, torch.cat([C.bp_gd(link, X, y, W0, T, lr) for lr in [.01, .03, .1]])))
    with tm(f'bp_lm_{Rb}'):
        rec_w(f'bp_lm_{Rb}', C.pick_best(link, X, y, C.bp_lm(link, X, y, W0, T)))
    for k, W in [(1, W0[:1]), (R, W0[:R])]:
        with tm(f'pc_alm_{k}x2'):
            rec_w(f'pc_alm_{k}x2', C.pick_best(link, X, y, torch.cat([C.lifted(link, X, y, W, T, rho, grid_lim=grid_lim) for rho in [.1, 1.]])))
        with tm(f'pc_alm_gradact_{k}x2'):
            rec_w(f'pc_alm_gradact_{k}x2', C.pick_best(link, X, y, torch.cat(
                [C.lifted(link, X, y, W, T, rho, activity='grad', grid_lim=grid_lim) for rho in [.1, 1.]])))
    with tm('pc_penalty_cont'):
        rec_w('pc_penalty_cont', C.lifted(link, X, y, W0[:1], T, .1, False, rho_growth=1000 ** (1 / T), grid_lim=grid_lim)[0])
    with tm('altmin'):
        rec_w('altmin', C.lifted(link, X, y, W0[:1], T, 1e-3, False, grid_lim=grid_lim)[0])
    with tm('hybrid_transform_lm'):
        w0 = C.hybrid_transform_init(link, X, y, sigma, zlim=6.0 if grid_lim is None else 20.0)
        rec_w('hybrid_transform_lm', C.bp_lm(link, X, y, w0.unsqueeze(0), T)[0])
    return res, tm


def summarise(res, extra):
    rows = []
    for name, r in res.items():
        m, nm = r['mse'], r['nmse']
        rows.append(dict(extra, method=name, mse=float(m.mean()), nmse=float(nm.mean()),
                         nmse_median=float(nm.median()), nmse_se=float(nm.std() / len(nm) ** .5),
                         success=float((nm < 0.05).float().mean()),
                         overlap=None if r['ov'] is None else float(r['ov'].mean())))
    return rows
