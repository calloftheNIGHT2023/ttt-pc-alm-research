"""Report 443 v2 (adds same-rho no-multiplier control, RAF and GAMP baselines): phase retrieval of REAL images (CIFAR-10, grayscale) - Gaussian and coded-diffraction measurements.

Task = one real image x*. Context = n intensity measurements y = |A x*|^2 (+ small noise). Evaluator only:
the image itself and held-out query measurements. All methods receive the same (A, y) and iteration cap.
float32 on GPU (documented deviation from float64 used in 442; tolerance 0.05 relative error is far above fp32 noise).

  --setting gauss : real Gaussian a_i ~ N(0, I_d), n = alpha*d, image normalised to unit norm
  --setting cdp   : y_l = |FFT(m_l * x)|^2 (orthonormal 2-D FFT), L random unit-modulus complex masks, n = L*d
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--setting', choices=['gauss', 'cdp'], required=True)
ap.add_argument('--sides', type=int, nargs='+', default=[16, 32])
ap.add_argument('--alphas', type=float, nargs='+', default=[1.5, 2, 2.5, 3, 4, 5, 6], help='gauss: n/d; cdp: masks L')
ap.add_argument('--episodes', type=int, default=32)
ap.add_argument('--restarts', type=int, default=16)
ap.add_argument('--T', type=int, default=500)
ap.add_argument('--sigma', type=float, default=1e-3, help='relative intensity noise')
ap.add_argument('--random_direction', action='store_true', help='control: random unit vectors instead of images')
ap.add_argument('--krr', action='store_true', help='also run the quadratic-feature closed-form head (small d only)')
ap.add_argument('--dataset', default='cifar10', choices=['cifar10', 'tinyimagenet'])
ap.add_argument('--only', nargs='*', default=None, help='run only these method names')
ap.add_argument('--seed', type=int, default=443001)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dev, dt = args.device, torch.float32
torch.manual_seed(args.seed)


# ------------------------------------------------------------------ real images
def load_images(E, side, seed):
    from datasets import load_dataset
    from PIL import Image
    if args.dataset == 'cifar10':
        ds, key = load_dataset('uoft-cs/cifar10', split='test'), 'img'
    else:                                   # native 64x64 real photographs
        ds, key = load_dataset('zh-plus/tiny-imagenet', split='valid'), 'image'
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), E, replace=False)
    imgs = []
    for i in idx:
        im = ds[int(i)][key].convert('L')
        if im.size != (side, side):
            im = im.resize((side, side), Image.BILINEAR)
        imgs.append(np.asarray(im, dtype=np.float64) / 255.0)
    return torch.tensor(np.stack(imgs), dtype=dt, device=dev), [int(i) for i in idx]


def rel_dist(Z, X):  # Z, X (E, d) real; global sign ambiguity
    return torch.minimum((Z - X).norm(dim=-1), (Z + X).norm(dim=-1)) / X.norm(dim=-1)


# ------------------------------------------------------------------ operators
class Gauss:
    def __init__(s, E, n, d, nq, g):
        s.A = torch.randn(E, n, d, generator=g, dtype=dt).to(dev)
        s.Aq = torch.randn(E, nq, d, generator=g, dtype=dt).to(dev)
        s.G = torch.linalg.cholesky(torch.einsum('end,enk->edk', s.A, s.A))
        s.n = n
    def fwd(s, Z): return torch.einsum('end,red->ren', s.A, Z)
    def adj(s, V): return torch.einsum('end,ren->red', s.A, V)
    def ls(s, V):   # argmin_z ||A z - V||^2  (exact local least squares, no chain rule)
        R = V.shape[0]
        return torch.cholesky_solve(s.adj(V).unsqueeze(-1), s.G.unsqueeze(0).expand(R, -1, -1, -1)).squeeze(-1)
    def query(s, Z): return torch.einsum('eqd,red->req', s.Aq, Z) ** 2


class CDP:
    def __init__(s, E, N, L, g):
        ph = torch.rand(L, N, N, generator=g) * 2 * math.pi
        s.M = torch.polar(torch.ones(L, N, N), ph).to(dev).to(torch.complex64)
        phq = torch.rand(2, N, N, generator=g) * 2 * math.pi               # 2 held-out query masks
        s.Mq = torch.polar(torch.ones(2, N, N), phq).to(dev).to(torch.complex64)
        s.L, s.N, s.n = L, N, L * N * N
    def fwd(s, Z):   # Z (R,E,d) real -> (R,E,L*d) complex
        R, E, d = Z.shape
        x = Z.reshape(R, E, 1, s.N, s.N).to(torch.complex64)
        return torch.fft.fft2(x * s.M, norm='ortho').reshape(R, E, -1)
    def adj(s, V):   # (R,E,L*d) complex -> (R,E,d) real part of A^* V
        R, E, _ = V.shape
        v = torch.fft.ifft2(V.reshape(R, E, s.L, s.N, s.N), norm='ortho') * s.M.conj()
        return v.sum(2).real.reshape(R, E, -1)
    def ls(s, V): return s.adj(V) / s.L         # A^*A = L I for unit-modulus masks, real projection
    def query(s, Z):
        R, E, d = Z.shape
        x = Z.reshape(R, E, 1, s.N, s.N).to(torch.complex64)
        return (torch.fft.fft2(x * s.Mq, norm='ortho').abs() ** 2).reshape(R, E, -1)


def amp(V): return V.abs()
def phase(V): return torch.where(V.abs() > 1e-12, V / V.abs().clamp_min(1e-12), torch.ones_like(V))


def sup_loss(op, Z, r):  # amplitude loss on support
    return ((amp(op.fwd(Z)) - r) ** 2).mean(-1)


def pick(op, Z, r):
    Lz = torch.nan_to_num(sup_loss(op, Z, r), nan=float('inf'))
    i = Lz.argmin(0)
    return Z[i, torch.arange(Z.shape[1])]


# ------------------------------------------------------------------ algorithms
def spectral(op, y, kind, iters=60):
    """Power iteration on (1/n) A^* diag(w) A restricted to real signals."""
    E = y.shape[0]; d = op.N ** 2 if isinstance(op, CDP) else op.A.shape[-1]
    lam2 = y.mean(-1, keepdim=True)
    if kind == 'trunc':      # Chen-Candes truncated spectral
        w = y * (y <= 9 * lam2)
    else:                    # TAF orthogonality-promoting: largest ceil(n/6) normalised intensities
        k = max(1, int(math.ceil(y.shape[-1] / 6)))
        thr = torch.topk(y, k, dim=-1).values[..., -1:]
        w = (y >= thr).to(y.dtype)
    g = torch.Generator().manual_seed(args.seed + 5)
    v = torch.randn(1, E, d, generator=g, dtype=dt).to(dev)
    for _ in range(iters):
        v = op.adj(w.unsqueeze(0) * op.fwd(v)); v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = (lam2.squeeze(-1) if isinstance(op, Gauss) else y.sum(-1) / op.L).sqrt()
    return v * scale.view(1, E, 1)


def wf(op, y, Z0, T, mu):
    """Wirtinger flow (gradient on the intensity loss). Step mu / (mean measured intensity of z0), which equals
    Candes et al.'s mu/||z0||^2 for Gaussian a_i and the matching curvature scale for orthonormal CDP."""
    Z = Z0.clone(); nrm = (amp(op.fwd(Z0)) ** 2).mean(-1, keepdim=True).clamp_min(1e-8)
    for _ in range(T):
        V = op.fwd(Z)
        g = op.adj((amp(V) ** 2 - y.unsqueeze(0)) * V) / (op.n if isinstance(op, Gauss) else op.L)
        Z = torch.nan_to_num(Z - mu / nrm * g)
    return Z


def af(op, r, Z0, T, mu, trunc=None):
    """(Truncated) amplitude flow; trunc=gamma gives TAF (Wang-Giannakis-Eldar)."""
    Z = Z0.clone(); scale = op.n if isinstance(op, Gauss) else op.L
    for _ in range(T):
        V = op.fwd(Z); res = V - r.unsqueeze(0) * phase(V)
        if trunc is not None:
            res = res * (amp(V) >= r.unsqueeze(0) / (1 + trunc))
        Z = torch.nan_to_num(Z - mu * op.adj(res) / scale)
    return Z


def adam(op, r, Z0, T, lr):
    Z = Z0.clone(); m = torch.zeros_like(Z); v = torch.zeros_like(Z)
    scale = op.n if isinstance(op, Gauss) else op.L
    for t in range(1, T + 1):
        V = op.fwd(Z); g = op.adj(V - r.unsqueeze(0) * phase(V)) / scale
        m = .9 * m + .1 * g; v = .999 * v + .001 * g ** 2
        Z = torch.nan_to_num(Z - lr * (m / (1 - .9 ** t)) / ((v / (1 - .999 ** t)).sqrt() + 1e-8))
    return Z


def lifted(op, r, Z0, T, rho, mult=True, growth=1.0, activity='exact'):
    """Lifted local credit on phase retrieval.

    Activities s = one free variable per measurement, constraint s = (A z)_i.
    Exact activity block: argmin (|s|-r)^2 + rho/2 |s - a|^2, a = Az - lam/rho  (closed form:
    the global minimiser lies on the anchor's ray/sign branch with modulus (2r + rho|a|)/(2+rho) for complex s;
    for real s both sign branches are compared).
    mult=False, rho->0  is Gerchberg-Saxton / error reduction (AltMin).
    """
    Z = Z0.clone(); V = op.fwd(Z); lam = torch.zeros_like(V); S = V.clone(); p = rho
    for _ in range(T):
        V = op.fwd(Z); a = V - lam / p
        if activity == 'exact':
            if V.is_complex():
                S = (2 * r + p * a.abs()) / (2 + p) * phase(a)
            else:
                sp = ((2 * r + p * a) / (2 + p)).clamp_min(0); sm = ((-2 * r + p * a) / (2 + p)).clamp_max(0)
                F = lambda s: (s.abs() - r) ** 2 + 0.5 * p * (s - a) ** 2
                S = torch.where(F(sp) <= F(sm), sp, sm)
        else:   # official-PC-ALM-style local gradient inference on the activity
            for _k in range(20):
                gs = 2 * (S.abs() - r) * phase(S) + p * (S - a)
                S = S - 0.05 * gs
        if mult:
            lam = lam + p * (S - V)
        Z = op.ls(S + lam / p)
        p = p * growth
    return Z


def altmin(op, r, Z0, T):
    Z = Z0.clone()
    for _ in range(T):
        Z = op.ls(r.unsqueeze(0) * phase(op.fwd(Z)))
    return Z


def raf(op, r, Z0, T, mu, beta=5.0):
    """Reweighted amplitude flow (Wang-Giannakis-Saad-Chen 2018): weights |a.z|/(|a.z| + beta r)."""
    Z = Z0.clone(); scale = op.n if isinstance(op, Gauss) else op.L
    for _ in range(T):
        V = op.fwd(Z); w = amp(V) / (amp(V) + beta * r.unsqueeze(0)).clamp_min(1e-12)
        Z = torch.nan_to_num(Z - mu * op.adj(w * (V - r.unsqueeze(0) * phase(V))) / scale)
    return Z


def gamp(op, r, Z0, T, sigma_r, damp=0.5):
    """Real GAMP for amplitude measurements r = |a.x| + w (Rangan 2011 / Schniter-Rangan prGAMP style).

    Gaussian prior x_j ~ N(0, v0), v0 = ||x||^2/d estimated from data. Output channel: sign mixture
    N(z; +r, s2) + N(z; -r, s2) (indicator truncation dropped; accurate for r >> sigma). Scalar variances,
    damping on x and s. Gaussian operator only. Z0 (R,E,d) initial estimates.
    """
    A = op.A; m, d = A.shape[1], A.shape[2]
    R = Z0.shape[0]
    v0 = (r ** 2).mean(-1).view(1, -1, 1) / d
    xh = Z0.clone(); tx = v0.expand(R, -1, 1).clone() * 0.5
    sh = torch.zeros(R, A.shape[0], m, dtype=dt, device=dev)
    s2 = max(sigma_r, 1e-3) ** 2
    rr = r.unsqueeze(0)
    for _ in range(T):
        tp = d * tx                                                     # (R,E,1)
        ph = op.fwd(xh) - tp * sh
        # sign-mixture posterior of z given (ph, tp) and r
        lp = -(ph - rr) ** 2 / (2 * (tp + s2)); ln = -(ph + rr) ** 2 / (2 * (tp + s2))
        pi = torch.sigmoid(lp - ln)
        mpos = (ph * s2 + rr * tp) / (tp + s2); mneg = (ph * s2 - rr * tp) / (tp + s2)
        v = tp * s2 / (tp + s2)
        zh = pi * mpos + (1 - pi) * mneg
        tz = v + pi * mpos ** 2 + (1 - pi) * mneg ** 2 - zh ** 2
        sn = (zh - ph) / tp
        ts = ((1 - tz / tp) / tp).mean(-1, keepdim=True).clamp_min(1e-12)
        sh = damp * sn + (1 - damp) * sh
        trr = 1 / (m * ts)
        rhat = xh + trr * op.adj(sh)
        xn = rhat * v0 / (v0 + trr)
        tx = damp * (v0 * trr / (v0 + trr)) + (1 - damp) * tx
        xh = torch.nan_to_num(damp * xn + (1 - damp) * xh)
    return xh


def krr_quadratic(op, y, nq_pred):
    """Closed-form head on the exact quadratic feature map vec(a a^T): kernel (a.a')^2 + 1, LOO ridge."""
    import common as C
    K = (op.A @ op.A.transpose(1, 2)).double() ** 2 + 1
    Kq = (op.Aq @ op.A.transpose(1, 2)).double() ** 2 + 1
    return C._dual_ridge_loo(K, y.double(), Kq, [1e-6, 1e-4, 1e-2, 1, 100]).to(dt)


# ------------------------------------------------------------------ experiment
rows = []
for side in args.sides:
    d = side * side; E = args.episodes
    if args.random_direction:
        g0 = torch.Generator().manual_seed(args.seed + side)
        X = torch.randn(E, d, generator=g0, dtype=dt).to(dev); img_ids = None
    else:
        Ximg, img_ids = load_images(E, side, args.seed + side); X = Ximg.reshape(E, d)
    X = X / X.norm(dim=-1, keepdim=True) * (1.0 if args.setting == 'gauss' else math.sqrt(d))
    for alpha in args.alphas:
        g = torch.Generator().manual_seed(args.seed + 1000 * side + int(10 * alpha))
        if args.setting == 'gauss':
            n = int(round(alpha * d)); op = Gauss(E, n, d, 512, g)
        else:
            L = int(alpha); op = CDP(E, side, L, g); n = op.n
        Y0 = amp(op.fwd(X.unsqueeze(0)))[0] ** 2
        y = (Y0 * (1 + args.sigma * torch.randn(Y0.shape, generator=g, dtype=dt).to(dev))).clamp_min(0)
        r = y.sqrt()
        yq = op.query(X.unsqueeze(0))[0]
        res, tm = {}, {}

        def run(name, fn):
            if args.only is not None and name not in args.only:
                return
            if dev.startswith('cuda'): torch.cuda.synchronize()
            t0 = time.perf_counter(); Z = fn()
            if dev.startswith('cuda'): torch.cuda.synchronize()
            tm[name] = time.perf_counter() - t0; res[name] = Z

        R = args.restarts
        scale = (y.mean(-1) if args.setting == 'gauss' else y.sum(-1) / op.L).sqrt()
        Z0 = torch.randn(R, E, d, generator=g, dtype=dt).to(dev)
        Z0 = Z0 / Z0.norm(dim=-1, keepdim=True) * scale.view(1, E, 1)       # random directions, norm from data
        T = args.T
        run('spectral_trunc', lambda: spectral(op, y, 'trunc')[0])
        run('spectral_orth', lambda: spectral(op, y, 'orth')[0])
        run('spectral_trunc+WF', lambda: pick(op, torch.cat([wf(op, y, spectral(op, y, 'trunc'), T, mu) for mu in [.05, .1, .2]]), r))
        run('spectral_orth+TAF', lambda: pick(op, torch.cat([af(op, r, spectral(op, y, 'orth'), T, mu, trunc=.7) for mu in [.3, .6, 1.]]), r))
        run('spectral_orth+ALTMIN', lambda: altmin(op, r, spectral(op, y, 'orth'), T)[0])
        run('spectral_orth+PCALM', lambda: pick(op, torch.cat([lifted(op, r, spectral(op, y, 'orth'), T, rho) for rho in [.1, 1.]]), r))
        run(f'rand{R}_WF', lambda: pick(op, torch.cat([wf(op, y, Z0, T, mu) for mu in [.05, .1, .2]]), r))
        run(f'rand{R}_AF', lambda: pick(op, torch.cat([af(op, r, Z0, T, mu) for mu in [.3, .6, 1.]]), r))
        run(f'rand{R}_TAF', lambda: pick(op, torch.cat([af(op, r, Z0, T, mu, trunc=.7) for mu in [.3, .6, 1.]]), r))
        run(f'rand{R}_Adam', lambda: pick(op, torch.cat([adam(op, r, Z0, T, lr) for lr in [.003, .01, .03]]), r))
        run(f'rand{R}_ALTMIN', lambda: pick(op, altmin(op, r, Z0, T), r))
        run('rand1_PCpenalty', lambda: lifted(op, r, Z0[:1], T, .1, mult=False, growth=1000 ** (1 / T))[0])
        run('rand1_PCALM', lambda: pick(op, torch.cat([lifted(op, r, Z0[:1], T, rho) for rho in [.1, 1.]]), r))
        run(f'rand{R}_PCALM', lambda: pick(op, torch.cat([lifted(op, r, Z0, T, rho) for rho in [.1, 1.]]), r))
        run('rand1_PCnomult_rhosel', lambda: pick(op, torch.cat([lifted(op, r, Z0[:1], T, rho, mult=False) for rho in [.1, 1.]]), r))
        run(f'rand{R}_PCnomult_rhosel', lambda: pick(op, torch.cat([lifted(op, r, Z0, T, rho, mult=False) for rho in [.1, 1.]]), r))
        run('spectral_orth+RAF', lambda: pick(op, torch.cat([raf(op, r, spectral(op, y, 'orth'), T, mu) for mu in [1., 2., 3.]]), r))
        if isinstance(op, Gauss):
            run(f'rand{R}_GAMP', lambda: pick(op, torch.cat([gamp(op, r, Z0 * 0.1, T, args.sigma, dm) for dm in [.3, .6]]), r))
            run('spectral_orth+GAMP', lambda: pick(op, torch.cat([gamp(op, r, spectral(op, y, 'orth'), T, args.sigma, dm) for dm in [.3, .6]]), r))
        run('rand1_PCALM_gradact', lambda: pick(op, torch.cat([lifted(op, r, Z0[:1], T, rho, activity='grad') for rho in [.1, 1.]]), r))
        qvar = yq.var(-1)
        for name, Z in res.items():
            dist = rel_dist(Z, X)
            qn = ((op.query(Z.unsqueeze(0))[0] - yq) ** 2).mean(-1) / qvar
            rows.append(dict(setting=args.setting, side=side, d=d, alpha=alpha, n=n, method=name,
                             success=float((dist < .05).float().mean()), dist_median=float(dist.median()),
                             dist_mean=float(dist.mean()), query_nmse=float(qn.mean()), seconds=tm[name]))
        if args.krr and args.setting == 'gauss':
            t0 = time.perf_counter(); P = krr_quadratic(op, y, 512); tt = time.perf_counter() - t0
            qn = ((P - yq) ** 2).mean(-1) / qvar
            rows.append(dict(setting=args.setting, side=side, d=d, alpha=alpha, n=n, method='closedform_quadratic_krr',
                             success=float((qn < .05).float().mean()), dist_median=None, dist_mean=None,
                             query_nmse=float(qn.mean()), seconds=tt))
        cur = [x for x in rows if x['side'] == side and x['alpha'] == alpha]
        print(f'side={side} d={d} {"alpha" if args.setting == "gauss" else "L"}={alpha} n={n}: ' +
              '  '.join(f"{x['method']}={x['success']:.2f}" for x in cur), flush=True)
        (out / 'rows.json').write_text(json.dumps(dict(image_ids=img_ids, rows=rows), indent=1))
print('DONE')
