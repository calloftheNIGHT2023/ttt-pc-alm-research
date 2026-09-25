"""Shared task family and methods for the lifted-credit main line (report 440).

Task: x ~ N(0, I_d), per-task u ~ Unif(S^{d-1}), y = g(u.x) + sigma*eps.
Every model-based method adapts w in f_w(x) = g(w.x) with the same link g.
Queries are only used by `evaluate`.
"""
import math
import time
import torch

SQ6 = math.sqrt(6.0)
GRID_LIM = 5.0   # activity search range; experiments on real features widen it


# ---------------------------------------------------------------- links
class He3:
    name = 'he3'
    @staticmethod
    def f(z): return (z ** 3 - 3 * z) / SQ6
    @staticmethod
    def df(z): return (3 * z ** 2 - 3) / SQ6
    @staticmethod
    def d2f(z): return 6 * z / SQ6


class Tanh:  # monotone control link (information exponent 1): P3
    name = 'tanh'
    @staticmethod
    def f(z): return torch.tanh(2 * z)
    @staticmethod
    def df(z): return 2 / torch.cosh(2 * z) ** 2
    @staticmethod
    def d2f(z):
        t = torch.tanh(2 * z)
        return -8 * t * (1 - t ** 2)


LINKS = {'he3': He3, 'tanh': Tanh}


# ---------------------------------------------------------------- tasks
def make_tasks(E, d, n, nq, link, sigma, seed, device='cpu', dtype=torch.float64):
    g = torch.Generator(device='cpu').manual_seed(seed)
    u = torch.randn(E, d, generator=g, dtype=dtype)
    u = u / u.norm(dim=1, keepdim=True)
    X = torch.randn(E, n, d, generator=g, dtype=dtype)
    Xq = torch.randn(E, nq, d, generator=g, dtype=dtype)
    y = link.f(torch.einsum('end,ed->en', X, u)) + sigma * torch.randn(E, n, generator=g, dtype=dtype)
    yq = link.f(torch.einsum('end,ed->en', Xq, u))  # clean query target (evaluator only)
    mv = lambda t: t.to(device)
    return dict(u=mv(u), X=mv(X), y=mv(y), Xq=mv(Xq), yq=mv(yq))


def init_w(E, d, R, seed, device, dtype):
    g = torch.Generator(device='cpu').manual_seed(seed)
    w = torch.randn(R, E, d, generator=g, dtype=dtype) / math.sqrt(d)
    return w.to(device)


def support_loss(link, X, y, w):  # w: (..., E, d)
    return ((link.f(torch.einsum('end,...ed->...en', X, w)) - y) ** 2).mean(-1)


def pick_best(link, X, y, W):
    """W: (R, E, d) candidates -> (E, d) chosen by support loss only."""
    L = support_loss(link, X, y, W)
    L = torch.nan_to_num(L, nan=float('inf'))
    idx = L.argmin(0)
    return W[idx, torch.arange(W.shape[1])]


# ---------------------------------------------------------------- BP on f_w
def bp_gd(link, X, y, W0, T, lr):
    W = W0.clone(); n = X.shape[1]
    for _ in range(T):
        s = torch.einsum('end,red->ren', X, W)
        r = link.f(s) - y
        grad = 2 / n * torch.einsum('ren,end->red', r * link.df(s), X)
        W = W - lr * grad
        W = torch.nan_to_num(W, nan=0.0).clamp(-50, 50)
    return W


def bp_adam(link, X, y, W0, T, lr, b1=0.9, b2=0.999, eps=1e-8):
    W = W0.clone(); m = torch.zeros_like(W); v = torch.zeros_like(W); n = X.shape[1]
    for t in range(1, T + 1):
        s = torch.einsum('end,red->ren', X, W)
        r = link.f(s) - y
        grad = 2 / n * torch.einsum('ren,end->red', r * link.df(s), X)
        m = b1 * m + (1 - b1) * grad; v = b2 * v + (1 - b2) * grad ** 2
        W = W - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
        W = torch.nan_to_num(W, nan=0.0).clamp(-50, 50)
    return W


def bp_lm(link, X, y, W0, T, mu0=1.0):
    """Levenberg-Marquardt / damped Gauss-Newton on the full parameter w."""
    W = W0.clone(); R, E, d = W.shape
    mu = torch.full((R, E), mu0, dtype=W.dtype, device=W.device)
    eye = torch.eye(d, dtype=W.dtype, device=W.device)
    L = support_loss(link, X, y, W)
    for _ in range(T):
        s = torch.einsum('end,red->ren', X, W)
        r = y - link.f(s)
        J = link.df(s).unsqueeze(-1) * X.unsqueeze(0)          # (R,E,n,d)
        A = torch.einsum('rend,renk->redk', J, J) + mu[..., None, None] * eye
        b = torch.einsum('rend,ren->red', J, r)
        step = torch.linalg.solve(A, b.unsqueeze(-1)).squeeze(-1)
        Wn = W + step
        Ln = torch.nan_to_num(support_loss(link, X, y, Wn), nan=float('inf'))
        ok = Ln < L
        W = torch.where(ok[..., None], Wn, W)
        L = torch.where(ok, Ln, L)
        mu = torch.where(ok, mu / 3, mu * 3).clamp(1e-8, 1e8)
    return W


# ---------------------------------------------------------------- lifted (PC / PC-ALM)
def _activity_step(link, y, a, rho, grid):
    """Exact 1-D minimisation of (g(s)-y)^2 + rho/2 (s-a)^2 per example.

    Global grid search followed by safeguarded Newton refinement.
    y, a: (..., n); rho scalar or broadcastable.
    """
    G = grid.view(*([1] * y.dim()), -1)
    F = (link.f(G) - y.unsqueeze(-1)) ** 2 + 0.5 * rho * (G - a.unsqueeze(-1)) ** 2
    s = grid[F.argmin(-1)]
    for _ in range(6):
        gs, dg, d2g = link.f(s), link.df(s), link.d2f(s)
        F1 = 2 * (gs - y) * dg + rho * (s - a)
        F2 = 2 * (dg ** 2 + (gs - y) * d2g) + rho
        step = torch.where(F2 > 1e-9, F1 / F2, 0.1 * F1)
        s = s - step.clamp(-0.25, 0.25)
    return s


def _activity_grad(link, y, a, rho, s, K=20, lr=0.05, lim=None):
    """Official-PC-ALM-style activity inference: K local gradient steps from the current activity."""
    for _ in range(K):
        F1 = 2 * (link.f(s) - y) * link.df(s) + rho * (s - a)
        L = (GRID_LIM if lim is None else lim) + 1
        s = (s - lr * F1.clamp(-20, 20)).clamp(-L, L)
    return s


def lifted(link, X, y, W0, T, rho, use_multiplier=True, rho_growth=1.0, mu=1e-2, grid_pts=241,
           activity='exact', grid_lim=None):
    """Lifted local credit: activities s_i are free variables tied by s_i = w.x_i.

    use_multiplier=True  -> PC-ALM (augmented Lagrangian, multiplier ascent)
    use_multiplier=False -> PC with quadratic penalty only (optionally rho continuation)
    activity='exact' -> global 1-D minimisation per example (enabled by locality)
    activity='grad'  -> local gradient inference as in the official PC-ALM code
    Parameter block is the exact local least-squares minimiser (no chain rule).
    """
    R, E, d = W0.shape
    lim = GRID_LIM if grid_lim is None else grid_lim
    if grid_lim == 'auto':   # cover every preimage of the observed support targets (support-only information)
        cand = torch.linspace(1, 60, 1181, dtype=X.dtype, device=X.device)
        ok = torch.minimum(link.f(cand).abs(), link.f(-cand).abs()) > 1.1 * y.abs().max() + 1
        lim = max(GRID_LIM, float(cand[ok][0]) if ok.any() else 60.0)
    grid = torch.linspace(-lim, lim, int(grid_pts * lim / 5), dtype=X.dtype, device=X.device)
    XtX = torch.einsum('end,enk->edk', X, X)
    eye = torch.eye(d, dtype=X.dtype, device=X.device)
    W = W0.clone()
    lam = torch.zeros(R, E, X.shape[1], dtype=X.dtype, device=X.device)
    s = torch.einsum('end,red->ren', X, W)
    r = rho
    for _ in range(T):
        pred = torch.einsum('end,red->ren', X, W)
        a = pred - lam / r
        if activity == 'exact':
            s = _activity_step(link, y, a, r, grid)
        else:
            s = _activity_grad(link, y, a, r, s, lim=lim)
        if use_multiplier:
            lam = lam + r * (s - pred)
        A = r * (XtX + mu * eye)                                # (E,d,d)
        b = torch.einsum('end,ren->red', X, r * s + lam)
        W = torch.linalg.solve(A.unsqueeze(0).expand(R, -1, -1, -1), b.unsqueeze(-1)).squeeze(-1)
        r = r * rho_growth
    return W


# ---------------------------------------------------------------- closed-form / fixed features
def _dual_ridge_loo(K, y, Kq, lams):
    """Kernel ridge with exact LOO selection over lams. K:(E,n,n) Kq:(E,nq,n)."""
    evals, evecs = torch.linalg.eigh(K)
    best_err = torch.full(y.shape[:1], float('inf'), dtype=K.dtype, device=K.device)
    best_pred = torch.zeros(Kq.shape[:2], dtype=K.dtype, device=K.device)
    Uy = torch.einsum('enk,en->ek', evecs, y)
    for lam in lams:
        inv = 1.0 / (evals.clamp_min(0) + lam)
        alpha = torch.einsum('enk,ek->en', evecs, inv * Uy)
        Hdiag = torch.einsum('enk,ek,enk->en', evecs, inv, evecs)   # diag((K+lam)^-1)
        loo = (alpha / Hdiag) ** 2
        err = loo.mean(-1)
        better = err < best_err
        pred = torch.einsum('eqn,en->eq', Kq, alpha)
        best_pred = torch.where(better[:, None], pred, best_pred)
        best_err = torch.where(better, err, best_err)
    return best_pred


LAMS = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def linear_ridge(X, y, Xq):
    one = torch.ones_like(X[..., :1]); oq = torch.ones_like(Xq[..., :1])
    Z, Zq = torch.cat([X, one], -1), torch.cat([Xq, oq], -1)
    return _dual_ridge_loo(Z @ Z.transpose(1, 2), y, Zq @ Z.transpose(1, 2), LAMS)


def features_ridge(F, Fq, y):
    return _dual_ridge_loo(F @ F.transpose(1, 2), y, Fq @ F.transpose(1, 2), LAMS)


def random_features(X, Xq, m, kind, link, seed):
    d = X.shape[-1]
    g = torch.Generator(device='cpu').manual_seed(seed)
    V = torch.randn(d, m, generator=g, dtype=X.dtype)
    V = (V / V.norm(dim=0, keepdim=True)).to(X.device)
    if kind == 'relu':
        b = (torch.rand(m, generator=g, dtype=X.dtype) * 2 - 1).to(X.device)
        act = lambda t: torch.relu(t @ V + b)
    else:  # same link g as the model: matched prior
        act = lambda t: link.f(t @ V)
    return act(X) / math.sqrt(m), act(Xq) / math.sqrt(m)


def rbf_krr(X, y, Xq):
    d = X.shape[-1]
    best = None
    D = torch.cdist(X, X) ** 2; Dq = torch.cdist(Xq, X) ** 2
    preds, errs = [], []
    for h in [0.5, 1.0, 2.0, 4.0]:
        bw = h * math.sqrt(d)
        K, Kq = torch.exp(-D / (2 * bw ** 2)), torch.exp(-Dq / (2 * bw ** 2))
        evals, evecs = torch.linalg.eigh(K)
        Uy = torch.einsum('enk,en->ek', evecs, y)
        for lam in LAMS:
            inv = 1.0 / (evals.clamp_min(0) + lam)
            alpha = torch.einsum('enk,ek->en', evecs, inv * Uy)
            Hd = torch.einsum('enk,ek,enk->en', evecs, inv, evecs)
            errs.append(((alpha / Hd) ** 2).mean(-1)); preds.append(torch.einsum('eqn,en->eq', Kq, alpha))
    errs = torch.stack(errs); preds = torch.stack(preds)
    idx = errs.argmin(0)
    return preds[idx, torch.arange(X.shape[0])]


def poly3_krr(X, y, Xq):
    d = X.shape[-1]
    K = (1 + X @ X.transpose(1, 2) / d) ** 3
    Kq = (1 + Xq @ X.transpose(1, 2) / d) ** 3
    return _dual_ridge_loo(K, y, Kq, LAMS)


def mlp_extra_layer(X, y, Xq, width=256, T=500, lr=1e-2, wd=1e-4, seed=0):
    """'Add one more layer' baseline: 2-layer ReLU MLP fit on the support by Adam."""
    E, n, d = X.shape
    g = torch.Generator(device='cpu').manual_seed(seed)
    W1 = (torch.randn(E, d, width, generator=g, dtype=X.dtype) / math.sqrt(d)).to(X.device).requires_grad_()
    b1 = torch.zeros(E, 1, width, dtype=X.dtype, device=X.device, requires_grad=True)
    a = (torch.randn(E, width, generator=g, dtype=X.dtype) / math.sqrt(width)).to(X.device).requires_grad_()
    c = torch.zeros(E, 1, dtype=X.dtype, device=X.device, requires_grad=True)
    opt = torch.optim.Adam([W1, b1, a, c], lr=lr, weight_decay=wd)
    f = lambda Z: torch.einsum('enh,eh->en', torch.relu(Z @ W1 + b1), a) + c
    for _ in range(T):
        opt.zero_grad(); ((f(X) - y) ** 2).mean().mul(E).backward(); opt.step()
    with torch.no_grad():
        return f(Xq)


def hybrid_transform_init(link, X, y, sigma, mu=1e-2, zlim=6.0):
    """Strong closed-form non-CSQ init: regress E[z|y] (1-D quadrature) on x, rescale to unit norm."""
    zg = torch.linspace(-zlim, zlim, int(400 * zlim) + 1, dtype=X.dtype, device=X.device)
    logp = -0.5 * zg ** 2 - (y.unsqueeze(-1) - link.f(zg)) ** 2 / (2 * sigma ** 2)
    p = torch.softmax(logp, -1)
    t = (p * zg).sum(-1)
    d = X.shape[-1]
    A = torch.einsum('end,enk->edk', X, X) + mu * torch.eye(d, dtype=X.dtype, device=X.device)
    w = torch.linalg.solve(A, torch.einsum('end,en->ed', X, t).unsqueeze(-1)).squeeze(-1)
    return w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)


# ---------------------------------------------------------------- evaluation
def query_mse_w(link, Xq, yq, w):
    return ((link.f(torch.einsum('eqd,ed->eq', Xq, w)) - yq) ** 2).mean(-1)


def overlap(w, u):
    return (w * u).sum(-1) / w.norm(dim=-1).clamp_min(1e-12)


class Timer:
    def __init__(self): self.t = {}
    def __call__(self, name):
        tm = self
        class _C:
            def __enter__(s):
                if torch.cuda.is_available(): torch.cuda.synchronize()
                s.t0 = time.perf_counter()
            def __exit__(s, *a):
                if torch.cuda.is_available(): torch.cuda.synchronize()
                tm.t[name] = tm.t.get(name, 0) + time.perf_counter() - s.t0
        return _C()
