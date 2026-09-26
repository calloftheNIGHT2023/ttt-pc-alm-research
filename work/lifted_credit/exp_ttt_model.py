"""Exp M1 (report 448): end-to-end trained TTT layers with different inner-loop updates.

In-context learning task (Garg et al. 2022 style): each sequence has n context pairs (o_i, y_i) and 64 queries.
o = M [x; z] in R^{2d} (x signal, z nuisance, M fixed unknown orthogonal mixing), y = He3(u.x)/sqrt6 + 0.1 eps with u
random per sequence. The outer loop must learn the key projection theta_K (d x 2d); the inner loop must find u.

All TTT variants share theta_K and the inner memory f(k; w) = He3(w.k)/sqrt6 (ridge: linear memory).
  ttt_gd_official   one pass over the context in chunks of 16, one GD step per chunk (learned step size)
  ttt_gd_full       K full-batch GD steps on the whole context (learned step size) -- stronger than official
  ttt_ridge         closed-form linear memory w = (K^T K + mu I)^-1 K^T y (MesaNet/RLS-optimal linear TTT)
  ttt_pcalm         K lifted PC-ALM iterations (per-token 1-D inversion, multipliers, local LS; learned rho)
  transformer       4-layer encoder on [o, y] tokens, queries attend to context
  oracle_pcalm      theta_K fixed to the true signal projection, PC-ALM inner loop, no training (upper reference)
Outer training: Adam, same steps and data for every model; gradients through the unrolled inner loop (the grid argmin
of the PC-ALM inversion is not differentiated; the Newton refinement and LS steps are). Query labels are used only as
the outer training loss on training sequences and for evaluation on fresh sequences.
"""
import argparse
import json
import math
import time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

ap = argparse.ArgumentParser()
ap.add_argument('--d', type=int, default=32)
ap.add_argument('--n_over_d', type=float, default=4)
ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--K', type=int, default=30, help='inner iterations for ttt_gd_full / ttt_pcalm')
ap.add_argument('--lr', type=float, default=3e-3)
ap.add_argument('--decorr', type=int, default=1, help='anchor-decorrelated first step (GPT credit_mechanism)')
ap.add_argument('--rho_init', type=float, default=0.1)
ap.add_argument('--learn_rho', type=int, default=1, help='0: rho is a fixed solver hyper-parameter (v2, report 448)')
ap.add_argument('--grid_pts', type=int, default=241)
ap.add_argument('--models', nargs='+', default=['oracle_pcalm', 'ttt_pcalm', 'ttt_gd_full', 'ttt_gd_official', 'ttt_ridge', 'transformer'])
ap.add_argument('--eval_n_over_d', type=float, nargs='+', default=[2, 3, 4, 6])
ap.add_argument('--eval_seqs', type=int, default=2048)
ap.add_argument('--seed', type=int, default=448001)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dev, dt = args.device, torch.float32
d, D = args.d, 2 * args.d
SQ6 = math.sqrt(6.0)
g = lambda z: (z ** 3 - 3 * z) / SQ6
dg = lambda z: (3 * z ** 2 - 3) / SQ6
d2g = lambda z: 6 * z / SQ6

gen0 = torch.Generator().manual_seed(args.seed)
M = torch.linalg.qr(torch.randn(D, D, generator=gen0))[0].to(dev).to(dt)       # fixed unknown mixing
theta_true = M.T[:d].contiguous()                                                # x = theta_true @ o


def sample(B, n, nq, gen):
    x = torch.randn(B, n + nq, d, generator=gen, device=dev); z = torch.randn(B, n + nq, d, generator=gen, device=dev)
    o = torch.cat([x, z], -1) @ M.T
    u = torch.randn(B, d, generator=gen, device=dev); u = u / u.norm(dim=-1, keepdim=True)
    yc = g(torch.einsum('bnd,bd->bn', x, u))
    y = yc + 0.1 * torch.randn(B, n + nq, generator=gen, device=dev)
    return o[:, :n], y[:, :n], o[:, n:], yc[:, n:]


GRID = torch.linspace(-5, 5, args.grid_pts, device=dev, dtype=dt)


def invert(y, a, rho):
    """argmin_s (g(s)-y)^2 + rho/2 (s-a)^2: global grid start (not differentiated) + 4 differentiable Newton steps."""
    with torch.no_grad():
        F = (g(GRID) - y.unsqueeze(-1)) ** 2 + 0.5 * rho.detach() * (GRID - a.unsqueeze(-1)) ** 2
        s = GRID[F.argmin(-1)]
    for _ in range(4):
        gs, g1, g2_ = g(s), dg(s), d2g(s)
        F1 = 2 * (gs - y) * g1 + rho * (s - a)
        F2 = 2 * (g1 ** 2 + (gs - y) * g2_) + rho
        s = s - torch.where(F2 > 1e-6, F1 / F2.clamp_min(1e-6), 0.1 * F1).clamp(-0.25, 0.25)
    return s


class TTT(nn.Module):
    def __init__(self, kind, oracle=False):
        super().__init__()
        self.kind = kind
        th = torch.randn(d, D, generator=torch.Generator().manual_seed(args.seed + 1))
        th = th / th.norm(dim=-1, keepdim=True)          # unit-norm rows: keys start with unit variance
        self.theta = nn.Parameter(theta_true.clone() if oracle else th.to(dev), requires_grad=not oracle)
        self.log_step = nn.Parameter(torch.tensor(math.log(0.05), device=dev))     # GD step size
        self.log_rho = nn.Parameter(torch.tensor(math.log(args.rho_init), device=dev), requires_grad=bool(args.learn_rho))  # PC-ALM penalty
        self.log_mu = nn.Parameter(torch.tensor(math.log(1e-2), device=dev))       # ridge

    def forward(self, o, y, oq, w0):
        K = o @ self.theta.T; Kq = oq @ self.theta.T
        B, n, _ = K.shape
        if self.kind == 'ridge':
            A = K.transpose(1, 2) @ K + self.log_mu.exp() * torch.eye(d, device=dev)
            w = torch.linalg.solve(A, (K.transpose(1, 2) @ y.unsqueeze(-1))).squeeze(-1)
            return torch.einsum('bqd,bd->bq', Kq, w)
        w = w0
        if self.kind == 'gd_official':
            eta = self.log_step.exp()
            for c in range(0, n, 16):
                Kc, yc = K[:, c:c + 16], y[:, c:c + 16]
                s = torch.einsum('bnd,bd->bn', Kc, w)
                w = w - eta * 2 / Kc.shape[1] * torch.einsum('bn,bnd->bd', (g(s) - yc) * dg(s), Kc)
                w = w.clamp(-20, 20)
        elif self.kind == 'gd_full':
            eta = self.log_step.exp()
            for _ in range(args.K):
                s = torch.einsum('bnd,bd->bn', K, w)
                w = (w - eta * 2 / n * torch.einsum('bn,bnd->bd', (g(s) - y) * dg(s), K)).clamp(-20, 20)
        elif self.kind == 'pcalm':
            rho = self.log_rho.exp()
            A = K.transpose(1, 2) @ K + 1e-2 * torch.eye(d, device=dev)
            L = torch.linalg.cholesky(A)
            lam = torch.zeros(B, n, device=dev)
            if args.decorr:
                # anchor-decorrelated first step: remove the random-anchor component b*a from the inverted
                # activities before the first LS (support-only; GPT credit_mechanism/THEORY.md), then rescale so
                # that Var(K w) = 1 (the unit-variance latent of the task family) and restart with zero multipliers
                a0 = torch.einsum('bnd,bd->bn', K, w)
                s0 = invert(y, a0, rho)
                bh = (s0 * a0).sum(-1, keepdim=True) / (a0 * a0).sum(-1, keepdim=True).clamp_min(1e-8)
                w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s0 - bh * a0).unsqueeze(-1), L).squeeze(-1)
                w = w / torch.einsum('bnd,bd->bn', K, w).std(-1, keepdim=True).clamp_min(1e-6)
            for _ in range(args.K):
                pred = torch.einsum('bnd,bd->bn', K, w)
                s = invert(y, pred - lam / rho, rho)
                lam = lam + rho * (s - pred)
                w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L).squeeze(-1)
        return g(torch.einsum('bqd,bd->bq', Kq, w))


class Transformer(nn.Module):
    def __init__(self, dm=128, layers=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(D + 2, dm)
        enc = nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=0.0, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(dm, 1)

    def forward(self, o, y, oq, w0):
        B, n, _ = o.shape; nq = oq.shape[1]
        ctx = torch.cat([o, y.unsqueeze(-1), torch.zeros_like(y).unsqueeze(-1)], -1)
        qry = torch.cat([oq, torch.zeros(B, nq, 1, device=dev), torch.ones(B, nq, 1, device=dev)], -1)
        h = self.inp(torch.cat([ctx, qry], 1))
        L = n + nq
        mask = torch.zeros(L, L, dtype=torch.bool, device=dev)
        mask[:, n:] = True                                     # nobody attends to query tokens ...
        mask[torch.arange(n, L), torch.arange(n, L)] = False   # ... except each query to itself
        h = self.enc(h, mask=mask)
        return self.head(h[:, n:]).squeeze(-1)


def w_init(B, gen):
    return 0.1 * torch.randn(B, d, generator=gen, device=dev) / math.sqrt(d)


def subspace_overlap(theta):
    Qt = torch.linalg.qr(theta_true.T)[0]; Qk = torch.linalg.qr(theta.detach().T)[0]
    return float((Qt.T @ Qk).pow(2).sum() / d)          # mean cos^2 of principal angles, 1 = same subspace


def build(name):
    return {'oracle_pcalm': lambda: TTT('pcalm', oracle=True), 'ttt_pcalm': lambda: TTT('pcalm'),
            'ttt_gd_full': lambda: TTT('gd_full'), 'ttt_gd_official': lambda: TTT('gd_official'),
            'ttt_ridge': lambda: TTT('ridge'), 'transformer': lambda: Transformer()}[name]().to(dev)


n_train = int(args.n_over_d * d)
results = {}
for name in args.models:
    torch.manual_seed(args.seed + 11)
    model = build(name)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr) if params else None
    gen = torch.Generator(device=dev).manual_seed(args.seed + 100)
    curve = []; t0 = time.perf_counter()
    steps = 0 if name == 'oracle_pcalm' else args.steps
    for step in range(steps):
        o, y, oq, yq = sample(args.batch, n_train, 64, gen)
        pred = model(o, y, oq, w_init(args.batch, gen))
        loss = ((pred - yq) ** 2).mean()
        opt.zero_grad(); (loss if torch.isfinite(loss) else torch.zeros((), device=dev, requires_grad=True)).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if step % 100 == 0 or step == steps - 1:
            ov = subspace_overlap(model.theta) if hasattr(model, 'theta') else None
            curve.append(dict(step=step, loss=float(loss), subspace_overlap=ov))
            print(f'{name} step {step} loss {float(loss):.4f} overlap {ov}', flush=True)
    train_s = time.perf_counter() - t0
    model.eval(); ev = {}
    egen = torch.Generator(device=dev).manual_seed(args.seed + 999)
    with torch.no_grad():
        for nd in args.eval_n_over_d:
            n = int(nd * d); errs = []
            for _ in range(args.eval_seqs // 256):
                o, y, oq, yq = sample(256, n, 64, egen)
                p = model(o, y, oq, w_init(256, egen))
                errs.append(((p - yq) ** 2).mean(-1) / yq.var(-1).clamp_min(1e-6))
            e = torch.nan_to_num(torch.cat(errs), nan=1e6)
            ev[str(nd)] = dict(nmse_mean=float(e.clamp_max(1e3).mean()), nmse_median=float(e.median()),
                               success=float((e < .05).float().mean()))
    results[name] = dict(train_seconds=train_s, curve=curve, eval=ev,
                         subspace_overlap=subspace_overlap(model.theta) if hasattr(model, 'theta') else None,
                         params=sum(p.numel() for p in model.parameters()),
                         learned=dict(step=float(model.log_step.exp()), rho=float(model.log_rho.exp()), mu=float(model.log_mu.exp()))
                         if hasattr(model, 'log_step') else None)
    print(f'== {name}: ' + '  '.join(f"n={k}d: succ {v['success']:.2f} nmse_med {v['nmse_median']:.3f}" for k, v in ev.items())
          + f"  overlap {results[name]['subspace_overlap']}  train {train_s:.0f}s", flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=1))
print('DONE')
