"""Exp R-CV (report 449): end-to-end trained TTT layers for compressive phase retrieval of REAL images.

Each sequence is one real CIFAR-10 image x (grayscale 32x32, unit norm). Context: n random Gaussian measurement vectors
a_i and amplitudes y_i = |a_i . x| (+ small noise); queries: 64 new measurement vectors, target |a_q . x|.
Keys k_i = theta a_i with a LEARNED image basis theta (m x 1024), so a_i . x ~ (theta a_i) . c for the image code c:
the inner memory |w . k| must find w = c in-context (sign lost: multi-branch). Outer training on the CIFAR-10 TRAIN
split, evaluation on held-out TEST-split images only. n is far below the 1024 pixels (compressive regime).
Models: ttt_pcalm, ttt_gd_full, ttt_gd_official, ttt_ridge, ttt_quad_ridge, transformer (as in exp_real_ttt.py), and
pca_pcalm_reference (theta = top-m PCA basis of training images, PC-ALM inner loop, no outer training).
--v3 (report 450, noise gap): the image energy outside the m-dim code acts as noise INSIDE the |.|
(y = |k.c + a.x_perp|). Adds ttt_pcalm_sigma / ttt_gd_soft (see exp_real_ttt.py), their fixed-PCA versions
pca_pcalm_sigma / pca_gd_soft (only scalars trained), the folded-normal readout for all |w.k| memories, and
--theta_init pca (every learned basis starts from the PCA basis; applies to all TTT models alike).
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

ap = argparse.ArgumentParser()
ap.add_argument('--m', type=int, default=64, help='latent image-basis dimension (key dim)')
ap.add_argument('--n_over_m', type=float, default=3)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--batch', type=int, default=32)
ap.add_argument('--K', type=int, default=30)
ap.add_argument('--lr', type=float, default=3e-3)
ap.add_argument('--sigma', type=float, default=0.01)
ap.add_argument('--models', nargs='+', default=['pca_pcalm_reference', 'ttt_pcalm', 'ttt_gd_full', 'ttt_gd_official', 'ttt_ridge', 'ttt_quad_ridge', 'transformer'])
ap.add_argument('--eval_n_over_m', type=float, nargs='+', default=[2.0, 3.0, 4.0])
ap.add_argument('--eval_seqs', type=int, default=1024)
ap.add_argument('--seed', type=int, default=449101)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--df_correct', action='store_true', help='report 450: unbiased noise estimate RSS/(n - df), df = tr((G+mu I)^-1 G)')
ap.add_argument('--ens_temps', type=float, nargs='*', default=[], help='chain closure (R2): also evaluate the R-chain mixture at these '
                'weight temperatures T (weights softmax(-n/2 (r2/min r2 - 1)/T); T=0 means uniform averaging)')
ap.add_argument('--ens', type=int, default=1, help='report 450: evaluate also with R chains from different inits, weighted by in-context likelihood')
ap.add_argument('--prox_noise', action='store_true', help='report 450: proximal weight also scales with r2/E[y^2], so it vanishes in the exact regime')
ap.add_argument('--snr_rule', default='none', choices=['none', 'leak', 'both'],
                help='report 450: scale the slack variance (leak) and optionally the branch temperature by the in-context '
                     'noise-to-signal ratio nu/(1-nu), nu = r2/E[y^2] (df-corrected): exact regime -> hard ALM, noisy -> relaxed')
ap.add_argument('--val_train', action='store_true', help='report 450: also evaluate on TRAIN-split images (model selection without test data)')
ap.add_argument('--g_init', type=float, default=1.0, help='initial multiplier leak (slack variance / residual variance); large = near no multiplier')
ap.add_argument('--v3', action='store_true')
ap.add_argument('--theta_init', default='random', choices=['random', 'pca'])
ap.add_argument('--out', required=True)
args = ap.parse_args()
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dev, dt = args.device, torch.float32
m = args.m

from datasets import load_dataset


def load_split(split):
    ds = load_dataset('uoft-cs/cifar10', split=split)
    X = np.stack([np.asarray(r['img'].convert('L'), dtype=np.float32).reshape(-1) / 255.0 for r in ds])
    return X / np.linalg.norm(X, axis=1, keepdims=True)


Xtr = torch.tensor(load_split('train'), device=dev); Xte = torch.tensor(load_split('test'), device=dev)
P = Xtr.shape[1]
U, S, Vh = torch.linalg.svd(Xtr[:20000] - 0, full_matrices=False)
pca_basis = Vh[:m]                                          # uncentred PCA basis of training images (m x P)
proj_err = float(((Xte - (Xte @ pca_basis.T) @ pca_basis) ** 2).sum(1).mean())
print(f'PCA-{m} residual energy on test images: {proj_err:.4f}', flush=True)


def soft_activity(y, a, rho, tau=None):
    sp = ((2 * y + rho * a) / (2 + rho)).clamp_min(0); sm = ((-2 * y + rho * a) / (2 + rho)).clamp_max(0)
    Fv = lambda s_: (s_.abs() - y) ** 2 + 0.5 * rho * (s_ - a) ** 2
    if tau is None:
        return torch.where(Fv(sp) <= Fv(sm), sp, sm)
    wp = torch.sigmoid((Fv(sm) - Fv(sp)) / tau)
    return wp * sp + (1 - wp) * sm


def foldmean(mu_, sd):
    return sd * math.sqrt(2 / math.pi) * torch.exp(-mu_ ** 2 / (2 * sd ** 2)) + mu_ * torch.erf(mu_ / (sd * math.sqrt(2)))


def sample(B, n, nq, pool, gen):
    x = pool[torch.randint(len(pool), (B,), generator=gen, device=dev)]
    a = torch.randn(B, n + nq, P, generator=gen, device=dev)
    z = torch.einsum('bnp,bp->bn', a, x)
    y = (z.abs() + args.sigma * torch.randn(B, n + nq, generator=gen, device=dev)).clamp_min(0)
    return a[:, :n], y[:, :n], a[:, n:], z[:, n:].abs()


class TTT(nn.Module):
    def __init__(self, kind, fixed=None):
        super().__init__()
        self.kind = kind
        th = torch.randn(m, P, generator=torch.Generator().manual_seed(args.seed + 1)); th = th / th.norm(dim=-1, keepdim=True)
        if fixed is None and args.theta_init == 'pca':
            th = pca_basis.detach().cpu()
        self.theta = nn.Parameter(fixed.clone() if fixed is not None else th.to(dev), requires_grad=fixed is None)
        self.log_step = nn.Parameter(torch.tensor(math.log(0.05), device=dev))
        self.log_mu = nn.Parameter(torch.tensor(math.log(1e-1), device=dev))
        self.log_mu_pc = nn.Parameter(torch.tensor(math.log(1e-2), device=dev), requires_grad=args.v3)
        self.cal = nn.Parameter(torch.tensor([1.0, 0.0], device=dev), requires_grad=args.v3)
        self.log_c = nn.Parameter(torch.tensor(0.0, device=dev))
        self.log_g = nn.Parameter(torch.tensor(math.log(args.g_init), device=dev))
        self.log_sr = nn.Parameter(torch.tensor(0.0, device=dev))
        self.log_prox = nn.Parameter(torch.tensor(0.0, device=dev))    # proximal weight / mean eigenvalue of K^T K
        self.log_kappa = nn.Parameter(torch.tensor(math.log(2.0), device=dev))  # Huber threshold (residual std units)

    def forward(self, a, y, aq, w0):
        K = a @ self.theta.T; Kq = aq @ self.theta.T
        B, n, _ = K.shape
        I = lambda k: torch.eye(k, device=dev)
        if self.kind == 'ridge':
            K1 = torch.cat([K, torch.ones(B, n, 1, device=dev)], -1); Kq1 = torch.cat([Kq, torch.ones(B, Kq.shape[1], 1, device=dev)], -1)
            w = torch.linalg.solve(K1.transpose(1, 2) @ K1 + self.log_mu.exp() * I(m + 1), K1.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1)
            return torch.einsum('bqd,bd->bq', Kq1, w)
        if self.kind == 'quad':
            iu = torch.triu_indices(m, m, device=dev)
            feat = lambda M: (M.unsqueeze(-1) * M.unsqueeze(-2))[..., iu[0], iu[1]]
            F, Fq = feat(K), feat(Kq)
            alpha = torch.linalg.solve(F @ F.transpose(1, 2) + self.log_mu.exp() * I(n), (y ** 2).unsqueeze(-1)).squeeze(-1)
            return torch.einsum('bqp,bnp,bn->bq', Fq, F, alpha).clamp_min(0).sqrt()
        w = w0
        if self.kind == 'gd_official':
            eta = self.log_step.exp()
            for c in range(0, n, 16):
                Kc, yc = K[:, c:c + 16], y[:, c:c + 16]
                s = torch.einsum('bnd,bd->bn', Kc, w)
                w = (w - eta * 2 / Kc.shape[1] * torch.einsum('bn,bnd->bd', (s.abs() - yc) * torch.sign(s), Kc)).clamp(-50, 50)
        elif self.kind == 'gd_full':
            eta = self.log_step.exp()
            for _ in range(args.K):
                s = torch.einsum('bnd,bd->bn', K, w)
                w = (w - eta * 2 / n * torch.einsum('bn,bnd->bd', (s.abs() - y) * torch.sign(s), K)).clamp(-50, 50)
        elif self.kind == 'pcalm':
            rho = 1.0
            L = torch.linalg.cholesky(K.transpose(1, 2) @ K + self.log_mu_pc.exp() * I(m))
            lam = torch.zeros(B, n, device=dev)
            for _ in range(args.K):
                pred = torch.einsum('bnd,bd->bn', K, w); av = pred - lam / rho
                sp = ((2 * y + rho * av) / (2 + rho)).clamp_min(0); sm = ((-2 * y + rho * av) / (2 + rho)).clamp_max(0)
                Fv = lambda s_: (s_.abs() - y) ** 2 + 0.5 * rho * (s_ - av) ** 2
                s = torch.where(Fv(sp) <= Fv(sm), sp, sm)
                lam = lam + rho * (s - pred)
                w = torch.cholesky_solve(torch.einsum('bnd,bn->bd', K, s + lam / rho).unsqueeze(-1), L).squeeze(-1)
        elif self.kind in ('pcalm_sigma', 'pcalm_prox', 'pcalm_prox_nolam', 'pcalm_robust'):
            # pcalm_robust (report 450, 3rd gap): Huber instead of Gaussian slack. The multiplier update becomes the prox of
            # the conjugate, lam+ = clip((lam + rho r)/(1 + rho g r2), +-kappa/(g sqrt(r2))): leaky AND bounded per example,
            # so heavy-tailed outliers cannot accumulate in lam. kappa -> inf gives pcalm_prox, r2 -> 0 the original PC-ALM.
            # pcalm_prox (report 450, 2nd gap): proximal LS parameter step
            #   w+ = argmin |s + lam/rho - K w|^2 + mu |w|^2 + p |w - w_t|^2,  p = prox * tr(K^T K)/dim (learned prox)
            # p -> 0 is the exact LS step of pcalm_sigma; large p is an implicit gradient step toward the same soft target.
            rho = 1.0
            G = K.transpose(1, 2) @ K
            p = (self.log_prox.exp() * G.diagonal(dim1=1, dim2=2).mean(-1))[:, None] if self.kind != 'pcalm_sigma' else torch.zeros(B, 1, device=dev)
            p0 = p
            L = torch.linalg.cholesky(G + (self.log_mu_pc.exp() + p)[:, :, None] * I(m))
            lam = torch.zeros(B, n, device=dev)
            infl = torch.ones(B, 1, device=dev)
            if args.df_correct:                      # in-context residuals shrink as the lifted fit absorbs noise: correct by df
                with torch.no_grad():
                    Lmu = torch.linalg.cholesky(G + self.log_mu_pc.exp() * I(m))
                    df = m - self.log_mu_pc.exp() * torch.cholesky_inverse(Lmu).diagonal(dim1=1, dim2=2).sum(-1)
                    infl = (n / (n - df).clamp_min(1.0))[:, None]
            for _ in range(args.K):
                pred = torch.einsum('bnd,bd->bn', K, w)
                r2 = infl * ((y - pred.abs()) ** 2).mean(-1, keepdim=True).detach()
                nsr = torch.ones_like(r2)
                if args.snr_rule != 'none':
                    nu = (r2 / (y ** 2).mean(-1, keepdim=True).clamp_min(1e-8)).clamp(0, 0.95)
                    nsr = nu / (1 - nu)                              # in-context noise-to-signal ratio
                tau = self.log_c.exp() * r2 * (nsr if args.snr_rule == 'both' else 1) + 1e-6
                s = soft_activity(y, pred - lam / rho, rho, tau)
                if self.kind != 'pcalm_prox_nolam':
                    lam = (lam + rho * (s - pred)) / (1 + rho * self.log_g.exp() * r2 * nsr)
                if self.kind == 'pcalm_robust':
                    bound = self.log_kappa.exp() / (self.log_g.exp() * r2.sqrt() + 1e-6)
                    lam = torch.maximum(torch.minimum(lam, bound), -bound)
                if args.prox_noise and self.kind != 'pcalm_sigma':      # every noise term vanishes with the residual
                    p = p0 * (r2 / (y ** 2).mean(-1, keepdim=True).clamp_min(1e-8)).clamp_max(1.0)
                    L = torch.linalg.cholesky(G + (self.log_mu_pc.exp() + p)[:, :, None] * I(m))
                w = torch.cholesky_solve((torch.einsum('bnd,bn->bd', K, s + lam / rho) + p * w).unsqueeze(-1), L).squeeze(-1)
        elif self.kind == 'gd_soft':
            eta = self.log_step.exp()
            for _ in range(args.K):
                pred = torch.einsum('bnd,bd->bn', K, w)
                r2 = ((y - pred.abs()) ** 2).mean(-1, keepdim=True).detach()
                s = soft_activity(y, pred, 1.0, self.log_c.exp() * r2 + 1e-6)
                w = (w - eta * 2 / n * torch.einsum('bn,bnd->bd', pred - s, K)).clamp(-50, 50)
        mq = torch.einsum('bqd,bd->bq', Kq, w)
        self.last_r2 = ((y - torch.einsum('bnd,bd->bn', K, w).abs()) ** 2).mean(-1, keepdim=True).detach()
        if args.v3:
            r2 = ((y - torch.einsum('bnd,bd->bn', K, w).abs()) ** 2).mean(-1, keepdim=True)
            return self.cal[0] * foldmean(mq, (self.log_sr.exp() * r2 + 1e-6).sqrt()) + self.cal[1]
        return mq.abs()


class Transformer(nn.Module):
    """Measurement vectors are 1024-d; they are first compressed by a learned linear map to 128-d tokens."""
    def __init__(self, dm=128, layers=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(P + 2, dm)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=0.0, batch_first=True, norm_first=True), layers)
        self.head = nn.Linear(dm, 1)

    def forward(self, a, y, aq, w0):
        B, n, _ = a.shape; nq = aq.shape[1]
        ctx = torch.cat([a, y.unsqueeze(-1), torch.zeros_like(y).unsqueeze(-1)], -1)
        qry = torch.cat([aq, torch.zeros(B, nq, 1, device=dev), torch.ones(B, nq, 1, device=dev)], -1)
        h = self.inp(torch.cat([ctx, qry], 1)); L = n + nq
        mask = torch.zeros(L, L, dtype=torch.bool, device=dev); mask[:, n:] = True
        mask[torch.arange(n, L), torch.arange(n, L)] = False
        return self.head(self.enc(h, mask=mask)[:, n:]).squeeze(-1)


def w_init(B, gen):
    return 0.1 * torch.randn(B, m, generator=gen, device=dev) / math.sqrt(m)


def evaluate(model, ens=1, pool=None, temp=1.0):
    pool = Xte if pool is None else pool
    egen = torch.Generator(device=dev).manual_seed(args.seed + 999); res = {}
    xgen = torch.Generator(device=dev).manual_seed(args.seed + 4242)   # extra chains: separate stream, eval set unchanged
    with torch.no_grad():
        for nm_ in args.eval_n_over_m:
            n = int(nm_ * m); errs = []
            for _ in range(max(1, args.eval_seqs // 128)):
                a, y, aq, yq = sample(128, n, 64, pool, egen)
                p = model(a, y, aq, w_init(128, egen))
                if ens > 1:          # posterior-weighted mixture over modes: weight_r ~ exp(-n r2_r / (2 min_r r2_r))
                    Ps, R2 = [p], [model.last_r2]
                    for _r in range(ens - 1):
                        Ps.append(model(a, y, aq, w_init(128, xgen))); R2.append(model.last_r2)
                    R2 = torch.cat(R2, -1)
                    if temp == 0:
                        wts = torch.full_like(R2, 1.0 / R2.shape[-1])
                    else:
                        wts = torch.softmax(-0.5 * n * (R2 / R2.min(-1, keepdim=True).values.clamp_min(1e-12) - 1) / temp, -1)
                    p = (torch.stack(Ps, -1) * wts[:, None, :]).sum(-1)
                errs.append(((p - yq) ** 2).mean(-1) / yq.var(-1).clamp_min(1e-8))
            e = torch.nan_to_num(torch.cat(errs), nan=1e3).clamp_max(1e3)
            res[str(float(nm_))] = dict(nmse_mean=float(e.mean()), nmse_median=float(e.median()), success=float((e < .05).float().mean()),
                                        per_seq=[round(float(v), 5) for v in e])
    return res


n_train = int(args.n_over_m * m)
# floor: predict with the true image projected on the PCA-m subspace (the best any m-dim code can do)
with torch.no_grad():
    fgen = torch.Generator(device=dev).manual_seed(args.seed + 5); fe = []
    for _ in range(4):
        idx = torch.randint(len(Xte), (128,), generator=fgen, device=dev); x = Xte[idx]; xp = (x @ pca_basis.T) @ pca_basis
        aq = torch.randn(128, 64, P, generator=fgen, device=dev)
        yq = torch.einsum('bnp,bp->bn', aq, x).abs(); pq = torch.einsum('bnp,bp->bn', aq, xp).abs()
        fe.append(((pq - yq) ** 2).mean(-1) / yq.var(-1))
    fe = torch.cat(fe)
results = dict(pca_residual_energy_test=proj_err, projected_true_image_floor=dict(nmse_mean=float(fe.mean()), nmse_median=float(fe.median())))
print('floor (true image projected on PCA basis):', results['projected_true_image_floor'], flush=True)
for name in args.models:
    t0 = time.perf_counter()
    torch.manual_seed(args.seed + 11)
    model = {'pca_pcalm_reference': lambda: TTT('pcalm', fixed=pca_basis), 'ttt_pcalm': lambda: TTT('pcalm'),
             'pca_gd_full': lambda: TTT('gd_full', fixed=pca_basis), 'pca_quad_ridge': lambda: TTT('quad', fixed=pca_basis),
             'pca_gd_official': lambda: TTT('gd_official', fixed=pca_basis),
             'pca_pcalm_sigma': lambda: TTT('pcalm_sigma', fixed=pca_basis), 'pca_gd_soft': lambda: TTT('gd_soft', fixed=pca_basis),
             'pca_pcalm_v3': lambda: TTT('pcalm', fixed=pca_basis),
             'ttt_pcalm_sigma': lambda: TTT('pcalm_sigma'), 'ttt_gd_soft': lambda: TTT('gd_soft'),
             'ttt_pcalm_prox': lambda: TTT('pcalm_prox'), 'ttt_pcalm_prox_nolam': lambda: TTT('pcalm_prox_nolam'),
             'pca_pcalm_prox': lambda: TTT('pcalm_prox', fixed=pca_basis), 'pca_pcalm_prox_nolam': lambda: TTT('pcalm_prox_nolam', fixed=pca_basis),
             'pca_pcalm_robust': lambda: TTT('pcalm_robust', fixed=pca_basis), 'ttt_pcalm_robust': lambda: TTT('pcalm_robust'),
             'ttt_gd_full': lambda: TTT('gd_full'), 'ttt_gd_official': lambda: TTT('gd_official'), 'ttt_ridge': lambda: TTT('ridge'),
             'ttt_quad_ridge': lambda: TTT('quad'), 'transformer': lambda: Transformer()}[name]().to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    curve = []
    if name != 'pca_pcalm_reference':
        opt = torch.optim.Adam(params, lr=args.lr); gen = torch.Generator(device=dev).manual_seed(args.seed + 100)
        for step in range(args.steps):
            a, y, aq, yq = sample(args.batch, n_train, 64, Xtr, gen)
            loss = ((model(a, y, aq, w_init(args.batch, gen)) - yq) ** 2).mean() / yq.var().clamp_min(1e-8)
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 250 == 0 or step == args.steps - 1:
                curve.append(dict(step=step, loss=float(loss))); print(f'{name} step {step} nloss {float(loss):.4f}', flush=True)
    model.eval()
    ev = evaluate(model)
    if args.ens > 1 and isinstance(model, TTT) and model.kind not in ('ridge', 'quad'):
        results.setdefault('ensemble', {})[name] = evaluate(model, args.ens)
        for T in args.ens_temps:
            results.setdefault(f'ensemble_T{T:g}', {})[name] = evaluate(model, args.ens, temp=T)
            if args.val_train:      # temperature chosen on TRAIN-split images only (R6)
                results.setdefault(f'val_train_ensemble_T{T:g}', {})[name] = evaluate(model, args.ens, Xtr, temp=T)
            print(f'   {name} ens={args.ens} T={T:g}: ' + '  '.join(f"n={k}m: {v['nmse_mean']:.4f}" for k, v in results[f'ensemble_T{T:g}'][name].items()), flush=True)
        if args.val_train:
            results.setdefault('val_train_ensemble', {})[name] = evaluate(model, args.ens, Xtr)
            print(f'   {name} VAL(train images) ens={args.ens}: ' + '  '.join(f"n={k}m: {v['nmse_mean']:.3f}" for k, v in results['val_train_ensemble'][name].items()), flush=True)
        print(f'   {name} ens={args.ens}: ' + '  '.join(f"n={k}m: nmse {v['nmse_mean']:.3f}/{v['nmse_median']:.3f}" for k, v in results['ensemble'][name].items()), flush=True)
    learned = {k: float(v.exp()) if k.startswith('log_') else v.tolist() for k, v in model.named_parameters() if v.numel() <= 2}
    results[name] = dict(train_seconds=time.perf_counter() - t0, curve=curve, eval_test_images=ev, learned_scalars=learned)
    print(f'== {name}: ' + '  '.join(f"n={k}m: nmse {v['nmse_mean']:.3f}/{v['nmse_median']:.3f} succ {v['success']:.2f}" for k, v in ev.items())
          + f"  ({results[name]['train_seconds']:.0f}s)", flush=True)
    (out / 'results.json').write_text(json.dumps(results, indent=1))
print('DONE')
