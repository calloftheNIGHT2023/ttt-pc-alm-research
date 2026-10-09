"""Exp R (report 449): end-to-end trained TTT layers on REAL multi-attribute data (NLP words / QM9 molecules).

Task (per sequence): pick a real attribute j, sample n context pairs (a, b) and 64 query pairs of items that have j;
input o = E[a] - E[b] (PCA-whitened real embeddings), label y = |A[a, j] - A[b, j]| (real annotations, standardised).
"How different are a and b in property j?" -- a multi-branch (sign-lost) relation along a task-specific direction.
Outer training uses ONLY train attributes; evaluation uses held-out attributes never seen by the encoder or outer loop.
Models share the key projection theta_K (d x D):
  ttt_pcalm        inner memory |w.k|, K lifted iterations (closed-form 2-branch inversion, multipliers, local LS), rho=1
  ttt_gd_full      same memory, K full-batch GD steps (learned step)
  ttt_gd_official  same memory, one pass in chunks of 16, one GD step per chunk (learned step)
  ttt_ridge        closed-form linear memory with intercept
  ttt_quad_ridge   closed-form ridge on quadratic features vec(k k^T) predicting y^2, output sqrt (exact function class)
  transformer      4-layer encoder on [o, y] tokens
  probe_reference  NOT few-shot: linear probe of attribute j trained on ALL labelled items, |probe(a)-probe(b)|
--v3 (report 450, noise gap): real labels are y = |u.k + eps| (undecodable residual INSIDE the |.|). Adds
  ttt_pcalm_sigma  noise-calibrated PC-ALM: Gibbs posterior over the two local branches at temperature c*r2
                   (r2 = per-sequence residual variance, so iterations anneal from spectral-like to exact inversion),
                   leaky multiplier lam <- (lam + rho r)/(1 + rho g r2) (= ALM with a Gaussian slack of variance g*r2);
                   c, g learned by the outer loop; c, g -> 0 recovers ttt_pcalm exactly.
  ttt_gd_soft      same noise-calibrated local target, but a gradient step instead of lifted LS + multiplier
  and, for every |w.k| memory, the folded-normal readout E|N(w.k_q, sr*r2)| (a, b calibrated) instead of |w.k_q|.
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
ap.add_argument('--data', required=True)
ap.add_argument('--train_attrs', nargs='*', default=None)
ap.add_argument('--v2', action='store_true', help='post-hoc (report 449): learned LS ridge for PC-ALM + learned output calibration a|w.k|+b for all |.| models')
ap.add_argument('--test_attrs', nargs='*', default=None)
ap.add_argument('--df_correct', action='store_true', help='report 450: unbiased noise estimate RSS/(n - df), df = tr((G+mu I)^-1 G)')
ap.add_argument('--prox_noise', action='store_true', help='report 450: proximal weight also scales with r2/E[y^2], so it vanishes in the exact regime')
ap.add_argument('--snr_rule', default='none', choices=['none', 'leak', 'both'],
                help='report 450: scale the slack variance (leak) and optionally the branch temperature by the in-context '
                     'noise-to-signal ratio nu/(1-nu), nu = r2/E[y^2] (df-corrected): exact regime -> hard ALM, noisy -> relaxed')
ap.add_argument('--g_init', type=float, default=1.0, help='initial multiplier leak (slack variance / residual variance); large = near no multiplier')
ap.add_argument('--v3', action='store_true', help='report 450: noise-calibrated models + folded-normal readout (implies --v2)')
ap.add_argument('--D', type=int, default=64)
ap.add_argument('--d', type=int, default=16)
ap.add_argument('--n_over_d', type=float, default=4)
ap.add_argument('--steps', type=int, default=3000)
ap.add_argument('--batch', type=int, default=64)
ap.add_argument('--K', type=int, default=30)
ap.add_argument('--lr', type=float, default=3e-3)
ap.add_argument('--select_pair', action='store_true', help='chain closure: per-sequence member selection between ttt_pcalm_prox and '
                'ttt_gd_soft by 4-fold cross-validation inside the context (deployment data only, R6)')
ap.add_argument('--models', nargs='+', default=['ttt_pcalm', 'ttt_gd_full', 'ttt_gd_official', 'ttt_ridge', 'ttt_quad_ridge', 'transformer', 'probe_reference'])
ap.add_argument('--eval_n_over_d', type=float, nargs='+', default=[2, 4, 8])
ap.add_argument('--eval_seqs', type=int, default=1024)
ap.add_argument('--seed', type=int, default=449001)
ap.add_argument('--device', default='cuda:0')
ap.add_argument('--out', required=True)
args = ap.parse_args()
args.v2 = args.v2 or args.v3
out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
(out / 'args.json').write_text(json.dumps(vars(args), indent=1))
dev, dt = args.device, torch.float32
d, D = args.d, args.D

Z = np.load(args.data)
E0, A0 = Z['E'], Z['A']; names = [str(x) for x in Z['names']]
train_attr, test_attr = [int(i) for i in Z['train_idx']], [int(i) for i in Z['test_idx']]
if args.test_attrs:                 # split chosen from decodability/correlation structure only (report 449)
    test_attr = [names.index(x) for x in args.test_attrs]
    train_attr = [names.index(x) for x in args.train_attrs]
mu = E0.mean(0); ev, evec = np.linalg.eigh(np.cov(E0 - mu, rowvar=False)); o_ = np.argsort(ev)[::-1][:D]
Ew = (E0 - mu) @ (evec[:, o_] / np.sqrt(ev[o_]))            # PCA-whitened real embeddings (no labels used)
E = torch.tensor(Ew, dtype=dt, device=dev)
A = torch.tensor(np.nan_to_num(A0, nan=0.0), dtype=dt, device=dev)
avail = {j: torch.tensor(np.flatnonzero(np.isfinite(A0[:, j])), device=dev) for j in range(A0.shape[1])}


def soft_activity(y, a, rho, tau=None):
    """Local 2-branch problem min_s (|s|-y)^2 + rho/2 (s-a)^2. tau=None: exact argmin; else Gibbs posterior mean over
    the two branch minimisers at temperature tau (noise-calibrated inversion; tau -> 0 is the exact argmin)."""
    sp = ((2 * y + rho * a) / (2 + rho)).clamp_min(0); sm = ((-2 * y + rho * a) / (2 + rho)).clamp_max(0)
    Fv = lambda s_: (s_.abs() - y) ** 2 + 0.5 * rho * (s_ - a) ** 2
    if tau is None:
        return torch.where(Fv(sp) <= Fv(sm), sp, sm)
    wp = torch.sigmoid((Fv(sm) - Fv(sp)) / tau)
    return wp * sp + (1 - wp) * sm


def foldmean(mu_, sd):
    """E|N(mu, sd^2)| (folded-normal mean): the Bayes readout when the residual sits inside the |.|."""
    return sd * math.sqrt(2 / math.pi) * torch.exp(-mu_ ** 2 / (2 * sd ** 2)) + mu_ * torch.erf(mu_ / (sd * math.sqrt(2)))


def sample(B, n, nq, attrs, gen):
    js = torch.tensor(attrs, device=dev)[torch.randint(len(attrs), (B,), generator=gen, device=dev)]
    O = torch.empty(B, n + nq, D, device=dev); Y = torch.empty(B, n + nq, device=dev)
    for j in js.unique().tolist():
        rows = (js == j).nonzero().squeeze(-1); pool = avail[j]
        ia = pool[torch.randint(len(pool), (len(rows), n + nq), generator=gen, device=dev)]
        ib = pool[torch.randint(len(pool), (len(rows), n + nq), generator=gen, device=dev)]
        O[rows] = E[ia] - E[ib]; Y[rows] = (A[ia, j] - A[ib, j]).abs()
    return O[:, :n], Y[:, :n], O[:, n:], Y[:, n:], js


class TTT(nn.Module):
    def __init__(self, kind, K_iter=None):
        super().__init__()
        self.K_iter = K_iter or args.K
        self.kind = kind
        th = torch.randn(d, D, generator=torch.Generator().manual_seed(args.seed + 1)); th = th / th.norm(dim=-1, keepdim=True)
        self.theta = nn.Parameter(th.to(dev))
        self.log_step = nn.Parameter(torch.tensor(math.log(0.05), device=dev))
        self.log_mu = nn.Parameter(torch.tensor(math.log(1e-1), device=dev))
        self.log_mu_pc = nn.Parameter(torch.tensor(math.log(1.0 if args.v2 else 1e-2), device=dev), requires_grad=args.v2)
        self.cal = nn.Parameter(torch.tensor([1.0, 0.0], device=dev), requires_grad=args.v2)
        self.log_c = nn.Parameter(torch.tensor(0.0, device=dev))       # branch temperature / residual variance
        self.log_g = nn.Parameter(torch.tensor(math.log(args.g_init), device=dev))       # multiplier leak (slack variance) / residual variance
        self.log_sr = nn.Parameter(torch.tensor(0.0, device=dev))
        self.log_prox = nn.Parameter(torch.tensor(0.0, device=dev))    # proximal weight / mean eigenvalue of K^T K
        self.log_kappa = nn.Parameter(torch.tensor(math.log(2.0), device=dev))  # Huber threshold (residual std units)      # readout noise variance / residual variance

    def forward(self, o, y, oq, w0):
        K = o @ self.theta.T; Kq = oq @ self.theta.T
        B, n, _ = K.shape
        I = lambda m: torch.eye(m, device=dev)
        if self.kind == 'ridge':
            K1 = torch.cat([K, torch.ones(B, n, 1, device=dev)], -1); Kq1 = torch.cat([Kq, torch.ones(B, Kq.shape[1], 1, device=dev)], -1)
            w = torch.linalg.solve(K1.transpose(1, 2) @ K1 + self.log_mu.exp() * I(d + 1), K1.transpose(1, 2) @ y.unsqueeze(-1)).squeeze(-1)
            return torch.einsum('bqd,bd->bq', Kq1, w)
        if self.kind == 'quad':
            iu = torch.triu_indices(d, d, device=dev)
            feat = lambda M: torch.cat([(M.unsqueeze(-1) * M.unsqueeze(-2))[..., iu[0], iu[1]], torch.ones(*M.shape[:2], 1, device=dev)], -1)
            F, Fq = feat(K.double()), feat(Kq.double())
            sc = F.pow(2).sum(-1).mean(-1).clamp_min(1e-12).sqrt()[:, None, None]    # per-sequence feature scale
            F, Fq = F / sc, Fq / sc
            # dual form (n x n) since p > n is typical; float64 (v2: float32 overflowed on QM9, report 449)
            G = F @ F.transpose(1, 2) + self.log_mu.exp().double() * torch.eye(n, device=dev, dtype=torch.float64)
            alpha = torch.linalg.solve(G, (y.double() ** 2).unsqueeze(-1)).squeeze(-1)
            return torch.einsum('bqp,bnp,bn->bq', Fq, F, alpha).clamp_min(0).sqrt().float()
        w = w0
        if self.kind == 'gd_official':
            eta = self.log_step.exp()
            for c in range(0, n, 16):
                Kc, yc = K[:, c:c + 16], y[:, c:c + 16]
                s = torch.einsum('bnd,bd->bn', Kc, w)
                w = (w - eta * 2 / Kc.shape[1] * torch.einsum('bn,bnd->bd', (s.abs() - yc) * torch.sign(s), Kc)).clamp(-50, 50)
        elif self.kind == 'gd_full':
            eta = self.log_step.exp()
            for _ in range(self.K_iter):
                s = torch.einsum('bnd,bd->bn', K, w)
                w = (w - eta * 2 / n * torch.einsum('bn,bnd->bd', (s.abs() - y) * torch.sign(s), K)).clamp(-50, 50)
        elif self.kind == 'pcalm':
            rho = 1.0
            L = torch.linalg.cholesky(K.transpose(1, 2) @ K + self.log_mu_pc.exp() * I(d))
            lam = torch.zeros(B, n, device=dev)
            for _ in range(self.K_iter):
                pred = torch.einsum('bnd,bd->bn', K, w); a = pred - lam / rho
                sp = ((2 * y + rho * a) / (2 + rho)).clamp_min(0); sm = ((-2 * y + rho * a) / (2 + rho)).clamp_max(0)
                Fv = lambda s_: (s_.abs() - y) ** 2 + 0.5 * rho * (s_ - a) ** 2
                s = torch.where(Fv(sp) <= Fv(sm), sp, sm)                         # exact 2-branch local inversion
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
            L = torch.linalg.cholesky(G + (self.log_mu_pc.exp() + p)[:, :, None] * I(d))
            lam = torch.zeros(B, n, device=dev)
            infl = torch.ones(B, 1, device=dev)
            if args.df_correct:                      # in-context residuals shrink as the lifted fit absorbs noise: correct by df
                with torch.no_grad():
                    Lmu = torch.linalg.cholesky(G + self.log_mu_pc.exp() * I(d))
                    df = d - self.log_mu_pc.exp() * torch.cholesky_inverse(Lmu).diagonal(dim1=1, dim2=2).sum(-1)
                    infl = (n / (n - df).clamp_min(1.0))[:, None]
            for _ in range(self.K_iter):
                pred = torch.einsum('bnd,bd->bn', K, w)
                r2 = infl * ((y - pred.abs()) ** 2).mean(-1, keepdim=True).detach()
                nsr = torch.ones_like(r2)
                if args.snr_rule != 'none':
                    nu = (r2 / (y ** 2).mean(-1, keepdim=True).clamp_min(1e-8)).clamp(0, 0.95)
                    nsr = nu / (1 - nu)                              # in-context noise-to-signal ratio
                tau = self.log_c.exp() * r2 * (nsr if args.snr_rule == 'both' else 1) + 1e-4
                s = soft_activity(y, pred - lam / rho, rho, tau)
                if self.kind != 'pcalm_prox_nolam':
                    lam = (lam + rho * (s - pred)) / (1 + rho * self.log_g.exp() * r2 * nsr)
                if self.kind == 'pcalm_robust':
                    bound = self.log_kappa.exp() / (self.log_g.exp() * r2.sqrt() + 1e-6)
                    lam = torch.maximum(torch.minimum(lam, bound), -bound)
                if args.prox_noise and self.kind != 'pcalm_sigma':      # every noise term vanishes with the residual
                    p = p0 * (r2 / (y ** 2).mean(-1, keepdim=True).clamp_min(1e-8)).clamp_max(1.0)
                    L = torch.linalg.cholesky(G + (self.log_mu_pc.exp() + p)[:, :, None] * I(d))
                w = torch.cholesky_solve((torch.einsum('bnd,bn->bd', K, s + lam / rho) + p * w).unsqueeze(-1), L).squeeze(-1)
        elif self.kind == 'gd_soft':
            eta = self.log_step.exp()
            for _ in range(self.K_iter):
                pred = torch.einsum('bnd,bd->bn', K, w)
                r2 = ((y - pred.abs()) ** 2).mean(-1, keepdim=True).detach()
                s = soft_activity(y, pred, 1.0, self.log_c.exp() * r2 + 1e-4)
                w = (w - eta * 2 / n * torch.einsum('bn,bnd->bd', pred - s, K)).clamp(-50, 50)
        mq = torch.einsum('bqd,bd->bq', Kq, w)
        if args.v3:
            r2 = ((y - torch.einsum('bnd,bd->bn', K, w).abs()) ** 2).mean(-1, keepdim=True)
            return self.cal[0] * foldmean(mq, (self.log_sr.exp() * r2 + 1e-4).sqrt()) + self.cal[1]
        return self.cal[0] * mq.abs() + self.cal[1]


class Transformer(nn.Module):
    def __init__(self, dm=128, layers=4, heads=4):
        super().__init__()
        self.inp = nn.Linear(D + 2, dm)
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(dm, heads, 4 * dm, dropout=0.0, batch_first=True, norm_first=True), layers)
        self.head = nn.Linear(dm, 1)

    def forward(self, o, y, oq, w0):
        B, n, _ = o.shape; nq = oq.shape[1]
        ctx = torch.cat([o, y.unsqueeze(-1), torch.zeros_like(y).unsqueeze(-1)], -1)
        qry = torch.cat([oq, torch.zeros(B, nq, 1, device=dev), torch.ones(B, nq, 1, device=dev)], -1)
        h = self.inp(torch.cat([ctx, qry], 1)); L = n + nq
        mask = torch.zeros(L, L, dtype=torch.bool, device=dev); mask[:, n:] = True
        mask[torch.arange(n, L), torch.arange(n, L)] = False
        return self.head(self.enc(h, mask=mask)[:, n:]).squeeze(-1)


def w_init(B, gen):
    return 0.1 * torch.randn(B, d, generator=gen, device=dev) / math.sqrt(d)


def evaluate(fn, attrs_list, label):
    res = {}
    egen = torch.Generator(device=dev).manual_seed(args.seed + 999)
    for nd in args.eval_n_over_d:
        n = int(nd * d); per = {}
        for j in attrs_list:
            errs = []
            for _ in range(max(1, args.eval_seqs // (256 * len(attrs_list)))):
                o, y, oq, yq, _ = sample(256, n, 64, [j], egen)
                p = fn(o, y, oq, w_init(256, egen), j)
                errs.append(((p - yq) ** 2).mean(-1) / yq.var(-1).clamp_min(1e-6))
            e = torch.nan_to_num(torch.cat(errs), nan=1e3).clamp_max(1e3)
            per[names[j]] = dict(nmse_mean=float(e.mean()), nmse_median=float(e.median()), per_seq=[round(float(v), 5) for v in e])
        res[str(float(nd))] = dict(per_attribute=per, pooled_nmse_mean=float(np.mean([v['nmse_mean'] for v in per.values()])),
                            pooled_nmse_median=float(np.mean([v['nmse_median'] for v in per.values()])))
    return res


n_train = int(args.n_over_d * d)
results = {}
trained = {}
for name in args.models:
    t0 = time.perf_counter()
    if name == 'probe_reference':
        probes = {}
        for j in train_attr + test_attr:
            idx = avail[j]; X = E[idx]; t = A[idx, j]
            probes[j] = torch.linalg.solve(X.T @ X + 1e-2 * torch.eye(D, device=dev), X.T @ t)
        fn = lambda o, y, oq, w0, j: (oq @ probes[j]).abs()
        results[name] = dict(train_seconds=time.perf_counter() - t0, eval_test=evaluate(fn, test_attr, 'test'),
                             eval_train=evaluate(fn, train_attr, 'train'))
    else:
        torch.manual_seed(args.seed + 11)
        model = {'ttt_pcalm': lambda: TTT('pcalm'), 'ttt_pcalm_sigma': lambda: TTT('pcalm_sigma'), 'ttt_gd_soft': lambda: TTT('gd_soft'),
                 'ttt_pcalm_prox': lambda: TTT('pcalm_prox'), 'ttt_pcalm_prox_nolam': lambda: TTT('pcalm_prox_nolam'),
                 'ttt_pcalm_robust': lambda: TTT('pcalm_robust'), 'ttt_gd_full': lambda: TTT('gd_full'), 'ttt_gd_official': lambda: TTT('gd_official'),
                 'ttt_gd_soft_k45': lambda: TTT('gd_soft', 45), 'ttt_gd_full_k45': lambda: TTT('gd_full', 45),
                 'ttt_ridge': lambda: TTT('ridge'), 'ttt_quad_ridge': lambda: TTT('quad'), 'transformer': lambda: Transformer()}[name]().to(dev)
        params = list(model.parameters()); opt = torch.optim.Adam(params, lr=args.lr)
        gen = torch.Generator(device=dev).manual_seed(args.seed + 100); curve = []
        for step in range(args.steps):
            o, y, oq, yq, _ = sample(args.batch, n_train, 64, train_attr, gen)
            loss = ((model(o, y, oq, w_init(args.batch, gen)) - yq) ** 2).mean()
            opt.zero_grad()
            if torch.isfinite(loss):
                loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % 250 == 0 or step == args.steps - 1:
                curve.append(dict(step=step, loss=float(loss))); print(f'{name} step {step} loss {float(loss):.4f}', flush=True)
        model.eval(); trained[name] = model
        fn = lambda o, y, oq, w0, j: model(o, y, oq, w0)
        with torch.no_grad():
            learned = {k: float(v.exp()) if k.startswith('log_') else v.tolist() for k, v in model.named_parameters() if v.numel() <= 2}
            results[name] = dict(train_seconds=time.perf_counter() - t0, curve=curve, learned_scalars=learned,
                                 eval_test=evaluate(fn, test_attr, 'test'), eval_train=evaluate(fn, train_attr, 'train'))
    r = results[name]
    print(f"== {name}: held-out attrs pooled NMSE " + '  '.join(f"n={k}d {v['pooled_nmse_mean']:.3f}/{v['pooled_nmse_median']:.3f}" for k, v in r['eval_test'].items())
          + f"  | train attrs n=4d {r['eval_train']['4.0']['pooled_nmse_mean']:.3f}  ({r['train_seconds']:.0f}s)", flush=True)
    (out / 'results.json').write_text(json.dumps(dict(names=names, train=[names[i] for i in train_attr], test=[names[i] for i in test_attr], results=results), indent=1))
if args.select_pair:
    MA, MB = trained['ttt_pcalm_prox'], trained['ttt_gd_soft']
    stats = dict(chose_pcalm=[], n=[])

    def cv_error(model, o, y, w0, F=4):
        n = y.shape[1]; err = torch.zeros(y.shape[0], device=dev)
        for f in range(F):
            lo, hi = f * n // F, (f + 1) * n // F
            tr = torch.cat([torch.arange(0, lo, device=dev), torch.arange(hi, n, device=dev)])
            p = model(o[:, tr], y[:, tr], o[:, lo:hi], w0)
            err = err + torch.nan_to_num(((p - y[:, lo:hi]) ** 2).mean(-1), nan=1e6).clamp_max(1e6)
        return err

    def select_fn(o, y, oq, w0, j):
        ea, eb = cv_error(MA, o, y, w0), cv_error(MB, o, y, w0)
        pick = (ea <= eb)[:, None]
        stats['chose_pcalm'].append(float(pick.float().mean())); stats['n'].append(int(y.shape[1]))
        return torch.where(pick, MA(o, y, oq, w0), MB(o, y, oq, w0))

    with torch.no_grad():
        t0 = time.perf_counter()
        results['select_cv'] = dict(eval_test=evaluate(select_fn, test_attr, 'test'), eval_train=evaluate(select_fn, train_attr, 'train'),
                                    seconds=time.perf_counter() - t0)
        ns = sorted(set(stats['n']))
        results['select_cv']['fraction_choosing_pcalm'] = {str(n / d): float(np.mean([c for c, m in zip(stats['chose_pcalm'], stats['n']) if m == n])) for n in ns}
    r = results['select_cv']
    print("== select_cv: held-out attrs pooled NMSE " + '  '.join(f"n={k}d {v['pooled_nmse_mean']:.3f}/{v['pooled_nmse_median']:.3f}" for k, v in r['eval_test'].items())
          + f"  | chose PC-ALM {r['fraction_choosing_pcalm']}", flush=True)
    (out / 'results.json').write_text(json.dumps(dict(names=names, train=[names[i] for i in train_attr], test=[names[i] for i in test_attr], results=results), indent=1))
print('DONE')
