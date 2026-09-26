"""Paper Sec. 6: diagnostics of a pretrained base for inference-time self-iteration.

For each base (an embedding npz produced by prep_real_attr.py) and each attribute j:
  R2_j   linear decodability of attribute j from the base, with the same PCA-whitened D=64 input the TTT layer sees
         (ridge probe fitted on 80% of items, R2 on the other 20%);
  nu_j   = 1 - R2_j, the undecodable fraction = noise-to-total ratio of the base residual inside |.| (condition B2);
  floor  the Bayes floor of Prop. B1 for the pair task |A_j(a) - A_j(b)| under the Gaussian residual model,
         E Var(y|k) / Var(y) with y = |m + e|, m ~ N(0, 2 R2), e ~ N(0, 2 nu)  (Monte Carlo).
"""
import argparse
import json
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--bases', nargs='+', required=True, help='name=path.npz')
ap.add_argument('--D', type=int, default=64)
ap.add_argument('--out', required=True)
a = ap.parse_args()
rng = np.random.default_rng(0)


def fold(m, s):
    from math import erf, sqrt, pi
    s = np.maximum(s, 1e-12)
    return s * np.sqrt(2 / np.pi) * np.exp(-m ** 2 / (2 * s ** 2)) + m * np.vectorize(erf)(m / (s * np.sqrt(2)))


res = {}
for spec in a.bases:
    name, path = spec.split('=')
    Z = np.load(path); E0, A0 = Z['E'], Z['A']; names = [str(x) for x in Z['names']]
    mu = E0.mean(0); ev, evec = np.linalg.eigh(np.cov(E0 - mu, rowvar=False)); o = np.argsort(ev)[::-1][:a.D]
    E = (E0 - mu) @ (evec[:, o] / np.sqrt(ev[o]))
    out = {}
    for j, nm in enumerate(names):
        idx = np.flatnonzero(np.isfinite(A0[:, j])); rng.shuffle(idx)
        cut = int(0.8 * len(idx)); tr, te = idx[:cut], idx[cut:]
        X, t = E[tr], A0[tr, j]
        w = np.linalg.solve(X.T @ X + 1e-2 * np.eye(a.D), X.T @ (t - t.mean()))
        p = E[te] @ w + t.mean(); r2 = 1 - np.mean((A0[te, j] - p) ** 2) / np.var(A0[te, j])
        r2 = float(np.clip(r2, 0, 1)); nu = 1 - r2
        m = rng.standard_normal(200000) * np.sqrt(2 * r2); s = np.sqrt(2 * nu)
        y = np.abs(m + s * rng.standard_normal(m.shape))
        fl = float(np.mean(m ** 2 + s ** 2 - fold(m, s) ** 2) / np.var(y)) if nu > 0 else 0.0
        out[nm] = dict(R2=r2, nu=nu, bayes_floor_nmse=fl, n_items=int(len(idx)))
    res[name] = out
    print(name, {k: round(v['R2'], 3) for k, v in out.items()}, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(res, indent=1))
print('DONE')
