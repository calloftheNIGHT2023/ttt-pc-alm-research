"""Method-independent bounds for a base with noise ratio nu (paper Thm B1).
Pair task: y = |m + e|, m ~ N(0, 2(1-nu)) (decodable part), e ~ N(0, 2 nu) (undecodable residual of the base).
floor(nu) = E Var(y|m) / Var(y): lower bound on the NMSE of EVERY learner that sees the base (even with the true direction).
I(nu)     = I(y; m) in nats: per-sample information the |.|-channel carries about the task direction (Fano bound)."""
import json, sys
import numpy as np
from math import erf
rng = np.random.default_rng(0)
erfv = np.vectorize(erf)
def fold(m, s): return s*np.sqrt(2/np.pi)*np.exp(-m**2/(2*s**2)) + m*erfv(m/(s*np.sqrt(2)))
out = []
for nu in [0.0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9]:
    N = 400000
    m = rng.standard_normal(N) * np.sqrt(2 * (1 - nu)); s = np.sqrt(2 * nu)
    if nu == 0:
        out.append(dict(nu=0.0, floor=0.0, info_nats=float('inf'))); continue
    y = np.abs(m + s * rng.standard_normal(N))
    floor = float(np.mean(m**2 + s**2 - fold(m, s)**2) / np.var(y))
    # h(y): y = |z|, z ~ N(0, 2)  =>  h(y) = 0.5 log(2 pi e 2) - log 2
    hy = 0.5*np.log(2*np.pi*np.e*2) - np.log(2)
    phi = lambda x: np.exp(-x**2/(2*s**2))/(np.sqrt(2*np.pi)*s)
    hyx = float(-np.mean(np.log(phi(y - m) + phi(y + m))))
    out.append(dict(nu=nu, floor=floor, info_nats=hy - hyx))
    print(f"nu={nu:.2f}  floor={floor:.3f}  I(y;m)={hy-hyx:.4f} nats   Gaussian-channel I={0.5*np.log(1+(1-nu)/nu):.4f}")
json.dump(out, open(sys.argv[1], 'w'), indent=1)
