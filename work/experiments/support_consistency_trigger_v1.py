"""324 exact public-support first-fit selection; no query/geometry inputs."""
from fractions import Fraction as F
import numpy as np

EPS = F(.001)


def exact_forward(b, x):
    h = [F(float(v)) for v in x]
    patterns = []
    for bias in b:
        z = [v + F(float(bias)) for v in h]
        patterns.append([int(v >= 0) + int(v >= F(1, 2)) + int(v >= 1) for v in z])
        h = [max(F(0), 1 - abs(2*v - 1)) for v in z]
    return h, patterns


def support_loss(b, x, v):
    pred, _ = exact_forward(b, x)
    rr = [max(F(0), abs(a-F(float(y))) - EPS) for a, y in zip(pred, v)]
    return sum((r*r for r in rr), F(0)) / (2*len(rr))


def choose_first_fit(bs, x, v):
    best = None
    for i, b in enumerate(bs):
        value = support_loss(b, x, v)
        if best is None or value < best[0]:
            best = (value, i)
        if value == 0:
            return dict(index=i, loss='0', support_fit=True, fallback=False, exact_forward_calls=i+1)
    assert best is not None
    return dict(index=best[1], loss=str(best[0]), support_fit=False,
                fallback=True, exact_forward_calls=len(bs))


def locations():
    return [(0, t, r) for t in range(1, 33) for r in range(33)] + [(1, t, 0) for t in range(1, 33)]


def uniform_index(seed):
    return int(np.random.default_rng(np.random.SeedSequence([324071, int(seed)])).integers(1088))
