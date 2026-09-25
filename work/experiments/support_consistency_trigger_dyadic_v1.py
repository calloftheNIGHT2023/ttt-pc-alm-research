"""335 exact dyadic support computation; loss remains a Fraction at the API."""
from fractions import Fraction
import math


def scaled(b, x, v=()):
    ratios = []
    for value in [*b, *x, *v, .001, .5]:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError('Only finite binary64 inputs are supported')
        ratios.append(value.as_integer_ratio())
    scale = max(d for _, d in ratios)
    q = [num*(scale//den) for num, den in ratios]
    d, n = len(b), len(x)
    return scale, q[:d], q[d:d+n], q[d+n:-2], q[-2]


def exact_forward(b, x):
    scale, bb, h, _, _ = scaled(b, x); patterns = []
    for bias in bb:
        z = [a+bias for a in h]
        patterns.append([int(a >= 0)+int(a >= scale//2)+int(a >= scale) for a in z])
        h = [max(0, scale-abs(2*a-scale)) for a in z]
    return [Fraction(a, scale) for a in h], patterns


def support_loss(b, x, v):
    assert len(x) == len(v) and len(x) > 0
    scale, bb, h, vv, eps = scaled(b, x, v)
    for bias in bb:
        h = [max(0, scale-abs(2*(a+bias)-scale)) for a in h]
    residual = [max(0, abs(a-y)-eps) for a, y in zip(h, vv)]
    return Fraction(sum(r*r for r in residual), 2*len(residual)*scale*scale)
