"""Exact scalar activity coordinate in a piecewise-linear local constraint.

Minimize (s-center)^2 + trust*(s-old)^2 + sum_k(g(a_k*s+c_k)-target_k)^2.
Only adjacent-layer coefficients are used. No whole-network differentiation.
"""
import numpy as np
import vector_interval_memory as base
SLOPES = np.array([0., 2., -2., 0.])
INTERCEPTS = np.array([-1., 1., 1., -1.])


def interval_bounds(x, weights, bound):
    lower = upper = np.array(x, copy=True)
    lows, highs = [], []
    for weight in weights:
        positive, negative = np.maximum(weight, 0.), np.minimum(weight, 0.)
        zlo = lower @ positive.T + upper @ negative.T - bound
        zhi = upper @ positive.T + lower @ negative.T + bound
        left, right = base.family.activation(zlo), base.family.activation(zhi)
        lower = np.minimum(left, right)
        upper = np.where((zlo <= 0) & (zhi >= 0), 1., np.maximum(left, right))
        lows.append(lower); highs.append(upper)
    return np.array(lows), np.array(highs)


def energy(s, center, target, offset, coefficient, old, trust):
    return ((s-center)**2 + trust*(s-old)**2 +
            np.sum((base.family.activation(s[:, None]*coefficient+offset)-target)**2, axis=1))


def solve(center, target, offset, coefficient, old, trust=.01, lower=None, upper=None):
    center, old = np.asarray(center), np.asarray(old)
    a = np.broadcast_to(coefficient, offset.shape)
    lo = np.full_like(center, -np.inf) if lower is None else np.broadcast_to(lower, center.shape).copy()
    hi = np.full_like(center, np.inf) if upper is None else np.broadcast_to(upper, center.shape).copy()
    assert np.all(lo <= hi)
    feasible_old = np.clip(old, lo, hi)
    old_energy = energy(feasible_old, center, target, offset, a, old, trust)
    # A feasible incumbent bounds the identity quadratic: every global minimizer
    # is within center +/- sqrt(E_incumbent), even on an unbounded domain.
    radius = np.sqrt(np.maximum(old_energy, 0.))
    lo, hi = np.maximum(lo, center-radius), np.minimum(hi, center+radius)
    # Roundoff at a zero-width intersection must not invert the interval.
    assert np.max(lo-hi) <= 1e-10
    hi = np.maximum(lo, hi)
    with np.errstate(divide='ignore', invalid='ignore'):
        roots = (np.array([-1., 0., 1.])[None,None,:]-offset[:,:,None])/a[:,:,None]
    crossings = np.sum(roots <= lo[:,None,None], axis=2)
    branch = np.where(a > 0, crossings, 3-crossings)
    branch = np.where(a == 0, np.searchsorted([-1.,0.,1.], offset, side='left'), branch)
    slope, intercept = SLOPES[branch], INTERCEPTS[branch]
    linear, constant = slope*a, slope*offset+intercept
    aa = 1+trust+np.sum(linear**2,axis=1)
    bb = -center-trust*old+np.sum(linear*(constant-target),axis=1)
    cc = center**2+trust*old**2+np.sum((constant-target)**2,axis=1)
    k = np.arange(3)[None,None,:]
    before = np.where(a[:,:,None] > 0,k,k+1)
    after = np.where(a[:,:,None] > 0,k+1,k)
    lm, lp = SLOPES[before]*a[:,:,None], SLOPES[after]*a[:,:,None]
    cm = SLOPES[before]*offset[:,:,None]+INTERCEPTS[before]-target[:,:,None]
    cp = SLOPES[after]*offset[:,:,None]+INTERCEPTS[after]-target[:,:,None]
    inside = np.isfinite(roots)&(roots > lo[:,None,None])&(roots < hi[:,None,None])&(a[:,:,None] != 0)
    cuts = np.where(inside,roots,hi[:,None,None]).reshape(len(center),-1)
    delta_a = np.where(inside,lp**2-lm**2,0.).reshape(len(center),-1)
    delta_b = np.where(inside,lp*cp-lm*cm,0.).reshape(len(center),-1)
    delta_c = np.where(inside,cp**2-cm**2,0.).reshape(len(center),-1)
    order = np.argsort(cuts,axis=1,kind='stable')
    cuts = np.take_along_axis(cuts,order,axis=1)
    def coefficients(initial, delta):
        return initial[:,None]+np.c_[np.zeros(len(center)),np.cumsum(np.take_along_axis(delta,order,axis=1),axis=1)]
    aa,bb,cc = coefficients(aa,delta_a),coefficients(bb,delta_b),coefficients(cc,delta_c)
    assert np.min(aa) > .99
    left,right = np.c_[lo,cuts],np.c_[cuts,hi]
    points = np.clip(-bb/aa,left,right)
    values = aa*points**2+2*bb*points+cc
    index = np.argmin(values,axis=1); answer = points[np.arange(len(center)),index]
    actual = energy(answer,center,target,offset,a,old,trust)
    answer = np.where(actual <= old_energy+1e-12,answer,feasible_old)
    meta = dict(coordinate_candidates=points.shape[1], working_array_bytes_subtotal=sum(t.nbytes for t in
        [roots,cuts,delta_a,delta_b,delta_c,order,aa,bb,cc,left,right,points,values]))
    return answer, meta
