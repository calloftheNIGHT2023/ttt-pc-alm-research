"""355 free-credit halfspace projections; exact certificates alone reject.

Established convex projection, not a claim of a new ALM optimizer. The only
PC-specific hypothesis is whether an existing ALM credit is a useful warm start.
"""
import time
import numpy as np
from branch_image_chain_dyadic_v1 import IntegerProblem

S = np.array([0., 2., -2., 0.])
C = np.array([0., 0., 2., 0.])
PREFIXES = (1, 4, 16, 64, 128)
DELTA = 1.


def boxes(v, regs):
    zl = np.array([-.12, 0., .5, 1.])[regs]
    zh = np.array([0., .5, 1., np.nextafter(1.12, np.inf)])[regs]
    hl = np.zeros(regs.shape)
    hh = np.isin(regs, [1, 2]).astype(float)
    hl[:, -1] = np.maximum(0., np.nextafter(v-.001, -np.inf))
    hh[:, -1] = np.minimum(hh[:, -1], np.nextafter(v+.001, np.inf))
    return zl, zh, hl, hh


def response(x, v, regs, a):
    """Analytic layer-local minimizer of the branch-image relaxation.

    Float response proposes, not certifies. Shared b is minimized at endpoints
    and coefficient-dependent knots. Tied minimizers use their midpoint.
    """
    count, d, n = regs.shape
    zl, zh, hl, hh = boxes(v, regs)
    sa = S[regs]*a
    b = np.zeros((count, d)); h = np.empty_like(a); z = np.empty_like(a)
    valid = (hl <= hh).all((1, 2))
    h[:, -1] = np.where(a[:, -1] > 0, hl[:, -1],
                       np.where(a[:, -1] < 0, hh[:, -1], (hl[:, -1]+hh[:, -1])/2))
    for j in range(d):
        pl = np.broadcast_to(x, (count, n)) if j == 0 else hl[:, j-1]
        ph = np.broadcast_to(x, pl.shape) if j == 0 else hh[:, j-1]
        previous = np.zeros_like(pl) if j == 0 else a[:, j-1]
        coef = previous-sa[:, j]
        low = np.maximum(-.12, (zl[:, j]-ph).max(1))
        high = np.minimum(.12, (zh[:, j]-pl).min(1))
        valid &= low <= high
        knots = np.where(coef >= 0, zl[:, j]-pl, zh[:, j]-ph)
        options = np.clip(np.column_stack([low, high, knots]), low[:, None], high[:, None])
        lo = np.maximum(pl[:, None], zl[:, j, None]-options[:, :, None])
        hi = np.minimum(ph[:, None], zh[:, j, None]-options[:, :, None])
        values = (coef[:, None]*np.where(coef[:, None] >= 0, lo, hi)
                  -sa[:, j, None]*options[:, :, None]).sum(2)
        ties = values == values.min(1)[:, None]
        b[:, j] = (np.where(ties, options, np.inf).min(1)
                   +np.where(ties, options, -np.inf).max(1))/2
        lo = np.maximum(pl, zl[:, j]-b[:, j, None])
        hi = np.minimum(ph, zh[:, j]-b[:, j, None])
        hp = np.where(coef > 0, lo, np.where(coef < 0, hi, (lo+hi)/2))
        if j:
            h[:, j-1] = hp
        z[:, j] = hp+b[:, j, None]
    residual = h-S[regs]*z-C[regs]
    objective = (a*residual).sum((1, 2))
    named = [zl, zh, hl, hh, sa, b, h, z, valid, residual, objective,
             coef, low, high, knots, options, lo, hi, values, ties, hp]
    return dict(b=b, h=h, z=z, residual=residual, objective=objective, valid=valid,
                named_array_bytes_subtotal=sum(q.nbytes for q in named))


def solve(x, v, regs, credit, steps=128):
    started = time.perf_counter()
    regs = np.asarray(regs, dtype=np.uint8)
    count, d, n = regs.shape
    if credit.shape != (d, n) or not np.isfinite(credit).all() or steps < 1:
        raise ValueError('Finite credit of shape (depth, observations) and positive steps required')
    scale = np.linalg.norm(credit)
    initial = credit/scale if scale > 0 else credit.copy()
    a = np.broadcast_to(initial, regs.shape).copy()
    accepted = np.zeros(count, np.int32); disabled = np.zeros(count, bool)
    proof_credit = np.zeros_like(a); proofs = []; traces = []
    calls = pairs = empty_calls = 0
    oracle_seconds = exact_seconds = update_seconds = 0.; named_peak = 0
    for step in range(1, steps+1):
        ids = np.flatnonzero((accepted == 0) & ~disabled)
        if len(ids):
            tick = time.perf_counter(); aa = a[ids]
            out = response(x, v, regs[ids], aa)
            oracle_seconds += time.perf_counter()-tick; pairs += len(ids)
            values = out['objective']; residual = out['residual']
            threshold = 1e-10*(1+abs(aa).sum((1, 2)))
            check = (values > threshold) | ~out['valid']
            for local in np.flatnonzero(check):
                index = int(ids[local]); tick = time.perf_counter()
                exact = IntegerProblem(x, v, aa[local]); value = exact.fixed(regs[index])
                exact_seconds += time.perf_counter()-tick; calls += 1
                empty_calls += value is None
                if value is None or value > 0:
                    accepted[index] = step; proof_credit[index] = aa[local]
                    proofs.append(dict(index=index, step=step, mode=regs[index].tobytes().hex(),
                                       lower=None if value is None else exact.string(value),
                                       structural_empty=value is None))
            tick = time.perf_counter()
            norm2 = (residual*residual).sum((1, 2))
            usable = out['valid'] & np.isfinite(values) & np.isfinite(norm2) & (norm2 > 0)
            unresolved = accepted[ids] == 0
            disabled[ids[unresolved & ~usable]] = True
            update = usable & unresolved
            selected = ids[update]
            a[selected] += ((np.maximum(DELTA-values[update], 0.)/norm2[update])[:, None, None]
                            *residual[update])
            disabled[selected[~np.isfinite(a[selected]).all((1, 2))]] = True
            update_seconds += time.perf_counter()-tick
            owned = [a, accepted, disabled, proof_credit, ids, aa, threshold, check, norm2, usable, unresolved, update]
            named_peak = max(named_peak, sum(t.nbytes for t in owned)+regs.nbytes+credit.nbytes
                             +x.nbytes+v.nbytes+out['named_array_bytes_subtotal'])
        if step in PREFIXES or step == steps:
            traces.append(dict(step=step, rejected=int((accepted > 0).sum()),
                               disabled=int(disabled.sum()), response_pairs=pairs,
                               elapsed_seconds=time.perf_counter()-started))
    arrays = dict(first_step=accepted, proof_credit=proof_credit, final_credit=a, disabled=disabled)
    return arrays, dict(proofs=proofs, traces=traces, exact_calls=calls, exact_empty_calls=empty_calls,
        response_pairs=pairs, oracle_seconds=oracle_seconds, exact_seconds=exact_seconds,
        update_seconds=update_seconds, total_seconds=time.perf_counter()-started,
        named_array_bytes_subtotal_max=named_peak, output_array_bytes=sum(t.nbytes for t in arrays.values()),
        memory_scope='Named arrays subtotal, not allocator peak; loaded credit generation excluded',
        query_targets_accessed=False, geometry_accessed=False)
