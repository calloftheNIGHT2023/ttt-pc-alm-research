"""Fixed-budget exponential credit gains with analytic local best responses.

Support-only: no LP, BP, query, or previous minimax solution is an input.
Floating iterations propose; original Fraction credit evaluation certifies.
"""
import time
import numpy as np
import optimized_branch_dual as original

PREFIXES = (1, 4, 16, 64, 128)
STEPS = 128


def response(x, v, regs, a):
    count, d, n = regs.shape
    zl, zh, hl, hh = original.screen.boxes(v, regs)
    sa = original.base.SLOPES[regs] * a
    b = np.zeros((count, d)); h = np.empty_like(a); z = np.empty_like(a)
    valid = np.ones(count, bool)
    h[:, -1] = np.where(a[:, -1] > 0, hl[:, -1],
                        np.where(a[:, -1] < 0, hh[:, -1], (hl[:, -1] + hh[:, -1]) / 2))
    for j in range(d):
        pl = np.broadcast_to(x, (count, n)) if j == 0 else hl[:, j-1]
        ph = np.broadcast_to(x, pl.shape) if j == 0 else hh[:, j-1]
        ap = np.zeros_like(pl) if j == 0 else a[:, j-1]
        c = ap - sa[:, j]
        low = np.maximum(-original.screen.B, (zl[:, j] - ph).max(1))
        high = np.minimum(original.screen.B, (zh[:, j] - pl).min(1))
        valid &= low <= high
        kink = np.where(c >= 0, zl[:, j] - pl, zh[:, j] - ph)
        options = np.clip(np.column_stack([low, high, kink]), low[:, None], high[:, None])
        lo = np.maximum(pl[:, None], zl[:, j, None] - options[:, :, None])
        hi = np.minimum(ph[:, None], zh[:, j, None] - options[:, :, None])
        values = (c[:, None] * np.where(c[:, None] >= 0, lo, hi)
                  - sa[:, j, None] * options[:, :, None]).sum(2)
        ties = values == values.min(1)[:, None]
        b[:, j] = (np.where(ties, options, np.inf).min(1)
                   + np.where(ties, options, -np.inf).max(1)) / 2
        lo = np.maximum(pl, zl[:, j] - b[:, j, None])
        hi = np.minimum(ph, zh[:, j] - b[:, j, None])
        hp = np.where(c > 0, lo, np.where(c < 0, hi, (lo + hi) / 2))
        if j:
            h[:, j-1] = hp
        z[:, j] = hp + b[:, j, None]
    residual = h - (original.base.SLOPES[regs] * z + original.base.INTERCEPTS[regs])
    objective = (a * residual).sum((1, 2))
    # A named-array subtotal, not an allocator peak or complete memory claim.
    named = [zl, zh, hl, hh, sa, b, h, z, valid, residual, objective,
             c, low, high, kink, options, lo, hi, values, ties, hp]
    return dict(b=b, h=h, z=z, residual=residual, objective=objective,
                valid=valid, named_array_bytes_subtotal=sum(q.nbytes for q in named))


def solve(x, v, regs, bank):
    started = time.perf_counter()
    count, d, n = regs.shape; k = len(bank)
    assert k > 0 and np.isfinite(bank).all() and np.max(abs(bank)) <= 1
    eta = float(np.sqrt(2 * np.log(k) / STEPS)); bound = d * n
    flat = bank.reshape(k, d * n)
    logits = np.zeros((count, k)); best_weights = np.zeros_like(logits)
    best_credit = np.zeros((count, d, n)); best = np.full(count, -np.inf)
    first = np.zeros(count, np.int32); best_step = np.zeros(count, np.int32)
    prefix_values = np.full((count, len(PREFIXES)), -np.inf)
    proofs = []; traces = []; exact_calls = 0; oracle_pairs = 0
    oracle_seconds = 0.; update_seconds = 0.; exact_seconds = 0.; peak_named = 0
    for step in range(1, STEPS + 1):
        ids = np.flatnonzero(first == 0)
        if len(ids):
            clock = time.perf_counter()
            shifted = logits[ids] - logits[ids].max(1, keepdims=True)
            weights = np.exp(shifted); weights /= weights.sum(1, keepdims=True)
            a = (weights @ flat).reshape(len(ids), d, n)
            update_seconds += time.perf_counter() - clock
            clock = time.perf_counter(); out = response(x, v, regs[ids], a)
            oracle_seconds += time.perf_counter() - clock; oracle_pairs += len(ids)
            assert np.all(out['valid']), 'Empty relaxation requires a separate proof route'
            assert np.max(abs(out['residual'])) <= 1 + 1e-12
            values = out['objective']; improved = values > best[ids]; ii = ids[improved]
            best[ii] = values[improved]; best_weights[ii] = weights[improved]
            best_credit[ii] = a[improved]; best_step[ii] = step
            for local in np.flatnonzero(values > 0):
                index = int(ids[local]); clock = time.perf_counter()
                exact = original.exact_optimum(x, v, regs[index], a[local])
                exact_seconds += time.perf_counter() - clock; exact_calls += 1
                if exact['positive']:
                    assert 'empty_layer' not in exact
                    first[index] = step; best_weights[index] = weights[local]
                    best_credit[index] = a[local]; best[index] = values[local]; best_step[index] = step
                    proofs.append(dict(index=index, step=step, pattern=regs[index].tobytes().hex(), exact=exact))
            clock = time.perf_counter()
            gains = out['residual'].reshape(len(ids), d*n) @ flat.T
            logits[ids] += (eta / bound) * gains
            logits[ids] -= logits[ids].max(1, keepdims=True)
            update_seconds += time.perf_counter() - clock
            owned = [logits, best_weights, best_credit, best, first, best_step, prefix_values,
                     ids, shifted, weights, a, gains, improved, ii]
            peak_named = max(peak_named, bank.nbytes + regs.nbytes + x.nbytes + v.nbytes
                             + sum(q.nbytes for q in owned) + out['named_array_bytes_subtotal'])
        if step in PREFIXES:
            prefix_values[:, PREFIXES.index(step)] = best
            traces.append(dict(step=step, positive=int(np.sum(first > 0)),
                               oracle_pairs=oracle_pairs, elapsed_seconds=time.perf_counter()-started))
    arrays = dict(credit=best_credit, weights=best_weights, first_step=first,
                  best_step=best_step, best_value=best, prefix_values=prefix_values)
    return arrays, dict(regions=count, directions=k, eta=eta, payoff_bound=bound, steps=STEPS,
        proofs=proofs, traces=traces, positive=len(proofs), exact_calls=exact_calls,
        oracle_pairs=oracle_pairs, oracle_seconds=oracle_seconds, update_seconds=update_seconds,
        exact_seconds=exact_seconds, total_seconds=time.perf_counter()-started,
        named_array_bytes_subtotal_max=peak_named,
        output_array_bytes=sum(q.nbytes for q in arrays.values()),
        memory_scope='Named NumPy arrays subtotal only; Python objects, hidden temporaries and allocator peak need fresh-process measurement')
