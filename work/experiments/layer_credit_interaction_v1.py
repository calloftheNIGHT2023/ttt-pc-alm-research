"""309: finite causal layer masks, not a fitted online selection rule."""
from fractions import Fraction as F
from itertools import product
import time

import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from complete_credit_rational_reference_v1 import direct_step
from complete_credit_amplitude_events_v2 import sweep


def mode_list(b, x):
    previous = np.broadcast_to(x, (len(b), len(x)))
    codes = []
    for j in range(b.shape[1]):
        z = previous + b[:, j, None]
        codes.append(np.searchsorted([0., .5, 1.], z, side='right').astype(np.uint8))
        previous = cold.base.g(z)
    return [np.stack(codes, axis=1)[i].tobytes().hex() for i in range(len(b))]


def masks(depth):
    return list(product((0, 1), repeat=depth))


def masked(state, mask):
    assert len(mask) == len(state['b']) and all(m in (0, 1) for m in mask)
    return {**state, 'direction': [[m * q for q in row] for m, row in zip(mask, state['direction'])]}


def decode(state):
    def fractions(v):
        return [fractions(q) for q in v] if isinstance(v, list) else F(v)
    return {k: fractions(v) for k, v in state.items()}


def at_one(v):
    return [at_one(q) for q in v] if isinstance(v, list) else v.at(F(1))


def exact_outputs(state):
    result = []
    norm = sum(q*q for r in state['direction'] for q in r)
    begin = time.perf_counter()
    for mask in masks(len(state['b'])):
        s = masked(state, mask)
        assert sum(q*q for r in s['direction'] for q in r) <= norm
        exact = direct_step(s, 1)
        # Independent full symbolic solver rebuilds the same complete update.
        other = sweep(s, F(1), strict=False)
        for name in ['b', 'h', 'forward']:
            assert at_one(other[name]) == exact[name], (mask, name)
        assert exact['mode'] == other['mode']
        result.append(dict(mask=''.join(map(str, mask)), exact=exact,
                           exact_mode=bytes(map(int, exact['mode'])).hex()))
    return result, time.perf_counter() - begin


def float_outputs(state):
    b, h, u, x, v = [np.array(state[k], dtype=float) for k in ['b', 'h', 'direction', 'x', 'v']]
    mm = np.array(masks(len(b)), dtype=float)
    count = len(mm)
    begin = time.perf_counter()
    with discovery_box(float(state['bound'])):
        local = cold.Local(np.repeat(b[None], count, axis=0), x, v, 'alm')
        local.h = np.repeat(h[:, None], count, axis=1)
        local.u = u[:, None] * mm.T[:, :, None]
        local.step()
        output = dict(b=local.b.copy(), h=local.h.copy(), u_after=local.u.copy(),
                      best=local.best.copy(), modes=mode_list(local.b, x))
    return output, time.perf_counter() - begin


def combine(state):
    exact, exact_seconds = exact_outputs(state)
    floating, float_seconds = float_outputs(state)
    for i, row in enumerate(exact):
        gaps = {name: float(np.max(abs(np.array(row['exact'][name], dtype=float) - value)))
                for name, value in [('b', floating['b'][i]), ('h', floating['h'][:, i])]}
        row.update(float_b=floating['b'][i].tolist(), float_h=floating['h'][:, i].tolist(),
                   float_u_after=floating['u_after'][:, i].tolist(), float_best=floating['best'][i].tolist(),
                   float_mode=floating['modes'][i], gaps=gaps,
                   float_value_discrepancy=max(gaps.values()) > 1e-10,
                   float_mode_discrepancy=floating['modes'][i] != row['exact_mode'])
    return exact, dict(exact_with_independent_verification_seconds=exact_seconds,
                       float_batch_initialization_step_and_mode_seconds=float_seconds,
                       float_batch_states=len(exact), full_online_fit=False,
                       resources_matched=False)


def branch_response(u, slopes, clipped, tau=F(1, 100)):
    """Exact local derivative rows for fixed internal activity policies.

    slopes[j] is the selected next-layer slope; the final entry is unused.
    This linear algebra primitive assumes that the stated policies remain valid.
    It does not infer policy validity or differentiate the parameter network.
    """
    depth = len(u)
    hh = [[F(0) for _ in u] for _ in u]
    for j in range(depth - 1, -1, -1):
        if clipped[j]:
            continue
        if j == depth - 1:
            hh[j][j] = -u[j] / (1 + tau)
        else:
            s = slopes[j]
            hh[j] = [s * q for q in hh[j + 1]]
            hh[j][j] -= u[j]
            hh[j][j + 1] += s * u[j + 1]
            hh[j] = [q / (1+s*s+tau) for q in hh[j]]
    return hh
