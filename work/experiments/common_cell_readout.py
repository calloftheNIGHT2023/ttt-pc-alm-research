"""Common global algebraic control, NOT a purely local PC learning rule.

Context-only cell projection and one-dimensional constant-output fiber readout.
No query targets enter any interface here. Branch affine maps use explicit
slopes/intercepts, including at kinks; no loss backpropagation is used here.
"""
import time
import numpy as np
from scipy.optimize import linprog, minimize, nnls
import vector_interval_memory as base

CHECK_TOL = 1e-7
SLOPES = np.array([0., 2., -2., 0.])
INTERCEPTS = np.array([-1., 1., 1., -1.])


def cell(point, x, weights):
    depth, width = point.shape
    size = point.size
    theta = point.ravel()
    offset = np.array(x, copy=True)
    design = np.zeros((*x.shape, size))
    matrices, bounds, patterns = [], [], []
    for layer, weight in enumerate(weights):
        z_design = np.einsum('ab,nbp->nap', weight, design, optimize=True)
        z_design[:, np.arange(width), layer * width + np.arange(width)] += 1
        z_offset = offset @ weight.T
        z = z_offset + np.einsum('nwp,p->nw', z_design, theta)
        branch = np.searchsorted([-1., 0., 1.], z, side='left')
        lower = np.array([-np.inf, -1., 0., 1.])[branch]
        upper = np.array([-1., 0., 1., np.inf])[branch]
        for sign, limit in [(1., upper), (-1., -lower)]:
            mask = np.isfinite(limit)
            matrices.append(sign * z_design[mask])
            bounds.append(limit[mask] - sign * z_offset[mask])
        design = SLOPES[branch, None] * z_design
        offset = SLOPES[branch] * z_offset + INTERCEPTS[branch]
        patterns.append(branch)
    matrices.extend([np.eye(size), -np.eye(size)])
    bounds.extend([np.full(size, base.BOUND), np.full(size, base.BOUND)])
    return dict(J=design.reshape(-1, size), c=offset.ravel(),
                A=np.concatenate(matrices), b=np.concatenate(bounds),
                patterns=patterns)


def project(point, anchor, x, v, weights):
    before = time.perf_counter()
    region = cell(point, x, weights)
    j, c, a, b = [region[k] for k in ('J', 'c', 'A', 'b')]
    all_a = np.vstack([a, j, -j])
    all_b = np.r_[b, v.ravel() + base.EPS - c, -v.ravel() + base.EPS + c]
    meta = dict(cell_constraints=len(all_b), geometry_matrix_bytes=int(
        sum(z.nbytes for z in [j, c, a, b, all_a, all_b])),
        projection_status='pending', qp_success=False)
    lp = linprog(np.zeros(point.size), A_ub=all_a, b_ub=all_b,
                 bounds=[(None, None)] * point.size, method='highs',
                 options={'primal_feasibility_tolerance': 1e-9,
                          'dual_feasibility_tolerance': 1e-9})
    meta.update(lp_status=int(lp.status))
    result = point.copy()
    if lp.success:
        old = anchor.ravel()
        start = point.ravel() if np.max(all_a @ point.ravel() - all_b) <= 1e-10 else lp.x
        qp = minimize(lambda z: .5 * np.sum((z - old) ** 2), start,
                      jac=lambda z: z - old, method='SLSQP',
                      constraints=[{'type': 'ineq', 'fun': lambda z: all_b - all_a @ z,
                                    'jac': lambda z: -all_a}],
                      options={'maxiter': 300, 'ftol': 1e-12})
        trial = qp.x.reshape(point.shape)
        violation = float(np.max(all_a @ qp.x - all_b))
        true_error = float(np.max(np.abs(base.forward(trial[None], x, weights)[0] - v)))
        meta.update(qp_success=bool(qp.success), qp_iterations=int(qp.nit),
                    qp_linear_violation=violation, qp_true_max_error=true_error)
        if qp.success and violation <= CHECK_TOL and true_error <= base.EPS + CHECK_TOL:
            result = trial
            meta['projection_status'] = 'projected'
            slack = all_b - all_a @ qp.x
            active = slack <= 1e-7
            if np.any(active):
                try:
                    multiplier, residual = nnls(all_a[active].T, old - qp.x, maxiter=10000)
                    meta['qp_stationarity_residual'] = float(residual)
                    meta['qp_complementarity_residual'] = float(np.max(np.abs(multiplier * slack[active])))
                except RuntimeError:
                    meta['qp_stationarity_residual'] = None
            else:
                meta['qp_stationarity_residual'] = float(np.linalg.norm(qp.x - old))
        else:
            meta['projection_status'] = 'qp_failed_raw_fallback'
    else:
        meta['projection_status'] = 'cell_infeasible_raw_fallback' if lp.status == 2 else 'lp_failed_raw_fallback'
    meta['projection_seconds'] = time.perf_counter() - before
    return result, region, meta


def fiber(point, region, x, v, weights):
    before = time.perf_counter()
    j, a, b = [region[k] for k in ('J', 'A', 'b')]
    _, singular, vh = np.linalg.svd(j, full_matrices=True)
    tolerance = 100 * max(j.shape) * np.finfo(float).eps * max(1., singular[0])
    rank = int(np.sum(singular > tolerance))
    meta = dict(rank=rank, nullity=point.size - rank, rank_tolerance=float(tolerance),
                fiber_status='not_one_dimensional', fiber_state_bytes=0)
    state = None
    if point.size - rank == 1:
        direction = vh[-1].copy()
        index = np.flatnonzero(np.abs(direction) > 1e-12)[0]
        if direction[index] < 0:
            direction *= -1
        coefficients = a @ direction
        margins = b - a @ point.ravel()
        if np.min(margins) >= -CHECK_TOL:
            margins = np.maximum(margins, 0.)
            positive, negative = coefficients > 1e-12, coefficients < -1e-12
            lower = float(np.max(margins[negative] / coefficients[negative]))
            upper = float(np.min(margins[positive] / coefficients[positive]))
            lower *= 1 - 1e-9
            upper *= 1 - 1e-9
            unit = direction.reshape(point.shape)
            endpoints = point[None] + np.array([lower, 0., upper])[:, None, None] * unit
            output = base.forward(endpoints, x, weights)
            invariance = float(np.max(np.abs(output - output[1])))
            error = float(np.max(np.abs(output - v)))
            meta.update(fiber_interval=[lower, upper], fiber_width=upper - lower,
                        fiber_output_invariance=invariance, fiber_support_max_error=error,
                        null_residual=float(np.linalg.norm(j @ direction)))
            if (upper - lower > 1e-10 and invariance <= 1e-9 and
                    error <= base.EPS + CHECK_TOL and
                    np.max(np.abs(endpoints)) <= base.BOUND + CHECK_TOL):
                state = dict(point=point.copy(), direction=unit, lower=lower, upper=upper)
                meta.update(fiber_status='valid', fiber_state_bytes=int(point.nbytes + unit.nbytes + 16))
            else:
                meta['fiber_status'] = 'degenerate_or_failed_checks'
        else:
            meta['fiber_status'] = 'outside_cell'
    meta['fiber_seconds'] = time.perf_counter() - before
    return state, meta


def integrate_line(x, weights, state):
    """Piecewise-affine function integral, analytically exact apart from floats.

    Each query has its own branch subdivision. No quadrature density or target-
    dependent stopping rule. Splits only at true preactivation roots in the
    open current interval. Returns working-array accounting, not allocator peak.
    """
    low, high = state['lower'], state['upper']
    point, direction = state['point'], state['direction']
    lower, upper = np.full(len(x), low), np.full(len(x), high)
    owner = np.arange(len(x))
    slope, intercept = np.zeros_like(x), x.copy()
    maximum_segments, working_bytes = len(x), 0
    for layer, weight in enumerate(weights):
        za = slope @ weight.T + direction[layer]
        zb = intercept @ weight.T + point[layer]
        with np.errstate(divide='ignore', invalid='ignore'):
            roots = (np.array([-1., 0., 1.])[None, None, :] - zb[:, :, None]) / za[:, :, None]
        roots = roots.reshape(len(lower), -1)
        roots = np.where(np.isfinite(roots), np.clip(roots, lower[:, None], upper[:, None]), lower[:, None])
        cuts = np.sort(np.c_[lower, roots, upper], axis=1)
        left, right = cuts[:, :-1], cuts[:, 1:]
        parent, column = np.nonzero(right > left)
        next_lower, next_upper = left[parent, column], right[parent, column]
        midpoint = (next_lower + next_upper) / 2
        branch = np.searchsorted([-1., 0., 1.], za[parent] * midpoint[:, None] + zb[parent], side='left')
        slope = SLOPES[branch] * za[parent]
        intercept = SLOPES[branch] * zb[parent] + INTERCEPTS[branch]
        owner = owner[parent]
        lower, upper = next_lower, next_upper
        maximum_segments = max(maximum_segments, len(owner))
        working_bytes = max(working_bytes, sum(z.nbytes for z in
            [za, zb, roots, cuts, parent, column, slope, intercept, owner, lower, upper, branch]))
    integral = (upper - lower)[:, None] * (slope * ((lower + upper) / 2)[:, None] + intercept)
    result = np.zeros_like(x)
    np.add.at(result, owner, integral / (high - low))
    return result, dict(integration_max_segments=maximum_segments,
                        integration_working_array_bytes=int(working_bytes))


def read(x, weights, point, state, mode):
    if mode == 'point' or state is None:
        return base.forward(point[None], x, weights)[0], {}
    if mode == 'midpoint':
        middle = point + .5 * (state['lower'] + state['upper']) * state['direction']
        return base.forward(middle[None], x, weights)[0], {}
    if mode == 'mean':
        return integrate_line(x, weights, state)
    raise ValueError(mode)
