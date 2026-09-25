"""Independent LP/Fraction tests; LP and BP forbidden inside actual solver."""
from contextlib import contextmanager
from fractions import Fraction as F
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import scipy.optimize as opt
import region_conditioned_credit_v1 as model
import branch_image_chain_v1 as exact
import budget_reinvestment_suite_v1 as io
import batched_bp_discovery as bp

DESIGN = 'outputs/ttt-pc-alm-research/355_region_conditioned_credit_protocol_v1.md'
SOURCES = ['region_conditioned_credit_v1.py', 'test_region_conditioned_credit_v1.py',
           'run_region_conditioned_credit_v1.py', 'audit_region_conditioned_credit_v1.py',
           'branch_image_chain_dyadic_v1.py', 'branch_image_chain_v1.py',
           'factorized_dual_branch_search_v1.py', 'local_region_screen.py',
           'certified_branch_solver.py', 'budget_reinvestment_suite_v1.py']


def hashes(root):
    return {p: io.sha(root/p) for p in [DESIGN]+['work/experiments/'+s for s in SOURCES]}


@contextmanager
def guarded():
    def forbidden(*args, **kwargs):
        raise AssertionError('Global LP/BP/optimizer entered region-conditioned solver')
    targets = [opt.linprog, opt.minimize, bp.evaluate, bp.refine]
    patches = []
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for name, value in list(vars(module).items()):
            if any(value is target for target in targets):
                patches.append(patch.object(module, name, forbidden))
    torch = sys.modules.get('torch')
    if torch is not None:
        patches += [patch.object(torch.autograd, 'backward', forbidden),
                    patch.object(torch.autograd, 'grad', forbidden), patch.object(torch.Tensor, 'backward', forbidden)]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in reversed(patches):
            p.stop()


def lp_response(x, v, reg, credit):
    # Independent reduced variables [biases, activities]; no model.boxes call.
    d, n = reg.shape; m = d*n; dim = d+m
    r = np.zeros((m, dim)); offset = np.zeros(m); inequalities = []; rhs = []
    h_bounds = []
    for j in range(d):
        for i in range(n):
            code = int(reg[j, i]); slope = [0., 2., -2., 0.][code]
            intercept = [0., 0., 2., 0.][code]
            row = np.zeros(dim); row[j] = 1.; constant = x[i] if j == 0 else 0.
            if j:
                row[d+(j-1)*n+i] = 1.
            r[j*n+i, d+j*n+i] = 1.; r[j*n+i] -= slope*row
            offset[j*n+i] = -slope*constant-intercept
            inequalities.extend([row, -row])
            rhs.extend([[0., .5, 1., 1.12][code]-constant,
                        constant-[-.12, 0., .5, 1.][code]])
            lo, hi = 0., float(code in [1, 2])
            if j == d-1:
                lo, hi = max(lo, v[i]-.001), min(hi, v[i]+.001)
            if lo > hi:
                return None
            h_bounds.append((lo, hi))
    a = credit.ravel()
    result = opt.linprog(a@r, A_ub=inequalities, b_ub=rhs,
                        bounds=[(-.12, .12)]*d+h_bounds,
                        options=dict(primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9))
    if result.status == 2:
        return None
    assert result.success, result.message
    return result.fun+a@offset


def forward(x, b):
    h = x.copy(); regs = []
    for bias in b:
        z = h+bias; regs.append(np.searchsorted([0., .5, 1.], z, side='right'))
        h = np.maximum(0., 1.-abs(2*z-1))
    return h, np.array(regs, dtype=np.uint8)


def test():
    rng = np.random.default_rng(355071)
    counts = dict(oracle_cases=0, independent_lp=0, independent_fraction=0,
                  exact_empty=0, known_feasible_retained=0, repeat_arrays=0,
                  proof_checks=0, guarded_solves=0, projection_inequalities=0)
    maximum = 0.
    for d, n in [(1, 2), (2, 4), (4, 4), (4, 8)]:
        x = rng.uniform(0, 1, n); teacher = rng.uniform(-.12, .12, d)
        v, truth = forward(x, teacher)
        regs = np.array([truth]+[forward(x, b)[1] for b in rng.uniform(-.12, .12, (15, d))])
        credits = np.r_[rng.normal(size=(4, d, n)), np.zeros((1, d, n))]
        for credit in credits:
            out = model.response(x, v, regs, np.broadcast_to(credit, regs.shape))
            zl, zh, hl, hh = model.boxes(v, regs)
            for index, reg in enumerate(regs):
                reference = exact.fixed_value(x, v, credit, reg)
                lp = lp_response(x, v, reg, credit)
                counts['oracle_cases'] += 1; counts['independent_lp'] += 1
                counts['independent_fraction'] += 1
                if reference is None:
                    assert lp is None
                    counts['exact_empty'] += 1
                    continue
                assert out['valid'][index] and lp is not None
                gap = max(abs(float(reference)-out['objective'][index]), abs(lp-float(reference)))
                maximum = max(maximum, gap); assert gap < 1e-9
                assert np.all(out['z'][index] >= zl[index]-1e-12)
                assert np.all(out['z'][index] <= zh[index]+1e-12)
                assert np.all(out['h'][index] >= hl[index]-1e-12)
                assert np.all(out['h'][index] <= hh[index]+1e-12)
                prev = np.r_[x[None], out['h'][index, :-1]]
                assert np.max(abs(out['z'][index]-prev-out['b'][index, :, None])) < 1e-12
            with guarded():
                arrays, meta = model.solve(x, v, regs, credit)
                again, repeated = model.solve(x, v, regs, credit)
            counts['guarded_solves'] += 2
            assert arrays['first_step'][0] == 0
            counts['known_feasible_retained'] += 1
            for key, arr in arrays.items():
                assert arr.tobytes() == again[key].tobytes(); counts['repeat_arrays'] += 1
            assert meta['proofs'] == repeated['proofs']
            for proof in meta['proofs']:
                i = proof['index']; value = exact.fixed_value(x, v, arrays['proof_credit'][i], regs[i])
                assert (value is None and proof['structural_empty']) or (value == F(proof['lower']) > 0)
                counts['proof_checks'] += 1
    for _ in range(100):
        g = rng.normal(size=16); a = rng.normal(size=16)
        astar = rng.normal(size=16)
        astar += max(1.-astar@g, 0.)/(g@g)*g
        violation = max(1.-a@g, 0.)
        anew = a+violation/(g@g)*g
        assert np.sum((anew-astar)**2) <= np.sum((a-astar)**2)-violation**2/(g@g)+1e-10
        counts['projection_inequalities'] += 1
    return dict(passed=True, counts=counts, max_objective_gap=maximum,
                query_targets_accessed=False, global_solver_guard_passed=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    out = root/'results/region_conditioned_credit/preflight_v1'; out.mkdir(parents=True, exist_ok=False)
    frozen = hashes(root); result = test(); assert hashes(root) == frozen
    result['source_sha256'] = frozen
    io.save(out/'summary.json', result); print(result, flush=True)
