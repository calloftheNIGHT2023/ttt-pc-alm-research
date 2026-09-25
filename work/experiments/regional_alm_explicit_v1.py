"""378 explicit steps and reentrant global-solver guard; 376 math unchanged."""
from contextlib import contextmanager
from contextvars import ContextVar
import time
import numpy as np
import regional_alm_v1 as old

_depth = ContextVar('regional_solver_guard_depth', default=0)
FAMILIES = ['regional_active', 'regional_passive', 'regional_instant', 'pdhg_cold', 'pdhg_box']


@contextmanager
def guarded():
    if _depth.get():
        token = _depth.set(_depth.get()+1)
        try:
            yield
        finally:
            _depth.reset(token)
    else:
        with old.guarded():
            token = _depth.set(1)
            try:
                yield
            finally:
                _depth.reset(token)


def solve(x, v, regs, *, family, steps):
    assert family in FAMILIES and type(steps) is int and steps > 0
    start = time.perf_counter()
    with guarded():
        arrays, meta = _solve(x, v, regs, family, steps, start)
    meta['total_seconds'] = time.perf_counter()-start
    return arrays, meta


def _solve(x, v, regs, family, steps, start):
    r, d, n = regs.shape
    assert r > 0 and n > 0
    pdhg = family.startswith('pdhg'); project = family != 'pdhg_cold'
    active = family == 'regional_active'; instant = family == 'regional_instant'
    b, z, h, box = old.initialize(x, v, regs, project)
    zl, zh, hl, hh = box
    s = old.screen.base.SLOPES[regs]; c = old.screen.base.INTERCEPTS[regs]
    assert np.all(zl <= zh) and np.all(hl <= hh)
    p = np.zeros_like(z); a = p.copy(); lastp = p.copy(); lasta = a.copy()
    initial = {'initial_b': b.copy(), 'initial_z': z.copy(), 'initial_h': h.copy()}
    zero = np.zeros_like(p); bb = b.copy(); zb = z.copy(); hb = h.copy()
    tb = .99/n; tz = .99/(1+abs(s)); th = np.full((1, d, 1), .99/2); th[:, -1] = .99
    sp = np.full((1, d, 1), .99/3); sp[:, 0] = .99/2; sa = .99/(1+abs(s))
    first = np.full(r, -1, np.int32); proofp = np.zeros_like(p); proofa = np.zeros_like(a)
    lower = np.zeros(r); kind = np.full(r, -1, np.int8)
    checkpoints = []; proposal_rows = 0; certified_checks = 0

    def check(step, rp, ra):
        nonlocal lastp, lasta, proposal_rows, certified_checks
        for code, (pp, aa) in enumerate([(p, a), (p-lastp, a-lasta), (rp, ra)]):
            proposal_rows += r
            rough = old.screen.float_bound(x, v, regs, pp, aa)
            ids = np.flatnonzero((rough > 1e-10*(1+abs(pp).sum((1, 2))+abs(aa).sum((1, 2)))) & (first < 0))
            if len(ids):
                vals = old.screen.certified_lower_bound(x, v, regs[ids], pp[ids], aa[ids])
                certified_checks += len(ids); good = vals > 0; accepted = ids[good]
                first[accepted] = step; proofp[accepted] = pp[accepted]; proofa[accepted] = aa[accepted]
                lower[accepted] = vals[good]; kind[accepted] = code
        lastp = p.copy(); lasta = a.copy()
        checkpoints.append(dict(step=step, certified=int((first >= 0).sum()), seconds=time.perf_counter()-start,
                                proposal_rows=proposal_rows, rounded_checks=certified_checks))

    check(0, *old.residual(x, s, c, b, z, h))
    for it in range(steps):
        if pdhg:
            p += sp*(zb-old.previous(x, hb)-bb[:, :, None]); a += sa*(hb-s*zb-c)
            oldb, oldz, oldh = b, z, h
            b = np.clip(oldb+tb*p.sum(2), -.12, .12); z = np.clip(oldz-tz*(p-s*a), zl, zh)
            hc = a.copy(); hc[:, :-1] -= p[:, 1:]; h = np.clip(oldh-th*hc, hl, hh)
            bb, zb, hb = 2*b-oldb, 2*z-oldz, 2*h-oldh
        else:
            pp, aa = (p, a) if active else (zero, zero)
            b = old.block_b(x, b, z, h, pp)
            z = old.block_z(x, b, z, h, pp, aa, s, c, zl, zh)
            h = old.block_h(b, z, h, pp, aa, s, c, hl, hh)
        rp, ra = old.residual(x, s, c, b, z, h)
        if not pdhg and not instant:
            p += .5*rp; a += .5*ra
        if (it+1) % 32 == 0 or it+1 == steps:
            check(it+1, rp, ra)
    arrays = {**initial, 'final_b': b, 'final_z': z, 'final_h': h, 'final_p': p, 'final_a': a,
              'first_step': first, 'proof_p': proofp, 'proof_a': proofa, 'proof_lower': lower, 'proof_kind': kind}
    named = [b, z, h, p, a, lastp, lasta, bb, zb, hb, zero, zl, zh, hl, hh, s, c, tz, sa, rp, ra,
             first, proofp, proofa, lower, kind]
    meta = dict(family=family, steps=steps, checkpoints=checkpoints, proposal_rows=proposal_rows,
                rounded_checks=certified_checks, live_named_array_bytes=sum(t.nbytes for t in named),
                initial_archive_bytes=sum(t.nbytes for t in initial.values()),
                returned_array_bytes=sum(t.nbytes for t in arrays.values()),
                memory_scope='Named arrays only, not process peak', explicit_steps=True,
                query_targets_accessed=False, suffix_accessed=False, global_bp_used=False, lp_used=False)
    return arrays, meta
