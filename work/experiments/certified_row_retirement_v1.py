"""380 row-separable solver: preserve first proofs, retire only strict certificates."""
import time
import numpy as np
import regional_alm_explicit_v1 as reference

ORIGINAL_SOLVE = reference.solve
FIELDS = ['b', 'z', 'h', 'p', 'a', 'lastp', 'lasta', 'zero', 'bb', 'zb', 'hb',
          'zl', 'zh', 'hl', 'hh', 's', 'c', 'tz', 'sa']


def solve(x, v, regs, *, family, steps):
    assert family in reference.FAMILIES and type(steps) is int and steps > 0
    start = time.perf_counter()
    with reference.guarded():
        arrays, meta = _solve(x, v, regs, family, steps, start)
    meta['total_seconds'] = time.perf_counter()-start
    return arrays, meta


def _solve(x, v, regs, family, steps, start):
    old = reference.old; r, d, n = regs.shape
    assert r > 0 and n > 0
    pdhg = family.startswith('pdhg'); active = family == 'regional_active'; instant = family == 'regional_instant'
    b, z, h, box = old.initialize(x, v, regs, family != 'pdhg_cold')
    zl, zh, hl, hh = box; assert np.all(zl <= zh) and np.all(hl <= hh)
    s = old.screen.base.SLOPES[regs]; c = old.screen.base.INTERCEPTS[regs]
    p = np.zeros_like(z); a = p.copy()
    state = dict(b=b, z=z, h=h, p=p, a=a, lastp=p.copy(), lasta=a.copy(), zero=np.zeros_like(p),
                 bb=b.copy(), zb=z.copy(), hb=h.copy(), zl=zl, zh=zh, hl=hl, hh=hh, s=s, c=c,
                 tz=.99/(1+abs(s)), sa=.99/(1+abs(s)))
    initial = {'initial_b': b.copy(), 'initial_z': z.copy(), 'initial_h': h.copy()}
    stopped = {key: np.empty_like(state[key]) for key in ['b', 'z', 'h', 'p', 'a']}
    # Avoid keeping aliases to full work arrays after compaction.
    del b, z, h, box, zl, zh, hl, hh, s, c, p, a
    tb = .99/n; th = np.full((1, d, 1), .99/2); th[:, -1] = .99
    sp = np.full((1, d, 1), .99/3); sp[:, 0] = .99/2
    first = np.full(r, -1, np.int32); proofp = np.zeros((r, d, n)); proofa = np.zeros_like(proofp)
    lower = np.zeros(r); kind = np.full(r, -1, np.int8); alive = np.arange(r)
    proposal_rows = rounded_checks = row_steps = executed_sweeps = peak_work_bytes = 0
    checkpoints = []

    def memory():
        return sum(value.nbytes for value in state.values())+sum(value.nbytes for value in stopped.values())+sum(
            value.nbytes for value in [alive, first, proofp, proofa, lower, kind, th, sp])

    def check(step, rp, ra):
        nonlocal proposal_rows, rounded_checks, alive, peak_work_bytes
        live_before = len(alive); peak_work_bytes = max(peak_work_bytes, memory())
        if live_before:
            current_regs = regs[alive]
            for code, (pp, aa) in enumerate([(state['p'], state['a']),
                                             (state['p']-state['lastp'], state['a']-state['lasta']), (rp, ra)]):
                proposal_rows += live_before
                rough = old.screen.float_bound(x, v, current_regs, pp, aa)
                ids = np.flatnonzero((rough > 1e-10*(1+abs(pp).sum((1, 2))+abs(aa).sum((1, 2)))) & (first[alive] < 0))
                if len(ids):
                    vals = old.screen.certified_lower_bound(x, v, current_regs[ids], pp[ids], aa[ids])
                    rounded_checks += len(ids); good = vals > 0; accepted = ids[good]; original_ids = alive[accepted]
                    first[original_ids] = step; proofp[original_ids] = pp[accepted]; proofa[original_ids] = aa[accepted]
                    lower[original_ids] = vals[good]; kind[original_ids] = code
            state['lastp'] = state['p'].copy(); state['lasta'] = state['a'].copy()
            retired = first[alive] >= 0
            for key in stopped: stopped[key][alive[retired]] = state[key][retired]
            if np.any(retired):
                keep = ~retired
                state.update({key: state[key][keep].copy() for key in FIELDS})
                alive = alive[keep].copy()
        checkpoints.append(dict(step=step, certified=int((first >= 0).sum()), active_before=live_before, active_after=len(alive),
                                seconds=time.perf_counter()-start, proposal_rows=proposal_rows, rounded_checks=rounded_checks,
                                cumulative_row_steps=row_steps))

    check(0, *old.residual(x, state['s'], state['c'], state['b'], state['z'], state['h']))
    for it in range(steps):
        if len(alive):
            row_steps += len(alive); executed_sweeps += 1
            if pdhg:
                state['p'] += sp*(state['zb']-old.previous(x, state['hb'])-state['bb'][:, :, None])
                state['a'] += state['sa']*(state['hb']-state['s']*state['zb']-state['c'])
                b, z, h = state['b'], state['z'], state['h']
                state['b'] = np.clip(b+tb*state['p'].sum(2), -.12, .12)
                state['z'] = np.clip(z-state['tz']*(state['p']-state['s']*state['a']), state['zl'], state['zh'])
                hc = state['a'].copy(); hc[:, :-1] -= state['p'][:, 1:]
                state['h'] = np.clip(h-th*hc, state['hl'], state['hh'])
                state['bb'], state['zb'], state['hb'] = 2*state['b']-b, 2*state['z']-z, 2*state['h']-h
                del b, z, h, hc
            else:
                pp, aa = (state['p'], state['a']) if active else (state['zero'], state['zero'])
                state['b'] = old.block_b(x, state['b'], state['z'], state['h'], pp)
                state['z'] = old.block_z(x, state['b'], state['z'], state['h'], pp, aa, state['s'], state['c'], state['zl'], state['zh'])
                state['h'] = old.block_h(state['b'], state['z'], state['h'], pp, aa, state['s'], state['c'], state['hl'], state['hh'])
                del pp, aa
            rp, ra = old.residual(x, state['s'], state['c'], state['b'], state['z'], state['h'])
            if not pdhg and not instant:
                state['p'] += .5*rp; state['a'] += .5*ra
        else:
            rp = ra = None
        if (it+1) % 32 == 0 or it+1 == steps: check(it+1, rp, ra)
        del rp, ra
    for key in stopped: stopped[key][alive] = state[key]
    arrays = {**initial, **{'stopped_'+key: value for key, value in stopped.items()},
              'first_step': first, 'proof_p': proofp, 'proof_a': proofa, 'proof_lower': lower, 'proof_kind': kind}
    assert row_steps == int(np.where(first >= 0, first, steps).sum())
    meta = dict(family=family, steps=steps, checkpoints=checkpoints, proposal_rows=proposal_rows, rounded_checks=rounded_checks,
                row_steps=row_steps, coordinate_steps=row_steps*d*n, full_row_steps_upper_bound=r*steps,
                full_coordinate_steps_upper_bound=r*steps*d*n, executed_sweeps=executed_sweeps,
                live_named_array_bytes=peak_work_bytes, initial_archive_bytes=sum(value.nbytes for value in initial.values()),
                returned_array_bytes=sum(value.nbytes for value in arrays.values()), explicit_steps=True,
                strict_certificate_retirement=True, state_meaning='First strict proof state for retired rows, cap state for unresolved rows',
                memory_scope='Named array accounting, not process peak; compaction temporaries not fully included',
                query_targets_accessed=False, suffix_accessed=False, global_bp_used=False, lp_used=False)
    return arrays, meta
