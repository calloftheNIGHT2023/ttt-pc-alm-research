"""347 actual full-trial strong pipeline plus common support-only credit search.

No old saved pool, predictor, certificate, teacher or query answer is read.
"""
import hashlib
import time
import numpy as np
import scipy.optimize as opt
import cold_stagnation_switch as cold
import local_dual_jump_transfer as transfer
import shared_mode_readout as shared
import conditioned_mode_geometry as conditioned
import online_credit_branch_search_v1 as local_search
import support_consistency_trigger_dyadic_v1 as trigger
import support_language_chain_v1 as language
from audit_local_dual_jump_modes import modes
from run_factorized_dual_branch_search_v1 import signals, bp_control
from verify_known_range_projection_v1 import project

CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']


def state_digest(state):
    h = hashlib.sha256()
    for name, a in sorted(state.arrays().items()):
        h.update(name.encode()); h.update(a.dtype.str.encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


class FirstFit:
    def __init__(self, x, v):
        self.x, self.v = x, v
        self.index = self.calls = self.copies = 0
        self.best = self.selected = None
        self.done = False; self.seconds = 0.

    def observe(self, state, step):
        begin = time.perf_counter(); offset = self.index; self.index += len(state.b)
        if not self.done:
            for r in range(len(state.b)):
                loss = trigger.support_loss(state.b[r], self.x, self.v); self.calls += 1
                if self.best is None or loss < self.best:
                    self.best = loss; self.copies += 1
                    self.selected = dict(index=offset+r, location=[2, step, r],
                        b=state.b[r].copy(), h=state.h[:, r].copy(), u=state.u[:, r].copy())
                if loss == 0:
                    self.done = True; break
        self.seconds += time.perf_counter()-begin

    def finish(self, expected):
        assert self.index == expected and self.selected is not None
        begin = time.perf_counter(); s = self.selected
        loss = trigger.support_loss(s['b'], self.x, self.v)
        _, pattern = trigger.exact_forward(s['b'], self.x)
        assert loss == self.best
        s.update(exact_support_loss=str(loss), support_fit=loss == 0,
                 original_mode=np.array(pattern, dtype=np.uint8).tobytes().hex())
        self.seconds += time.perf_counter()-begin
        return s


def fit(x, v, q, seed, *, channel=None, watch=False, solver='alm', steps=64, trace_hash=False):
    assert channel is None or channel in CHANNELS
    assert solver in ['alm', 'pc', 'nodual', 'adam'] and steps > 0
    assert channel is None or (solver == 'alm' and steps == 64)
    assert not watch or (solver == 'alm' and steps == 64 and channel is None)
    with local_search.no_bp(solver != 'adam' and channel != 'bp'):
        return _fit(x, v, q, seed, channel, watch, solver, steps, trace_hash)


def _fit(x, v, q, seed, channel, watch, solver, steps, trace_hash):
    assert cold.base.BOUND == .12 and len(x) == len(v) == 4
    begin = time.perf_counter(); state = cold.Local(cold.starts(), x, v, 'nodual')
    for _ in range(16): state.step()
    preparation = time.perf_counter()-begin; tick = time.perf_counter()
    bb = []; hh = []; uu = []; best = []; seen = {}; atomic_trials = 0
    def collect(bank):
        for key, b in modes(x, bank).items():
            if key not in seen: seen[key] = b.copy()
    for r in range(17):
        a, _ = transfer.run(state.b[r], state.h[:, r], state.best[r], x, v, None, 'branch_probe')
        collect(a['trial_b']); n = len(a['trial_b']); atomic_trials += n
        bb.extend(a['trial_b']); hh.extend(a['trial_h'].transpose(1, 0, 2)); uu.extend(a['trial_u'].transpose(1, 0, 2))
        best.extend(np.tile(a['best'], (n, 1)))
    b = np.array(bb); h = np.stack(hh, axis=1); u = np.stack(uu, axis=1); best = np.array(best)
    assert len(b) == 629 and not u.any()
    atomic = time.perf_counter()-tick; tick = time.perf_counter()
    observer = FirstFit(x, v) if watch or channel is not None else None
    fingerprints = []; digest_seconds = 0.
    if solver == 'adam':
        saved = cold.bp.evaluate
        def record(bank, xx, vv, with_jacobian=True):
            collect(bank); return saved(bank, xx, vv, with_jacobian)
        try:
            cold.bp.evaluate = record
            answer, meta = cold.bp.refine(b, x, v, np.zeros(4), solver='adam', steps=steps)
        finally: cold.bp.evaluate = saved
        cold.frozen.retain(best, answer, x, v, np.zeros(4)); solver_bytes = meta['major_arrays_bytes_subtotal']
    else:
        state = cold.Local(b, x, v, solver); state.h = h; state.u = u; state.best = best.copy()
        cold.frozen.retain(state.best, b, x, v, np.zeros(4)); state.errors, state.moves = cold.base.score(state.best, x, v, np.zeros(4))
        collect(state.b)
        def digest():
            nonlocal digest_seconds
            t = time.perf_counter(); fingerprints.append(state_digest(state)); digest_seconds += time.perf_counter()-t
        if trace_hash: digest()
        for step in range(1, steps+1):
            state.step(); collect(state.b)
            if observer is not None: observer.observe(state, step)
            if trace_hash: digest()
        best = state.best.copy(); solver_bytes = state.numeric_state_bytes()
    _, selected = cold.select(best, x, v)
    selected_state = observer.finish(629*steps) if observer is not None else None
    continuation = time.perf_counter()-tick; tick = time.perf_counter()
    proposal = None; credit = None; bp_gap = None
    if channel is not None:
        s = selected_state
        if channel == 'bp': credit, bp_gap = bp_control(s['b'], x, v)
        else: credit = signals(s['b'], s['h'], s['u'], x, seed, s['location'])[channel]
        previous = []
        def forbid(*args, **kwargs): raise AssertionError('Global solver or BP entered strong-pool branch selector')
        try:
            for obj, name in [(cold.bp, 'evaluate'), (cold.bp, 'refine'), (opt, 'linprog'), (opt, 'minimize')]:
                previous.append((obj, name, getattr(obj, name))); setattr(obj, name, forbid)
            pattern = np.array(list(bytes.fromhex(s['original_mode']))).reshape(4, 4)
            proposal = language.propose(x, v, credit, pattern, k=8)
        finally:
            for obj, name, method in previous: setattr(obj, name, method)
    credit_search = time.perf_counter()-tick; tick = time.perf_counter()
    polys = {}; feasible = []; repairs = []
    with conditioned.geometry_scope(repairs):
        for key, representative in sorted(seen.items()):
            note = cold.base.branch_feasibility(x, v, representative)
            if note['current_branch_feasible']:
                feasible.append(key); poly, proof = shared.build(x, v, representative)
                if poly is not None: polys[key] = poly
    original_positive = sorted(polys); geometry_seconds = time.perf_counter()-tick; tick = time.perf_counter()
    classifications = {}; hits = 0
    if proposal is not None:
        for key in sorted({p['mode'] for p in proposal['proposals']}):
            if key in polys:
                hits += 1; continue
            poly, note = local_search.explicit_geometry(x, v, key); classifications[key] = note
            if poly is not None: polys[key] = poly
    extra_geometry = time.perf_counter()-tick; keys = sorted(polys); tick = time.perf_counter()
    if keys:
        points, allocation = shared.draw(polys, keys, 2048, np.random.default_rng(np.random.SeedSequence([249911, seed, 2048])))
    else: points = selected[None].copy(); allocation = np.empty(0, dtype=int)
    sampling = time.perf_counter()-tick; tick = time.perf_counter()
    point_prediction = project(cold.base.forward(q, selected)); prediction = project(shared.geometry.make_predict(points)(q))
    reading = time.perf_counter()-tick
    arrays = dict(points=points, allocation=allocation, prediction=prediction,
                  point_prediction=point_prediction, selected_b=selected, best_bank=best)
    if observer is not None:
        arrays.update(trigger_b=selected_state['b'], trigger_h=selected_state['h'], trigger_u=selected_state['u'],
                      trigger_location=np.array(selected_state['location']))
    if credit is not None: arrays['trigger_credit'] = credit
    metadata = dict(solver=solver, steps=steps, channel=channel, watch_only=watch, prior_starts=17,
        atomic_trial_points=atomic_trials, restarts=len(b), continuation_restart_sweeps=629*steps,
        preparation_seconds=preparation, atomic_seconds=atomic, continuation_collection_seconds=continuation,
        credit_search_seconds=credit_search, original_geometry_seconds=geometry_seconds,
        extra_geometry_seconds=extra_geometry, sampling_seconds=sampling, read_seconds=reading,
        charged_complete_seconds=time.perf_counter()-begin, visited_modes=sorted(seen), feasible_modes=feasible,
        original_positive_modes=original_positive, positive_modes=keys,
        new_positive_modes=sorted(set(keys)-set(original_positive)),
        geometry_repairs=repairs, new_mode_classifications_detail=classifications, original_pool_cache_hits=hits,
        selected_state={k: val for k, val in selected_state.items() if k not in ['b', 'h', 'u']} if observer is not None else None,
        exact_trigger_forward_calls=observer.calls if observer is not None else 0,
        observed_states=observer.index if observer is not None else 0,
        trigger_copies=observer.copies if observer is not None else 0,
        trigger_seconds=observer.seconds if observer is not None else 0., proposal=proposal,
        bp_gradient_check_gap=bp_gap, trajectory_state_sha256=fingerprints, trajectory_hash_seconds=digest_seconds,
        solver_numeric_bytes_subtotal=solver_bytes,
        representative_numeric_bytes_subtotal=sum(a.nbytes for a in seen.values()),
        geometry_numeric_bytes_subtotal=sum(a.nbytes for p in polys.values() for a in p.values() if isinstance(a, np.ndarray)),
        returned_array_bytes=sum(a.nbytes for a in arrays.values()),
        memory_scope='Named-array counts, not process peak; retained lists, interval automata, geometry and temporaries also cost memory',
        no_global_bp_guard_enabled=solver != 'adam' and channel != 'bp', branch_selector_no_global_solver=True,
        geometry_is_global_lp_not_local_pc=True, query_targets_accessed=False, uses_complete_posterior_reference=False,
        execution_failed=False, resources_matched=False, independent_task_gain_established=False)
    return arrays, metadata
