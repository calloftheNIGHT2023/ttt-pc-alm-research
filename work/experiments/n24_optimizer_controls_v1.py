"""386 same-prior n24 optimizer adapter, retaining point AND union readouts.

All controls target support band feasibility, with the same support-only
best-point rule. Their finite-step surrogate objectives are not identical.
No statement about test risk or official TTT is made by this adapter.
"""
from contextlib import contextmanager
import time
import numpy as np
import cold_stagnation_switch as old
import candidate_set_readout_v1 as reader


@contextmanager
def prior_bounds():
    # Frozen legacy modules default to a wider parameter box. Change only the
    # scoped runtime scalar, including the actual bias-solve function globals.
    scopes = [vars(old.base), vars(old.bp.base), old.base.bias_solve.__globals__]
    unique = {id(scope):scope for scope in scopes}.values()
    saved = [(scope, scope['BOUND']) for scope in unique]
    for scope, _ in saved: scope['BOUND'] = .12
    try: yield
    finally:
        for scope, value in saved: scope['BOUND'] = value


def starts(restarts):
    assert isinstance(restarts, int) and restarts >= 1
    return np.r_[np.zeros((1, 4)), np.random.default_rng(731).uniform(-.12, .12, (restarts-1, 4))]


def fit(x, v, q, *, seed, family, steps, restarts=64, readout='point'):
    begin = time.perf_counter(); initial = starts(restarts)
    assert family in ['adam', 'gauss_newton', 'pc', 'alm', 'nodual']
    assert readout in ['point', 'posterior_union'] and steps > 0
    tick = time.perf_counter()
    with prior_bounds():
        if family in ['adam', 'gauss_newton']:
            bank, _, discovery = old.run_bp(initial, x, v, family, steps, trace=False)
        else:
            with reader.local.no_bp(True):
                bank, _, discovery = old.run_local(initial, x, v, family, steps, trace=False)
    discovery_seconds = time.perf_counter()-tick
    assert np.isfinite(bank).all() and np.max(abs(bank)) <= .12+1e-14
    index, point = old.select(bank, x, v)
    prediction = np.clip(old.base.forward(q, point), 0., 1.)
    arrays = dict(x=x.copy(), v=v.copy(), q=q.copy(), starts=initial, best_bank=bank,
        selected_point=point, point_prediction=prediction.copy())
    readout_meta = None; readout_seconds = 0.
    if readout == 'posterior_union':
        tick = time.perf_counter()
        keys = sorted({old.base.pattern(x, b).astype(np.uint8).tobytes() for b in bank})
        regions = np.array([np.frombuffer(k, np.uint8).reshape(4, len(x)) for k in keys], np.uint8)
        ra, readout_meta = reader.fit(x, v, q, regions, seed=seed, search_completed=False)
        arrays.update({'readout_'+k:value for k, value in ra.items()})
        if readout_meta['readout_available']: prediction = ra['prediction']
        readout_seconds = time.perf_counter()-tick
    arrays['prediction'] = prediction
    metadata = dict(family=family, steps=steps, restarts=restarts, readout=readout,
        parameter_prior_bounds=[-.12, .12], support_band=.001,
        selected_index=index, selected_support_max_error=float(np.max(abs(old.base.forward(x, point)-v))),
        discovery=discovery, discovery_seconds=discovery_seconds, readout_metadata=readout_meta,
        readout_seconds=readout_seconds, query_targets_accessed=False,
        global_bp_used=family in ['adam', 'gauss_newton'],
        shared_readout_available_to_optimizers=True, complete_posterior_claim=False,
        objective_scope='Common band-feasibility target and support best-point selection; different finite-step surrogates',
        returned_array_bytes=sum(a.nbytes for a in arrays.values()))
    metadata['total_seconds'] = time.perf_counter()-begin
    return arrays, metadata
