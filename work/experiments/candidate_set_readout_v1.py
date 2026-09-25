"""382 prototype common readout from actual candidate sets; no hidden recovery.

Not a local-only algorithm: exact feasibility proposals and numeric volumes use
the existing global LP geometry. No cached modes or query targets are accepted.
Search cost is excluded here and must be separately charged by the caller.
"""
from collections import Counter
from fractions import Fraction
import time
import numpy as np
import online_credit_branch_search_v1 as local
import shared_mode_readout as shared
from verify_known_range_projection_v1 import project


def serializable(value):
    if isinstance(value, Fraction): return str(value)
    if isinstance(value, np.ndarray): return serializable(value.tolist())
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {k: serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [serializable(v) for v in value]
    return value


def fit(x, v, q, regions, *, seed, search_completed, particles=2048):
    assert x.ndim == v.ndim == q.ndim == 1 and x.shape == v.shape
    assert regions.shape[1:] == (4, len(x)) and regions.dtype == np.uint8
    assert type(search_completed) is bool and type(particles) is int and particles > 0
    assert np.all(np.isfinite(x)) and np.all(np.isfinite(v)) and np.all(np.isfinite(q))
    begin = time.perf_counter(); keys = sorted(r.tobytes().hex() for r in regions)
    assert len(keys) == len(set(keys))
    polys = {}; notes = {}; classifications = Counter()
    with local.no_bp(True):
        for key in keys:
            poly, note = local.explicit_geometry(x, v, key)
            notes[key] = note; classifications[note['classification']] += 1
            if poly is not None:
                assert note['positive_volume_certified'] and note['numerical_volume_available']
                polys[key] = poly
        geometry_seconds = time.perf_counter()-begin
        accepted = sorted(polys); unresolved = [k for k in keys if notes[k]['classification'] == 'unresolved'
            or (notes[k]['classification'] == 'positive_volume' and k not in polys)]
        tick = time.perf_counter()
        if accepted:
            points, allocation = shared.draw(polys, accepted, particles,
                np.random.default_rng(np.random.SeedSequence([249911, seed, particles])))
        else:
            points = np.empty((0, 4)); allocation = np.empty(0, int)
        sampling_seconds = time.perf_counter()-tick; tick = time.perf_counter()
        prediction = project(shared.geometry.make_predict(points)(q)) if len(points) else np.empty(0)
        reading_seconds = time.perf_counter()-tick
    arrays = dict(x=x.copy(), v=v.copy(), q=q.copy(), points=points, allocation=allocation,
        prediction=prediction, search_regions=regions.copy(), positive_volumes=np.array([polys[k]['volume'] for k in accepted]))
    meta = dict(readout_available=bool(accepted), search_completed=search_completed, positive_modes=accepted,
        unresolved_modes=unresolved, classification_counts=dict(classifications), classified_modes=keys,
        classifications_detail=serializable(notes), geometry_seconds=geometry_seconds,
        sampling_seconds=sampling_seconds, reading_seconds=reading_seconds,
        search_seconds_excluded=True,
        full_candidate_set_resolved=search_completed and not unresolved,
        conditional_on_output_union=not search_completed or bool(unresolved),
        global_geometry_lp_used=any(note['lp_calls'] > 0 for note in notes.values()),
        geometry_component_requires_global_lp=True, global_bp_used=False, query_targets_accessed=False,
        mother_trajectory_called=False, cross_method_cache_used=False, hidden_recovery_used=False,
        exact_bayesian_posterior_claim=False, particles=particles,
        geometry_numeric_bytes_subtotal=sum(a.nbytes for p in polys.values() for a in p.values() if isinstance(a, np.ndarray)),
        returned_array_bytes=sum(a.nbytes for a in arrays.values()),
        memory_scope='Named retained geometry and returned arrays; not RSS peak or all transient state')
    meta['charged_readout_seconds'] = time.perf_counter()-begin
    return arrays, meta
