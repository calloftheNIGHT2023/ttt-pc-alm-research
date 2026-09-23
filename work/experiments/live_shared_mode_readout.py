"""True cold fit: stream visited modes, solve own geometry, sample, predict.

No saved candidates/geometry are accessed. Histories of h/u are not retained.
The shared observation-only positive-interior check is included in the cost.
"""
import time
import numpy as np
import cold_stagnation_switch as cold
import shared_mode_readout as shared

def fit(cfg,x,v,q,seed,repetition_seed=249911):
    begin=time.perf_counter();starts=cold.starts();seen={}
    def collect(bank):
        for b in bank:
            key=cold.base.pattern(x,b).astype(np.uint8).tobytes().hex()
            if key not in seen:seen[key]=b.copy()
    if cfg['family']=='local':
        state=cold.Local(starts,x,v,cfg['method']);collect(state.b)
        for _ in range(cfg['steps']):state.step();collect(state.b)
        best=state.best.copy();solver_bytes=state.numeric_state_bytes();trigger=state.first_trigger.tolist()
    else:
        saved=cold.bp.evaluate
        def record(b,x,v,with_jacobian=True):collect(b);return saved(b,x,v,with_jacobian)
        try:
            cold.bp.evaluate=record;best,mm=cold.bp.refine(starts,x,v,np.zeros(4),solver=cfg['method'],steps=cfg['steps'])
        finally:cold.bp.evaluate=saved
        solver_bytes=mm['major_arrays_bytes_subtotal'];trigger=None
    selected,b=cold.select(best,x,v);discovery=time.perf_counter()-begin;start=time.perf_counter();polys={};notes={};lp_calls=0;feasible=[]
    for key,representative in sorted(seen.items()):
        note=cold.base.branch_feasibility(x,v,representative);lp_calls+=1
        if note['current_branch_feasible']:
            feasible.append(key);poly,proof=shared.build(x,v,representative);notes[key]=proof
            if poly is not None:polys[key]=poly
    geometry=time.perf_counter()-start;keys=sorted(polys);start=time.perf_counter()
    if keys:points,allocation=shared.draw(polys,keys,2048,np.random.default_rng(np.random.SeedSequence([repetition_seed,seed,2048])))
    else:points=b[None].copy();allocation=np.empty(0,dtype=int)
    sampling=time.perf_counter()-start;start=time.perf_counter();prediction=shared.geometry.make_predict(points)(q);read=time.perf_counter()-start
    arrays=dict(points=points,allocation=allocation,prediction=prediction,selected_b=b,best_bank=best)
    meta=dict(discovery_seconds=discovery,geometry_seconds=geometry,sampling_seconds=sampling,read_seconds=read,total_seconds=discovery+geometry+sampling+read,
        observed_modes=len(seen),feasible_modes=feasible,positive_modes=keys,lp_calls=lp_calls,positive_interior_checks=len(notes),fallback=not bool(keys),
        selected_restart=selected,first_trigger=trigger,solver_numeric_bytes_subtotal=solver_bytes,retained_predictor_bytes=points.nbytes,
        geometry_numeric_bytes_subtotal=sum(a.nbytes for poly in polys.values() for a in poly.values() if isinstance(a,np.ndarray)),
        shared_prior_rule='same known uniform prior; fresh17 starts, no saved geometric cache')
    return arrays,meta
