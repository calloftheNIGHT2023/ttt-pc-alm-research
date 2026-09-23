"""256 fresh preparation, full certificate scan, atomic write, optimizer, readout.

No disk artifacts, saved states, geometry caches, or query answers are read.
The prefix's transient modes are not added: this matches the frozen254 pool.
"""
import time
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump as jump
import local_dual_jump_transfer as transfer
import shared_mode_readout as shared
from audit_local_dual_jump_modes import modes

def fit(cfg,x,v,q,seed,repetition_seed=249911):
    begin=time.perf_counter();state=cold.Local(cold.starts(),x,v,'nodual')
    for _ in range(16):state.step()
    preparation=time.perf_counter()-begin;start=time.perf_counter();events=[];counts=dict(scanned_states=0,tau_trials=0,accepted_trials=0)
    needs_event=cfg['initial'] in ['dual_jump','activity_only','reorder_no_dual']
    if needs_event:
        for r in range(17):
            record=jump.scan(x,state.b[r],state.h[:,r]);events.append(record['selected']);counts['scanned_states']+=1;counts['tau_trials']+=sum(len(b['trials']) for b in record['blocks']);counts['accepted_trials']+=len(record['candidates'])
        del record
    else:events=[None]*17
    search=time.perf_counter()-start;start=time.perf_counter();bb=[];hh=[];uu=[];best=[];origin=[];seen={};atomic_trials=0
    def collect(bank):
        # Actual forward patterns; retain first real parameter representative.
        for k,b in modes(x,bank).items():
            if k not in seen:seen[k]=b.copy()
    for r in range(17):
        a,meta=transfer.run(state.b[r],state.h[:,r],state.best[r],x,v,events[r],cfg['initial']);collect(a['trial_b']);atomic_trials+=len(a['trial_b'])
        if cfg.get('all_trials'):
            n=len(a['trial_b']);bb.extend(a['trial_b']);hh.extend(a['trial_h'].transpose(1,0,2));uu.extend(a['trial_u'].transpose(1,0,2));best.extend(np.tile(a['best'],(n,1)));origin.extend([r]*n)
        else:bb.append(a['b']);hh.append(a['h']);uu.append(a['u']);best.append(a['best']);origin.append(r)
    b=np.array(bb);h=np.stack(hh,axis=1);u=np.stack(uu,axis=1);best=np.array(best);atomic=time.perf_counter()-start;start=time.perf_counter()
    if cfg.get('bp'):
        saved=cold.bp.evaluate
        def record(bank,x,v,with_jacobian=True):collect(bank);return saved(bank,x,v,with_jacobian)
        try:cold.bp.evaluate=record;answer,meta=cold.bp.refine(b,x,v,np.zeros(4),solver=cfg['method'],steps=cfg['steps'])
        finally:cold.bp.evaluate=saved
        cold.frozen.retain(best,answer,x,v,np.zeros(4));solver_bytes=meta['major_arrays_bytes_subtotal']
    else:
        state=cold.Local(b,x,v,cfg['method']);state.h=h;state.u=np.zeros_like(u) if cfg.get('reset') else u;state.best=best.copy();cold.frozen.retain(state.best,b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4));collect(state.b)
        for _ in range(cfg['steps']):state.step();collect(state.b)
        best=state.best.copy();solver_bytes=state.numeric_state_bytes()
    selected,bb=cold.select(best,x,v);continuation_collection=time.perf_counter()-start;start=time.perf_counter();polys={};feasible=[];notes={}
    for key,representative in sorted(seen.items()):
        note=cold.base.branch_feasibility(x,v,representative)
        if note['current_branch_feasible']:
            feasible.append(key);poly,proof=shared.build(x,v,representative);notes[key]=proof
            if poly is not None:polys[key]=poly
    geometry=time.perf_counter()-start;keys=sorted(polys);start=time.perf_counter()
    if keys:points,allocation=shared.draw(polys,keys,2048,np.random.default_rng(np.random.SeedSequence([repetition_seed,seed,2048])))
    else:points=bb[None].copy();allocation=np.empty(0,dtype=int)
    sampling=time.perf_counter()-start;start=time.perf_counter();point_prediction=cold.base.forward(q,bb);prediction=shared.geometry.make_predict(points)(q);reading=time.perf_counter()-start
    arrays=dict(points=points,allocation=allocation,prediction=prediction,point_prediction=point_prediction,selected_b=bb,best_bank=best)
    meta=dict(preparation_seconds=preparation,search_seconds=search,atomic_seconds=atomic,continuation_collection_seconds=continuation_collection,geometry_seconds=geometry,
        sampling_seconds=sampling,read_seconds=reading,total_seconds=time.perf_counter()-begin,scans=counts,atomic_trial_points=atomic_trials,restarts=len(b),
        visited_modes=sorted(seen),feasible_modes=feasible,positive_modes=keys,lp_calls=len(seen),fallback=not bool(keys),solver_numeric_bytes_subtotal=solver_bytes,
        retained_predictor_bytes=points.nbytes,representative_numeric_bytes_subtotal=sum(b.nbytes for b in seen.values()),
        geometry_numeric_bytes_subtotal=sum(a.nbytes for poly in polys.values() for a in poly.values() if isinstance(a,np.ndarray)),
        memory_scope='named numeric subtotals only; certificates use arbitrary precision integers; trace peak measured separately')
    return arrays,meta
