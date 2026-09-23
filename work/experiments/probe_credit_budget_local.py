"""285 local budget adapter: only prefix/tail horizons differ from frozen281."""
from collections import Counter
import time
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump_short as jump
import minimum_sufficient_dual as minimum
import local_dual_jump_transfer as transfer
import shared_mode_readout as shared
from audit_local_dual_jump_modes import modes

def make_state(b,h,u,best,x,v,solver):
    state=cold.Local(b,x,v,solver);state.h=h.copy();state.u=u.copy();state.best=best.copy()
    cold.frozen.retain(state.best,b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
    return state


def fit(cfg,x,v,q,seed,trace=False):
    assert cold.base.BOUND==.12
    prefix=int(cfg.get('prefix',32));extra=int(cfg.get('extra',32));assert prefix>=0 and extra>=0
    begin=time.perf_counter();n=33
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(n-1,4))]
    prep=cold.Local(starts,x,v,'nodual')
    for _ in range(16):prep.step()
    preparation=time.perf_counter()-begin;start=time.perf_counter();events=[];counts=Counter()
    counts['scanned_states']=0
    for r in range(n):
        if cfg['certificate']:
            record=jump.scan(x,prep.b[r],prep.h[:,r]);first=minimum.minimum_event(x,prep.b[r],prep.h[:,r],record['selected']);events.append(first['selected'])
            counts['scanned_states']+=1;counts['first_crossing_events']+=int(first['selected'] is not None);counts['first_crossing_fallbacks']+=first['fallbacks']
            for key,value in record['counts'].items():counts[key]+=value
        else:events.append(None)
    search=time.perf_counter()-start;start=time.perf_counter();origins=list(range(33));actions=[cfg['action']]*33
    assert len(origins)==len(actions)==33
    bb=[];hh=[];uu=[];best=[];seen={};trials=[];trial_origins=[];atomic_work=Counter();atomic_notes=[]
    def collect(bank):
        for key,b in modes(x,bank).items():
            if key not in seen:seen[key]=b.copy()
    for r in origins:
        a,meta=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,events[r],cfg['atomic'])
        collect(a['trial_b']);trials.extend(a['trial_b']);trial_origins.extend([r]*len(a['trial_b']))
        bb.append(a['b']);hh.append(a['h']);uu.append(a['u']);best.append(a['best']);atomic_notes.append(meta)
        for key in ['activity_blocks','bias_blocks','dual_updates','activity_branch_proposals','dual_writes']:atomic_work[key]+=meta.get(key,0)
    b=np.array(bb);h=np.stack(hh,axis=1);u=np.stack(uu,axis=1);best=np.array(best)
    arrays=dict(initial_b=b.copy(),initial_h=h.copy(),initial_u=u.copy(),initial_best=best.copy(),origins=np.array(origins),assigned_actions=np.array(actions),atomic_trial_b=np.array(trials),atomic_trial_origins=np.array(trial_origins))
    effective=u.copy();arrays['effective_initial_u']=effective.copy()
    if cfg['solver'] in ['pc','nodual']:assert not effective.any()
    atomic=time.perf_counter()-start;start=time.perf_counter();state=make_state(b,h,effective,best,x,v,cfg['solver']);collect(state.b)
    history={k:[value] for k,value in state.arrays().items()} if trace else None
    for _ in range(prefix):
        state.step();collect(state.b)
        if trace:
            for k,value in state.arrays().items():history[k].append(value)
    bank=state.best.copy();solver_bytes=state.numeric_state_bytes()
    # Only the origin gets the declared extra steps; other states are not advanced.
    anchor=make_state(state.b[:1],state.h[:,:1],state.u[:,:1],state.best[:1],x,v,cfg['solver'])
    solver_bytes_temporary=solver_bytes+anchor.numeric_state_bytes()
    del state
    anchor_history={k:[value] for k,value in anchor.arrays().items()} if trace else None
    for _ in range(extra):
        anchor.step();collect(anchor.b)
        if trace:
            for k,value in anchor.arrays().items():anchor_history[k].append(value)
    bank[0]=anchor.best[0];_,selected=cold.select(bank,x,v);continuation=time.perf_counter()-start;start=time.perf_counter();polys={};feasible=[]
    for key,representative in sorted(seen.items()):
        note=cold.base.branch_feasibility(x,v,representative)
        if note['current_branch_feasible']:
            feasible.append(key);poly,proof=shared.build(x,v,representative)
            if poly is not None:polys[key]=poly
    geometry=time.perf_counter()-start;keys=sorted(polys);start=time.perf_counter()
    if keys:points,allocation=shared.draw(polys,keys,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
    else:points=selected[None].copy();allocation=np.empty(0,dtype=int)
    sampling=time.perf_counter()-start;start=time.perf_counter();prediction=shared.geometry.make_predict(points)(q);point_prediction=cold.base.forward(q,selected);reading=time.perf_counter()-start
    arrays.update(points=points,allocation=allocation,prediction=prediction,point_prediction=point_prediction,selected_b=selected,best_bank=bank)
    if trace:
        arrays.update({'prefix_'+k:np.array(value) for k,value in history.items()});arrays.update({'anchor_'+k:np.array(value) for k,value in anchor_history.items()})
    metadata=dict(preparation_seconds=preparation,search_seconds=search,atomic_seconds=atomic,continuation_collection_seconds=continuation,geometry_seconds=geometry,
        sampling_seconds=sampling,read_seconds=reading,total_seconds=time.perf_counter()-begin,scans=dict(counts),prior_starts=n,preparation_restart_sweeps=n*16,
        atomic_calls=33,atomic_trial_points=len(trials),atomic_trial_numeric_bytes=np.array(trials).nbytes,atomic_work=dict(atomic_work),atomic_notes=atomic_notes,continuation_solver=cfg['solver'],certificate_scan_enabled=cfg['certificate'],prefix_restart_sweeps=33*prefix,anchor_extra_sweeps=extra,total_restart_sweeps=33*prefix+extra,restarts=33,actions=actions,origins=origins,
        visited_modes=sorted(seen),feasible_modes=feasible,positive_modes=keys,lp_calls=len(seen),fallback=not bool(keys),
        solver_numeric_bytes_subtotal=solver_bytes_temporary,solver_memory_scope='33-state prefix plus transient1-state anchor handoff; excludes temporaries, returned arrays and trace',
        retained_predictor_bytes=points.nbytes,representative_numeric_bytes_subtotal=sum(b.nbytes for b in seen.values()),
        geometry_numeric_bytes_subtotal=sum(a.nbytes for poly in polys.values() for a in poly.values() if isinstance(a,np.ndarray)),trace_enabled=trace)
    return arrays,metadata
