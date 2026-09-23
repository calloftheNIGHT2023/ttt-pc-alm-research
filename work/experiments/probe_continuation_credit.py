"""283: identical zero-dual trial pool; isolate subsequent local credit."""
from collections import Counter
import time
import numpy as np
import certificate_activity_attribution as old
cold=old.cold
transfer=old.transfer
shared=old.shared
modes=old.modes

PRIMARY='credit_control_probe33'
GOLD=next(dict(c) for c in old.CONFIGS if c['name']==PRIMARY)
CONFIGS=[
    dict(name='probe_archive_only',atomic='branch_probe',solver='archive',certificate=False,action='Q'),
    dict(name='probe_then_nodual33',atomic='branch_probe',solver='nodual',certificate=False,action='Q'),
    dict(name='probe_then_pc33',atomic='branch_probe',solver='pc',certificate=False,action='Q'),
    dict(name='probe_then_adam240_33',atomic='branch_probe',solver='adam',certificate=False,action='Q',steps=240)]


def fit(cfg,x,v,q,seed,trace=False):
    if cfg['solver'] in ['alm','pc','nodual']:
        return old.fit(cfg,x,v,q,seed,trace=trace)
    assert cold.base.BOUND==.12 and cfg['solver'] in ['archive','adam']
    begin=time.perf_counter();starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(32,4))]
    prep=cold.Local(starts,x,v,'nodual')
    for _ in range(16):prep.step()
    preparation=time.perf_counter()-begin;start=time.perf_counter();seen={};bb=[];hh=[];uu=[];best=[];trials=[];origins=[];work=Counter();notes=[]
    def collect(bank):
        for key,b in modes(x,bank).items():
            if key not in seen:seen[key]=b.copy()
    for r in range(33):
        a,m=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,None,'branch_probe')
        collect(a['trial_b']);trials.extend(a['trial_b']);origins.extend([r]*len(a['trial_b']));bb.append(a['b']);hh.append(a['h']);uu.append(a['u']);best.append(a['best']);notes.append(m)
        for key in ['activity_blocks','bias_blocks','dual_updates','activity_branch_proposals','dual_writes']:work[key]+=m.get(key,0)
    b=np.array(bb);h=np.stack(hh,axis=1);u=np.stack(uu,axis=1);bank=np.array(best)
    arrays=dict(initial_b=b.copy(),initial_h=h.copy(),initial_u=u.copy(),initial_best=bank.copy(),effective_initial_u=u.copy(),origins=np.arange(33),assigned_actions=np.array(['Q']*33),atomic_trial_b=np.array(trials),atomic_trial_origins=np.array(origins))
    assert not u.any();atomic=time.perf_counter()-start;start=time.perf_counter();collect(b);cold.frozen.retain(bank,b,x,v,np.zeros(4));bpmeta={};history=[];roles=[]
    # State gathering is an observer of the frozen original BP implementation.
    # Its evaluations are necessary discovery work even when full trace is off.
    if cfg['solver']=='adam':
        original=cold.bp.evaluate
        def record(bs,*args,**kwargs):
            collect(bs)
            if trace:history.append(bs.copy());roles.append(kwargs.get('with_jacobian',args[2] if len(args)>2 else True))
            return original(bs,*args,**kwargs)
        cold.bp.evaluate=record
        try:bpbest,bpmeta=cold.bp.refine(b,x,v,np.zeros(4),solver='adam',steps=cfg['steps'],lr=.003)
        finally:cold.bp.evaluate=original
        cold.frozen.retain(bank,bpbest,x,v,np.zeros(4))
        if trace:arrays.update(bp_b=np.array(history),bp_roles=np.array(roles),bp_best=bpbest)
    _,selected=cold.select(bank,x,v);continuation=time.perf_counter()-start;start=time.perf_counter();polys={};feasible=[]
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
    metadata=dict(preparation_seconds=preparation,search_seconds=0.,atomic_seconds=atomic,continuation_collection_seconds=continuation,geometry_seconds=geometry,
        sampling_seconds=sampling,read_seconds=reading,total_seconds=time.perf_counter()-begin,scans={'scanned_states':0},prior_starts=33,preparation_restart_sweeps=528,
        atomic_calls=33,atomic_trial_points=len(trials),atomic_trial_numeric_bytes=np.array(trials).nbytes,atomic_work=dict(work),atomic_notes=notes,
        continuation_solver=cfg['solver'],certificate_scan_enabled=False,prefix_restart_sweeps=33*cfg.get('steps',0),anchor_extra_sweeps=0,total_restart_sweeps=33*cfg.get('steps',0),
        restarts=33,actions=['Q']*33,origins=list(range(33)),visited_modes=sorted(seen),feasible_modes=feasible,positive_modes=keys,lp_calls=len(seen),fallback=not bool(keys),
        solver_numeric_bytes_subtotal=bpmeta.get('major_arrays_bytes_subtotal',bank.nbytes),solver_memory_scope='BP reported major arrays or archive bank only; excludes shared preparation, trial arrays, temporaries and trace',
        retained_predictor_bytes=points.nbytes,representative_numeric_bytes_subtotal=sum(b.nbytes for b in seen.values()),
        geometry_numeric_bytes_subtotal=sum(a.nbytes for poly in polys.values() for a in poly.values() if isinstance(a,np.ndarray)),trace_enabled=trace,bp_metadata=bpmeta)
    return arrays,metadata
