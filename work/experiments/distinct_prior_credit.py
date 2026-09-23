"""275: one selected action at each of33 distinct prior points, protected A anchor."""
from collections import Counter
import time
import numpy as np
import cold_stagnation_switch as cold
import local_dual_jump_short as jump
import minimum_sufficient_dual as minimum
import local_dual_jump_transfer as transfer
import shared_mode_readout as shared
from audit_local_dual_jump_modes import modes

PRIMARY='distinct_DA_odd32_A64'
CONFIGS=[dict(name=PRIMARY,pattern='oddD'),dict(name='distinct_DA_even32_A64',pattern='evenD'),
         dict(name='distinct_RA_odd32_A64',pattern='oddR'),dict(name='distinct_BA_odd32_A64',pattern='oddB'),
         dict(name='distinct_D32_A64',pattern='allD')]


def assigned_actions(pattern):
    assert pattern in ['oddD','evenD','oddR','oddB','allD','allA']
    actions=['A']*33
    for r in range(1,33):
        if pattern=='allD':actions[r]='D'
        elif pattern=='evenD' and r%2==0:actions[r]='D'
        elif pattern.startswith('odd') and r%2:actions[r]=pattern[-1]
    return actions


def make_state(b,h,u,best,x,v):
    state=cold.Local(b,x,v,'alm');state.h=h.copy();state.u=u.copy();state.best=best.copy()
    cold.frozen.retain(state.best,b,x,v,np.zeros(4));state.errors,state.moves=cold.base.score(state.best,x,v,np.zeros(4))
    return state


def fit(cfg,x,v,q,seed,trace=False):
    assert cold.base.BOUND==.12
    begin=time.perf_counter();n=33
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(n-1,4))]
    prep=cold.Local(starts,x,v,'nodual')
    for _ in range(16):prep.step()
    preparation=time.perf_counter()-begin;start=time.perf_counter();events=[];counts=Counter()
    for r in range(n):
        record=jump.scan(x,prep.b[r],prep.h[:,r]);first=minimum.minimum_event(x,prep.b[r],prep.h[:,r],record['selected']);events.append(first['selected'])
        counts['scanned_states']+=1;counts['first_crossing_events']+=int(first['selected'] is not None);counts['first_crossing_fallbacks']+=first['fallbacks']
        for key,value in record['counts'].items():counts[key]+=value
    search=time.perf_counter()-start;start=time.perf_counter();origins=list(range(33));actions=assigned_actions(cfg['pattern'])
    assert len(origins)==len(actions)==33
    bb=[];hh=[];uu=[];best=[];seen={}
    def collect(bank):
        for key,b in modes(x,bank).items():
            if key not in seen:seen[key]=b.copy()
    for r,action in zip(origins,actions):
        method={'D':'dual_jump','R':'dual_jump','A':'activity_only','B':'bias_only'}[action]
        a,meta=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,events[r],method)
        assert len(a['trial_b'])==1;collect(a['trial_b']);bb.append(a['b']);hh.append(a['h']);uu.append(a['u']);best.append(a['best'])
    b=np.array(bb);h=np.stack(hh,axis=1);u=np.stack(uu,axis=1);best=np.array(best)
    arrays=dict(initial_b=b.copy(),initial_h=h.copy(),initial_u=u.copy(),initial_best=best.copy(),origins=np.array(origins),assigned_actions=np.array(actions))
    effective=u.copy();effective[:,np.array(actions)=='R']=0.;arrays['effective_initial_u']=effective.copy()
    atomic=time.perf_counter()-start;start=time.perf_counter();state=make_state(b,h,effective,best,x,v);collect(state.b)
    history={k:[value] for k,value in state.arrays().items()} if trace else None
    for _ in range(32):
        state.step();collect(state.b)
        if trace:
            for k,value in state.arrays().items():history[k].append(value)
    bank=state.best.copy();solver_bytes=state.numeric_state_bytes()
    # Only the anchor gets32 more steps. Other32 states are not advanced.
    anchor=make_state(state.b[:1],state.h[:,:1],state.u[:,:1],state.best[:1],x,v)
    solver_bytes_temporary=solver_bytes+anchor.numeric_state_bytes()
    del state
    anchor_history={k:[value] for k,value in anchor.arrays().items()} if trace else None
    for _ in range(32):
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
        atomic_calls=33,prefix_restart_sweeps=33*32,anchor_extra_sweeps=32,total_restart_sweeps=1088,restarts=33,actions=actions,origins=origins,
        visited_modes=sorted(seen),feasible_modes=feasible,positive_modes=keys,lp_calls=len(seen),fallback=not bool(keys),
        solver_numeric_bytes_subtotal=solver_bytes_temporary,solver_memory_scope='33-state prefix plus transient1-state anchor handoff; excludes temporaries, returned arrays and trace',
        retained_predictor_bytes=points.nbytes,representative_numeric_bytes_subtotal=sum(b.nbytes for b in seen.values()),
        geometry_numeric_bytes_subtotal=sum(a.nbytes for poly in polys.values() for a in poly.values() if isinstance(a,np.ndarray)),trace_enabled=trace)
    return arrays,metadata
