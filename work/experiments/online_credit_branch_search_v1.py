"""326 live local-prefix state selection, explicit branch search and readout.

No archived trajectory, reference posterior or query answer is an input. The
original probe33 trajectory is preserved. Global LP geometry is explicit and
charged; only the declared BP-credit control may call the global derivative.
"""
from collections import Counter
from contextlib import contextmanager
import time
import numpy as np
import scipy.optimize as opt
import certificate_activity_attribution as original
import cold_stagnation_switch as cold
import local_dual_jump_transfer as transfer
import shared_mode_readout as shared
import conditioned_mode_geometry as conditioned
import complete_credit_mode_geometry_v1 as classifier
import branch_image_chain_v1 as chain
import batched_bp_discovery as bp
from audit_local_dual_jump_modes import modes
from support_consistency_trigger_v1 import exact_forward,support_loss,uniform_index
from run_factorized_dual_branch_search_v1 import signals,bp_control

POLICIES=['first_fit','uniform_state']
CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


class Selector:
    def __init__(self,policy,seed,x,v):
        assert policy in POLICIES
        self.policy=policy;self.x=x;self.v=v;self.target=uniform_index(seed) if policy=='uniform_state' else None
        self.index=0;self.calls=0;self.copies=0;self.best=None;self.done=False;self.selected=None;self.seconds=0.

    def observe(self,state,phase,t):
        tick=time.perf_counter()
        for r in range(len(state.b)):
            i=self.index;self.index+=1
            if self.done:continue
            if self.policy=='uniform_state':take=(i==self.target);loss=None
            else:
                loss=support_loss(state.b[r],self.x,self.v);self.calls+=1
                take=self.best is None or loss<self.best
            if take:
                self.selected=dict(index=i,location=[phase,t,r],b=state.b[r].copy(),h=state.h[:,r].copy(),u=state.u[:,r].copy())
                self.best=loss;self.copies+=1
            if (self.policy=='uniform_state' and take) or loss==0:self.done=True
        self.seconds+=time.perf_counter()-tick

    def finish(self):
        assert self.index==1088 and self.selected is not None
        tick=time.perf_counter();s=self.selected
        loss=support_loss(s['b'],self.x,self.v);_,pattern=exact_forward(s['b'],self.x)
        s.update(exact_support_loss=str(loss),support_fit=loss==0,
                 original_mode=np.array(pattern,dtype=np.uint8).tobytes().hex())
        self.seconds+=time.perf_counter()-tick
        return s


@contextmanager
def no_bp(enabled):
    saved=[]
    def forbidden(*args,**kwargs):raise AssertionError('Global BP called by local-only candidate')
    try:
        if enabled:
            for obj,name in [(bp,'evaluate'),(bp,'refine'),(cold,'run_bp')]:
                saved.append((obj,name,getattr(obj,name)));setattr(obj,name,forbidden)
        yield
    finally:
        for obj,name,value in saved:setattr(obj,name,value)


def explicit_geometry(x,v,key):
    """Return the actual polytope already computed by the frozen classifier.

    A transparent observer wraps the existing geometry context, including its
    declared conditioning repair. No second LP or surrogate representative is
    used to recover geometry, and all patches are restored before returning.
    """
    old_scope=conditioned.geometry_scope;captured=[]
    @contextmanager
    def observe_scope(log):
        with old_scope(log):
            previous=shared.geometry.polytope
            def capture(g,rhs):
                poly,note=previous(g,rhs);captured.append(poly);return poly,note
            shared.geometry.polytope=capture
            try:yield
            finally:shared.geometry.polytope=previous
    conditioned.geometry_scope=observe_scope
    try:result=classifier.classify_mode(x,v,key)
    finally:conditioned.geometry_scope=old_scope
    poly=None
    if result['numerical_volume_available']:
        assert len(captured)==1 and captured[0] is not None
        poly=captured[0];assert float(poly['volume'])==result['volume']
    return poly,result


def fit(x,v,q,seed,*,policy='first_fit',channel='dual',trace=False):
    assert policy in POLICIES+['none'] and channel in CHANNELS
    with no_bp(channel!='bp'):
        return _fit(x,v,q,seed,policy=policy,channel=channel,trace=trace)


def _fit(x,v,q,seed,*,policy,channel,trace):
    assert cold.base.BOUND==.12 and len(x)==len(v)==4
    begin=time.perf_counter();n=33;selector=Selector(policy,seed,x,v) if policy!='none' else None
    starts=np.r_[np.zeros((1,4)),np.random.default_rng(731).uniform(-.12,.12,(n-1,4))]
    prep=cold.Local(starts,x,v,'nodual')
    for _ in range(16):prep.step()
    preparation=time.perf_counter()-begin;tick=time.perf_counter();seen={};trials=[];trial_origins=[];bb=[];hh=[];uu=[];best=[];atomic_work=Counter();atomic_notes=[]
    def collect(bank):
        for key,b in modes(x,bank).items():
            if key not in seen:seen[key]=b.copy()
    for r in range(33):
        a,meta=transfer.run(prep.b[r],prep.h[:,r],prep.best[r],x,v,None,'branch_probe')
        collect(a['trial_b']);trials.extend(a['trial_b']);trial_origins.extend([r]*len(a['trial_b']))
        bb.append(a['b']);hh.append(a['h']);uu.append(a['u']);best.append(a['best']);atomic_notes.append(meta)
        for key in ['activity_blocks','bias_blocks','dual_updates','activity_branch_proposals','dual_writes']:atomic_work[key]+=meta.get(key,0)
    b=np.array(bb);h=np.stack(hh,axis=1);u=np.stack(uu,axis=1);best=np.array(best)
    arrays=dict(initial_b=b.copy(),initial_h=h.copy(),initial_u=u.copy(),initial_best=best.copy(),
                origins=np.arange(33),assigned_actions=np.array(['Q']*33),atomic_trial_b=np.array(trials),atomic_trial_origins=np.array(trial_origins),effective_initial_u=u.copy())
    atomic=time.perf_counter()-tick;tick=time.perf_counter()
    state=original.make_state(b,h,u,best,x,v,'alm');collect(state.b)
    history={k:[value] for k,value in state.arrays().items()} if trace else None
    for t in range(1,33):
        state.step();collect(state.b)
        if selector is not None:selector.observe(state,0,t)
        if trace:
            for k,value in state.arrays().items():history[k].append(value)
    bank=state.best.copy();solver_bytes=state.numeric_state_bytes()
    anchor=original.make_state(state.b[:1],state.h[:,:1],state.u[:,:1],state.best[:1],x,v,'alm')
    solver_bytes+=anchor.numeric_state_bytes();del state
    ahistory={k:[value] for k,value in anchor.arrays().items()} if trace else None
    for t in range(1,33):
        anchor.step();collect(anchor.b)
        if selector is not None:selector.observe(anchor,1,t)
        if trace:
            for k,value in anchor.arrays().items():ahistory[k].append(value)
    bank[0]=anchor.best[0];_,selected=cold.select(bank,x,v)
    continuation=time.perf_counter()-tick;tick=time.perf_counter();selected_state=None;credit=None;proposal=None;credit_seconds=0.;bp_gap=None
    if selector is not None:
        selected_state=selector.finish();s=selected_state
        tick_credit=time.perf_counter()
        if channel=='bp':credit,bp_gap=bp_control(s['b'],x,v)
        else:credit=signals(s['b'],s['h'],s['u'],x,seed,s['location'])[channel]
        credit_seconds=time.perf_counter()-tick_credit
        pattern=np.array(list(bytes.fromhex(s['original_mode']))).reshape(4,4)
        saved=[]
        def forbid(*args,**kwargs):raise AssertionError('Global solver or BP entered branch selector')
        try:
            for obj,name in [(bp,'evaluate'),(bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
                saved.append((obj,name,getattr(obj,name)));setattr(obj,name,forbid)
            proposal=chain.propose(x,v,credit,pattern,k=8)
        finally:
            for obj,name,func in saved:setattr(obj,name,func)
        arrays.update(trigger_b=s['b'],trigger_h=s['h'],trigger_u=s['u'],trigger_credit=credit,trigger_location=np.array(s['location']))
    search=time.perf_counter()-tick;tick=time.perf_counter();polys={};feasible=[];original_repairs=[]
    with conditioned.geometry_scope(original_repairs):
        for key,representative in sorted(seen.items()):
            note=cold.base.branch_feasibility(x,v,representative)
            if note['current_branch_feasible']:
                feasible.append(key);poly,proof=shared.build(x,v,representative)
                if poly is not None:polys[key]=poly
    original_positive=sorted(polys);original_geometry_seconds=time.perf_counter()-tick;tick=time.perf_counter();classifications={};cache_hits=0
    if proposal is not None:
        for key in sorted({p['mode'] for p in proposal['proposals']}):
            if key in polys:cache_hits+=1;continue
            poly,note=explicit_geometry(x,v,key);classifications[key]=note
            if poly is not None:polys[key]=poly
    extra_geometry_seconds=time.perf_counter()-tick;keys=sorted(polys);tick=time.perf_counter()
    if keys:points,allocation=shared.draw(polys,keys,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
    else:points=selected[None].copy();allocation=np.empty(0,dtype=int)
    sampling=time.perf_counter()-tick;tick=time.perf_counter()
    prediction=shared.geometry.make_predict(points)(q);point_prediction=cold.base.forward(q,selected)
    reading=time.perf_counter()-tick;arrays.update(points=points,allocation=allocation,prediction=prediction,point_prediction=point_prediction,selected_b=selected,best_bank=bank)
    if trace:
        arrays.update({'prefix_'+k:np.array(val) for k,val in history.items()});arrays.update({'anchor_'+k:np.array(val) for k,val in ahistory.items()})
    numeric=lambda obj:sum(v.nbytes for v in obj.values() if isinstance(v,np.ndarray))
    meta=dict(policy=policy,channel=channel,preparation_seconds=preparation,atomic_seconds=atomic,
              continuation_collection_seconds=continuation,trigger_selection_seconds=selector.seconds if selector else 0.,
              trigger_seconds_in_continuation=True,credit_seconds=credit_seconds,search_seconds=search,
              original_geometry_seconds=original_geometry_seconds,extra_geometry_seconds=extra_geometry_seconds,
              sampling_seconds=sampling,read_seconds=reading,total_seconds=time.perf_counter()-begin,
              prior_starts=33,preparation_restart_sweeps=528,atomic_calls=33,atomic_trial_points=len(trials),atomic_work=dict(atomic_work),atomic_notes=atomic_notes,
              prefix_restart_sweeps=1056,anchor_extra_sweeps=32,total_restart_sweeps=1088,
              original_visited_modes=sorted(seen),original_feasible_modes=feasible,original_positive_modes=original_positive,
              positive_modes=keys,new_positive_modes=sorted(set(keys)-set(original_positive)),original_geometry_repairs=original_repairs,
              original_feasibility_calls=len(seen),new_mode_classifications=len(classifications),new_mode_classifier_lp_calls=sum(n['lp_calls'] for n in classifications.values()),
              original_poly_cache_hits=cache_hits,new_mode_classifications_detail=classifications,
              selected_state={k:val for k,val in selected_state.items() if k not in ['b','h','u']} if selected_state else None,
              selected_state_copies=selector.copies if selector else 0,exact_trigger_forward_calls=selector.calls if selector else 0,
              trigger_diagnostic_forward_calls=1 if selector else 0,global_bp_credit_check_gap=bp_gap,proposal=proposal,
              solver_numeric_bytes_subtotal=solver_bytes,cached_trigger_numeric_bytes=numeric(selected_state) if selected_state else 0,
              credit_numeric_bytes=credit.nbytes if credit is not None else 0,retained_predictor_bytes=points.nbytes,
              original_representative_numeric_bytes=sum(b.nbytes for b in seen.values()),geometry_numeric_bytes=sum(numeric(p) for p in polys.values()),
              returned_array_bytes=numeric(arrays),trace_enabled=trace,
              memory_scope='Named-array subtotals, NOT total process peak; exact Fractions, Python objects and transient DP/LP work need isolated profiling.',
              no_global_bp_guard_enabled=channel!='bp',uses_complete_posterior_reference=False,query_targets_accessed=False,
              geometry_is_global_lp_not_local_pc=True,resources_matched=False,independent_task_gain_established=False)
    return arrays,meta
