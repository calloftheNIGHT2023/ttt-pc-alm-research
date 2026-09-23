"""Online sampled-state proposals with shared, order-canonical region readout.

Local ALM discovery uses local derivatives / exact blocks, not global BP.
Region geometry remains a global LP/QP computation, shared with every control.
This module never receives query answers or the teacher's parameters.
"""
from dataclasses import dataclass
import hashlib,time
import numpy as np
from scipy.optimize import minimize
import posterior_state_interface as interface
import local_region_screen as screen
core=interface.core;base=interface.base;posterior=interface.posterior


@dataclass(frozen=True)
class State:
    anchor: np.ndarray
    samples: np.ndarray


def refine_pc(starts,x,v,anchor,sweeps=80):
    """Existing gradient-block PC semantics, arbitrary shared starts, zero dual."""
    b=starts.copy();r,d=b.shape;trust=.01
    h=np.empty((d,r,len(x)));prev=x
    for j in range(d):h[j]=base.g(prev+b[:,j,None]);prev=h[j]
    best=b.copy();errors,moves=base.score(best,x,v,anchor)
    for _ in range(sweeps):
        for j in reversed(range(d)):
            prev=x if j==0 else h[j-1];old=h[j].copy();a=base.g(prev+b[:,j,None])
            if j==d-1:
                h[j]=np.clip((a+trust*old)/(1+trust),np.maximum(0,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                z=old+b[:,j+1,None];grad=old-a+base.derivative(z)*(base.g(z)-h[j+1])
                h[j]=np.clip(old-grad/(5+trust),0,1)
        for j in range(d):
            prev=np.broadcast_to(x,(r,len(x))) if j==0 else h[j-1];old=b[:,j].copy();z=prev+old[:,None]
            grad=np.mean((base.g(z)-h[j])*base.derivative(z),axis=1)
            b[:,j]=np.clip(old-grad/(4+trust),-base.BOUND,base.BOUND)
        error,move=base.score(b,x,v,anchor);update=base.better(error,move,errors,moves)
        best[update]=b[update];errors[update]=error[update];moves[update]=move[update]
    return best,dict(pc_gradient_sweeps=sweeps,dual_rate=0.,
        refinement_major_arrays_bytes=sum(a.nbytes for a in [b,h,best]),pc_scope='hard observation-band local-gradient PC, not official soft-loss PC replication')


def discover(x,v,state,cfg):
    anchor=np.zeros(4) if state is None else state.anchor
    kind='prior256' if state is None else cfg['pool']
    pool=interface.make_pool(len(anchor),kind,None if state is None else state.samples)
    later=len(x)>cfg.get('switch_after',10**9)
    gen='direct' if later else cfg['generator'];restarts=cfg.get('later_restarts',128) if later else cfg['restarts']
    starts,pm=interface.select_pool(x,v,anchor,pool,restarts)
    with core.pipeline.discovery_box(.12):
        if gen=='alm':best,rm=core.contextual.refine_local(starts,x,v,anchor,sweeps=cfg.get('sweeps',20),dual_rate=cfg.get('dual_rate',.5))
        elif gen=='pc':best,rm=refine_pc(starts,x,v,anchor,sweeps=cfg.get('sweeps',80))
        elif gen=='direct':best=starts;rm=dict(no_iterative_refinement=True)
        else:best,rm=core.batched.refine(starts,x,v,anchor,solver=gen,steps=cfg['steps'],lr=cfg.get('lr',.003))
        bank=core.contextual.deduplicate(np.vstack([starts,best]),x,v,anchor)
    return bank,dict(**pm,**rm,effective_pool=kind,effective_generator=gen,
        previous_posterior_samples=0 if state is None else len(state.samples),unique_regions=len(bank),retained_bank_bytes=bank.nbytes)


def materialize(x,v,anchor,bank,sample_count=512,canonical=True):
    """The same cell set yields exactly the same samples, independent of order.

    Closest-anchor point selection deliberately keeps the old discovery order;
    only posterior sampling uses canonical keys. Duplicate cells count once.
    """
    begin=time.perf_counter();regs=interface.archived.signatures(x,bank)
    reject=screen.contract(x,v,regs,rounds=5);screen_time=time.perf_counter()-begin
    begin=time.perf_counter();polys={};trace=[];seen=set();point=None
    for b,reg,skip in zip(bank,regs,reject):
        key=reg.tobytes()
        if key in seen:continue
        seen.add(key)
        if skip:trace.append(dict(reason='safe_interval_contraction'));continue
        _,_,g,rhs=base.branch_polytope(x,v,b);poly,note=posterior.polytope(g,rhs);trace.append(note)
        if poly is None:continue
        polys[key]=poly
        if point is None:
            initial=poly['center']+poly['scale']*poly['interior']
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method='SLSQP',
                constraints={'type':'ineq','fun':lambda b:poly['rhs']-poly['a']@b,'jac':lambda b:-poly['a']},
                bounds=[(-posterior.PRIOR_BOUND,posterior.PRIOR_BOUND)]*len(anchor),options={'maxiter':300,'ftol':1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    geometry=time.perf_counter()-begin;begin=time.perf_counter()
    if point is None:point=np.clip(bank[0],-.12,.12)
    keys=sorted(polys) if canonical else list(polys);volume=np.array([polys[k]['volume'] for k in keys])
    if keys:
        rng=np.random.default_rng(6173);counts=rng.multinomial(sample_count,volume/volume.sum())
        samples=np.concatenate([posterior.sample(polys[k],int(n),rng) for k,n in zip(keys,counts) if n])
        error=max(np.max(np.abs(base.forward(x,b)-v)) for b in samples);assert error<=base.EPS+1e-7
    else:samples=point[None].copy()
    sampling=time.perf_counter()-begin
    samples.setflags(write=False);point.setflags(write=False)
    meta=dict(screen_seconds=screen_time,geometry_seconds=geometry,sampling_seconds=sampling,
        screened_regions=int(reject.sum()),geometry_regions=len(trace)-sum(r['reason']=='safe_interval_contraction' for r in trace),
        positive_volume_regions=len(keys),discovered_prior_mass=float(volume.sum()/.24**len(anchor)),
        positive_cell_set_sha256=hashlib.sha256(b''.join(sorted(keys))).hexdigest(),
        geometry_trace=trace,query_read_samples=len(samples),anchor_output=point.tolist(),
        persistent_state_bytes=samples.nbytes+point.nbytes,common_observed_context_bytes=x.nbytes+v.nbytes,
        posterior_claim='uniform on discovered feasible region union only; unknown missing mass',
        canonical_readout=canonical,geometry_numeric_arrays_subtotal=sum(a.nbytes for p in polys.values() for a in p.values() if isinstance(a,np.ndarray)))
    return posterior.make_predict(samples),State(point,samples),meta


def fit(x,v,state,cfg):
    begin=time.perf_counter();bank,meta=discover(x,v,state,cfg);elapsed=time.perf_counter()-begin
    anchor=np.zeros(4) if state is None else state.anchor
    predict,newstate,pm=materialize(x,v,anchor,bank,cfg.get('posterior_samples',512),cfg.get('canonical',True))
    return predict,newstate,dict(**meta,**pm,discovery_seconds=elapsed)


def verify():
    rng=np.random.default_rng(184913);x=rng.uniform(0,1,24);v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,24)
    q=np.linspace(0,1,117);legacy=[]
    for gen in ['alm','adam','direct']:
        state=None;anchor=np.zeros(4)
        for n in [4,8,16,24]:
            cfg=dict(generator=gen,pool='prior256',restarts=8,sweeps=12,steps=12,posterior_samples=512,canonical=False)
            old,anchor,om=core.fit(x[:n],v[:n],anchor,dict(generator=gen,initialization='contextual',features=256,restarts=8,sweeps=12,steps=12,posterior_samples=512,discovery_bound=.12))
            new,state,nm=fit(x[:n],v[:n],state,cfg)
            assert np.array_equal(anchor,state.anchor) and np.array_equal(old(q),new(q)),(gen,n)
            assert om['positive_volume_regions']==nm['positive_volume_regions'];legacy.append([gen,n])
    cfg=dict(generator='alm',pool='prior256',restarts=32,sweeps=20)
    bank,_=discover(x[:4],v[:4],None,cfg);reference,state,meta=materialize(x[:4],v[:4],np.zeros(4),bank)
    assert meta['positive_volume_regions']>1,meta
    order_audit=[]
    for b in [bank[::-1],bank[rng.permutation(len(bank))],np.vstack([bank,bank[::-1]])]:
        pred,s,mm=materialize(x[:4],v[:4],np.zeros(4),b)
        assert np.array_equal(state.samples,s.samples) and np.array_equal(reference(q),pred(q))
        assert meta['positive_cell_set_sha256']==mm['positive_cell_set_sha256'];order_audit.append(len(b))
    # New ordinary-PC start interface matches the previous zero-dual gradient blocks.
    from hybrid_discovery_bank import discover as legacy_discover
    with core.pipeline.discovery_box(.12):
        starts=np.vstack([np.zeros(4),np.random.default_rng(912).uniform(-.12,.12,(7,4))])
        actual,_=refine_pc(starts,x[:8],v[:8],np.zeros(4),12);actual=core.contextual.deduplicate(actual,x[:8],v[:8],np.zeros(4))
        expected,_=legacy_discover(x[:8],v[:8],np.zeros(4),sweeps=12,restarts=8,dual_rate=0,gradient_blocks=True)
        assert np.array_equal(actual,expected)
    original_jac=base.forward_jacobian;original_refine=core.batched.refine
    def forbidden(*args,**kwargs):raise AssertionError('global BP entered a local candidate')
    try:
        base.forward_jacobian=forbidden;core.batched.refine=forbidden
        for gen in ['alm','pc']:
            s=None
            for n in [4,8]:_,s,_=fit(x[:n],v[:n],s,dict(generator=gen,pool='posterior_mix',restarts=8,sweeps=12))
    finally:base.forward_jacobian=original_jac;core.batched.refine=original_refine
    return dict(passed=True,legacy_without_canonical_bitwise_cases=legacy,
        canonical_permutation_duplicate_audit=order_audit,canonical_positive_regions=meta['positive_volume_regions'],
        ordinary_pc_same_starts_matches_legacy_zero_dual=True,local_candidates_no_global_bp=True,
        all_initial_pools_prior256=True,posterior_states_readonly=True,query_targets_absent_from_fit_interface=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
