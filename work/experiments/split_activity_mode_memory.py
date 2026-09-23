"""Real online split-activity proposals, with shared feasible-cell readout.

Local ALM is not initialized with BP. BP controls obtain the same one-pass
exact activity relaxation at every visited parameter state, fully costed.
LP/QP and polytope geometry are global, not purely local PC operations.
"""
import hashlib,time
import numpy as np
from scipy.optimize import minimize
import neighbor_mode_memory as neighbor
import split_activity_mode_trace as trace
old=neighbor.previous;base=old.base;posterior=old.posterior


class Collector:
    def __init__(self,x):
        self.x=x;self.forward={};self.split={};self.seconds=0.;self.events=0

    def parameter(self,b,step=0):
        start=time.perf_counter()
        for reg in old.interface.archived.signatures(self.x,b):self.forward.setdefault(reg.tobytes(),None)
        self.seconds+=time.perf_counter()-start

    def activity(self,b,h,u=None,step=0,phase=''):
        start=time.perf_counter();r,d=b.shape;prev=np.broadcast_to(self.x,(r,len(self.x)));codes=[]
        for j in range(d):
            codes.append(np.searchsorted(base.KNOTS,prev+b[:,j,None],side='right').astype(np.uint8));prev=h[j]
        for reg in np.array(codes).transpose(1,0,2):self.split.setdefault(reg.tobytes(),None)
        self.events+=1;self.seconds+=time.perf_counter()-start


def one_pass_relaxation(b,x,v):
    """One identical zero-dual exact activity pass; parameter b never changes."""
    r,d=b.shape;trust=.01;h=np.empty((d,r,len(x)));prev=x
    for j in range(d):h[j]=base.g(prev+b[:,j,None]);prev=h[j]
    for j in reversed(range(d)):
        prev=x if j==0 else h[j-1];before=h[j].copy();a=base.g(prev+b[:,j,None])
        if j==d-1:
            h[j]=np.clip((a+trust*before)/(1+trust),np.maximum(0,v-base.EPS),np.minimum(1,v+base.EPS))
        else:
            nb=b[:,j+1];lo=np.maximum(0,np.array([-np.inf,0,.5,1.])[:,None]-nb)
            hi=np.minimum(1,np.array([0.,.5,1.,np.inf])[:,None]-nb)
            slope=base.SLOPES[:,None,None];offset=(base.SLOPES[:,None]*nb+base.INTERCEPTS[:,None])[:,:,None]
            cand=(a[None]+slope*(h[j+1][None]-offset)+trust*before[None])/(1+slope**2+trust)
            cand=np.minimum(np.maximum(cand,lo[:,:,None]),hi[:,:,None])
            energy=(cand-a)**2+(base.g(cand+nb[None,:,None])-h[j+1])**2+trust*(cand-before)**2
            energy=np.where((lo>hi)[:,:,None],np.inf,energy)
            h[j]=np.take_along_axis(cand,np.argmin(energy,axis=0)[None],axis=0)[0]
    return h


def discover(x,v,cfg):
    pool=old.interface.make_pool(4,'prior256');anchor=np.zeros(4)
    starts,meta=old.interface.select_pool(x,v,anchor,pool,cfg.get('restarts',64));collector=Collector(x)
    with old.core.pipeline.discovery_box(.12):
        gen=cfg['generator']
        if gen in ['alm','nodual','pc']:
            best=trace.refine(starts,x,v,anchor,gen,cfg['sweeps'],collector)
        elif gen=='adam':
            evaluate=old.core.batched.evaluate
            def wrapped(b,*args,**kwargs):
                collector.parameter(b);collector.activity(b,one_pass_relaxation(b,x,v))
                return evaluate(b,*args,**kwargs)
            try:
                old.core.batched.evaluate=wrapped
                best,_=old.core.batched.refine(starts,x,v,anchor,solver='adam',steps=cfg['steps'],lr=.003)
            finally:old.core.batched.evaluate=evaluate
        else:raise ValueError(gen)
        bank=old.core.contextual.deduplicate(np.vstack([starts,best]),x,v,anchor)
    first=[r.tobytes() for r in old.interface.archived.signatures(x,bank)]
    forward=list(dict.fromkeys(first+list(collector.forward)))
    extra=[k for k in collector.split if k not in set(forward)]
    keys=forward+extra;regs=np.array([np.frombuffer(k,np.uint8).reshape(4,len(x)) for k in keys])
    return bank,regs,dict(**meta,effective_generator=gen,effective_pool='prior256',
        forward_patterns=len(forward),split_patterns=len(collector.split),additional_split_patterns=len(extra),
        pattern_key_numeric_bytes=sum(map(len,keys)),collector_events=collector.events,
        collection_bookkeeping_seconds=collector.seconds,retained_pattern_array_bytes=regs.nbytes)


def materialize(x,v,bank,regs,sample_count=512):
    start=time.perf_counter();skip=old.screen.contract(x,v,regs,5);screen=time.perf_counter()-start
    start=time.perf_counter();polys={};notes=[];point=None;anchor=np.zeros(4)
    for reg,reject in zip(regs,skip):
        if reject:continue
        _,_,g,rhs=neighbor.pattern_matrix(x,v,reg);poly,note=posterior.polytope(g,rhs);notes.append(note)
        if poly is None:continue
        key=reg.tobytes();polys[key]=poly
        if point is None:
            initial=poly['center']+poly['scale']*poly['interior']
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method='SLSQP',
                constraints={'type':'ineq','fun':lambda b:poly['rhs']-poly['a']@b,'jac':lambda b:-poly['a']},
                bounds=[(-.12,.12)]*4,options={'maxiter':300,'ftol':1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    if point is None:point=np.clip(bank[0],-.12,.12)
    geometry=time.perf_counter()-start;start=time.perf_counter();keys=sorted(polys)
    volume=np.array([polys[k]['volume'] for k in keys])
    if keys:
        rng=np.random.default_rng(6173);counts=rng.multinomial(sample_count,volume/volume.sum())
        samples=np.concatenate([posterior.sample(polys[k],int(n),rng) for k,n in zip(keys,counts) if n])
        assert max(np.max(np.abs(base.forward(x,b)-v)) for b in samples)<=base.EPS+1e-7
    else:samples=point[None].copy()
    sampling=time.perf_counter()-start;samples.setflags(write=False);point.setflags(write=False)
    return posterior.make_predict(samples),old.State(point,samples),dict(screen_seconds=screen,geometry_seconds=geometry,sampling_seconds=sampling,
        initial_screened_patterns=int(skip.sum()),geometry_calls=len(notes),positive_volume_regions=len(keys),
        positive_mode_keys=[k.hex() for k in keys],positive_cell_set_sha256=hashlib.sha256(b''.join(keys)).hexdigest(),
        anchor_output=point.tolist(),query_read_samples=len(samples),persistent_state_bytes=point.nbytes+samples.nbytes,
        geometry_numeric_arrays_subtotal=sum(a.nbytes for p in polys.values() for a in p.values() if isinstance(a,np.ndarray)),
        discovered_prior_mass=float(volume.sum()/.24**4),geometry_trace=notes,
        posterior_claim='uniform over discovered union; split activities only propose patterns, never answer queries')


def fit(x,v,state,cfg):
    if not cfg.get('split_proposals',False) or state is not None:
        predict,state,meta=neighbor.fit(x,v,state,cfg)
        return predict,state,dict(**meta,split_proposals_active=False)
    start=time.perf_counter();bank,regs,meta=discover(x,v,cfg);discovery=time.perf_counter()-start
    predict,state,pm=materialize(x,v,bank,regs,cfg.get('posterior_samples',512))
    return predict,state,dict(**meta,**pm,discovery_seconds=discovery,split_proposals_active=True)


def verify():
    rng=np.random.default_rng(184913);x=rng.uniform(0,1,24);v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,24)
    checks=[]
    for gen in ['alm','nodual','pc','adam']:
        cfg=dict(generator=gen,sweeps=12,steps=12,restarts=8,archive=True,split_proposals=True,completion_rounds=0)
        bank,regs,meta=discover(x[:4],v[:4],cfg)
        pool=old.interface.make_pool(4,'prior256');starts,_=old.interface.select_pool(x[:4],v[:4],np.zeros(4),pool,8)
        if gen!='adam':
            obs=trace.Observer(x[:4],v[:4])
            with old.core.pipeline.discovery_box(.12):trace.refine(starts,x[:4],v[:4],np.zeros(4),gen,12,obs)
            expected=set(obs.forward)|set(obs.split)
            assert {r.tobytes().hex() for r in regs}==expected
        # Forward-only explicit geometry is bitwise identical to previous materialization.
        oldcfg=dict(generator='alm' if gen=='nodual' else gen,restarts=8,sweeps=12,steps=12,archive=True,
                    dual_rate=0. if gen=='nodual' else .5,completion_rounds=0)
        reference,state,om=neighbor.fit(x[:4],v[:4],None,oldcfg)
        direct,actual,_=materialize(x[:4],v[:4],bank,regs[:meta['forward_patterns']])
        assert np.array_equal(state.samples,actual.samples) and np.array_equal(state.anchor,actual.anchor),gen
        checks.append(gen)
    before_jac=base.forward_jacobian;before_bp=old.core.batched.refine
    def forbidden(*args,**kwargs):raise AssertionError('BP entered local ALM')
    try:
        base.forward_jacobian=forbidden;old.core.batched.refine=forbidden
        fit(x[:4],v[:4],None,dict(generator='alm',sweeps=12,restarts=8,split_proposals=True))
    finally:base.forward_jacobian=before_jac;old.core.batched.refine=before_bp
    # BP's augmentation is the exact same first zero-dual activity pass.
    class Check:
        def parameter(self,*args):pass
        def activity(self,b,h,u,step,phase):
            if phase=='after_activities':assert np.array_equal(h,one_pass_relaxation(b,x[:4],v[:4]))
    with old.core.pipeline.discovery_box(.12):trace.refine(starts,x[:4],v[:4],np.zeros(4),'alm',1,Check())
    return dict(passed=True,forward_only_original_state_bitwise=checks,local_pattern_collector_matches_diagnostic=True,
        bp_augmentation_same_zero_dual_activity_pass=True,local_candidate_no_global_bp=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
