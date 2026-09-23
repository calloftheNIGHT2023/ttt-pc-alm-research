"""Online safe rejection cache from already-computed local activity credit.

No extra primal/dual optimization, no BP initialization of the local candidate.
All uncertain patterns retain the previous common screening/global geometry.
"""
import time
import numpy as np
import split_activity_mode_memory as previous
import local_region_screen as screen
base=previous.base;old=previous.old;trace=previous.trace


class CreditCollector(previous.Collector):
    def __init__(self,x,v,mode,capture=False):
        super().__init__(x);self.v=v;self.mode=mode;self.capture=capture
        self.rejected={};self.proofs={};self.credit_seconds=0.;self.rough_tests=0;self.strict_tests=0
        _,_,self.hl,self.hh=screen.boxes(v,np.zeros((1,4,len(x)),dtype=np.uint8))

    def activity(self,b,h,u=None,step=0,phase=''):
        begin=time.perf_counter();r,d=b.shape;prev=np.broadcast_to(self.x,(r,len(self.x)));codes=[];residual=[]
        for j in range(d):
            z=prev+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8))
            residual.append(h[j]-base.g(z));prev=h[j]
        split=np.array(codes).transpose(1,0,2)
        for reg in split:self.split.setdefault(reg.tobytes(),None)
        self.events+=1;self.seconds+=time.perf_counter()-begin;begin=time.perf_counter()
        res=np.array(residual).transpose(1,0,2)
        credits=[('residual',res)]
        if self.mode=='all' and u is not None and np.any(u):
            raw=u.transpose(1,0,2);credits.extend([('raw',raw),('augmented',raw+res)])
        forward=old.interface.archived.signatures(self.x,b)
        for regs in [forward,split]:
            slopes=base.SLOPES[regs];offset=base.INTERCEPTS[regs]
            for label,credit in credits:
                p=slopes*credit;hc=credit.copy();hc[:,:-1]-=p[:,1:]
                # p=s*a makes the z coefficient exactly zero for slopes 0,+/-2.
                rough=-screen.B*np.abs(p.sum(2)).sum(1)+(hc*np.where(hc>=0,self.hl,self.hh)).sum((1,2))
                rough-=(credit*offset).sum((1,2))+(p[:,0]*self.x).sum(1)
                scale=1+np.abs(p).sum((1,2))+np.abs(credit).sum((1,2));ids=np.flatnonzero(rough>1e-10*scale)
                self.rough_tests+=len(regs)
                ids=np.array([i for i in ids if regs[i].tobytes() not in self.rejected],dtype=int)
                if not len(ids):continue
                lower=screen.certified_lower_bound(self.x,self.v,regs[ids],p[ids],credit[ids]);self.strict_tests+=len(ids)
                for i,value in zip(ids,lower):
                    if value<=0:continue
                    key=regs[i].tobytes()
                    self.rejected[key]=(float(value),label)
                    if self.capture:self.proofs[key]=dict(p=p[i].copy(),a=credit[i].copy(),lower=float(value),step=step,phase=phase,label=label)
        self.credit_seconds+=time.perf_counter()-begin


def discover(x,v,cfg):
    anchor=np.zeros(4);pool=old.interface.make_pool(4,'prior256')
    starts,meta=old.interface.select_pool(x,v,anchor,pool,cfg.get('restarts',64))
    collector=CreditCollector(x,v,cfg['credit_mode'],cfg.get('capture_proofs',False))
    with old.core.pipeline.discovery_box(.12):
        gen=cfg['generator']
        if gen in ['alm','nodual','pc']:
            best=trace.refine(starts,x,v,anchor,gen,cfg['sweeps'],collector)
        elif gen=='adam':
            evaluate=old.core.batched.evaluate
            def wrapped(b,*args,**kwargs):
                collector.parameter(b);collector.activity(b,previous.one_pass_relaxation(b,x,v))
                return evaluate(b,*args,**kwargs)
            try:
                old.core.batched.evaluate=wrapped
                best,_=old.core.batched.refine(starts,x,v,anchor,solver='adam',steps=cfg['steps'],lr=.003)
            finally:old.core.batched.evaluate=evaluate
        else:raise ValueError(gen)
        bank=old.core.contextual.deduplicate(np.vstack([starts,best]),x,v,anchor)
    first=[r.tobytes() for r in old.interface.archived.signatures(x,bank)]
    forward=list(dict.fromkeys(first+list(collector.forward)));seen=set(forward)
    extra=[k for k in collector.split if k not in seen];keys=forward+extra
    assert set(collector.rejected)<=set(keys)
    kept=[k for k in keys if k not in collector.rejected]
    regs=np.array([np.frombuffer(k,np.uint8).reshape(4,len(x)) for k in kept],dtype=np.uint8).reshape(-1,4,len(x))
    return bank,regs,dict(**meta,effective_generator=gen,effective_pool='prior256',
        forward_patterns=len(forward),split_patterns=len(collector.split),additional_split_patterns=len(extra),
        original_pattern_count=len(keys),remaining_pattern_count=len(kept),pattern_key_numeric_bytes=sum(map(len,keys)),
        credit_mode=cfg['credit_mode'],certified_rejected_patterns=len(collector.rejected),
        certified_mode_keys=[k.hex() for k in collector.rejected],certificate_lower_bounds=[v[0] for v in collector.rejected.values()],
        certificate_labels=[v[1] for v in collector.rejected.values()],certificate_cache_key_and_lower_bytes=sum(len(k)+8 for k in collector.rejected),
        credit_rough_row_tests=collector.rough_tests,credit_strict_row_tests=collector.strict_tests,credit_check_seconds=collector.credit_seconds,
        collector_events=collector.events,collection_bookkeeping_seconds=collector.seconds,retained_pattern_array_bytes=regs.nbytes),collector


def fit(x,v,state,cfg):
    mode=cfg.get('credit_mode','none')
    if state is not None or mode=='none':
        predict,newstate,meta=previous.fit(x,v,state,cfg)
        return predict,newstate,dict(**meta,credit_cache_active=False,credit_mode=mode)
    begin=time.perf_counter();bank,regs,meta,collector=discover(x,v,cfg);discovery=time.perf_counter()-begin
    predict,newstate,pm=previous.materialize(x,v,bank,regs,cfg.get('posterior_samples',512))
    return predict,newstate,dict(**meta,**pm,discovery_seconds=discovery,credit_cache_active=True,split_proposals_active=True)


def verify():
    from fractions import Fraction
    from certified_branch_solver import exact_certificate
    rng=np.random.default_rng(184913);x=rng.uniform(0,1,24);v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,24)
    states=0;proofs=0;sets=0
    for gen in ['alm','nodual','pc','adam']:
        cfg=dict(generator=gen,sweeps=12,steps=12,restarts=8,split_proposals=True)
        ref,state,rm=previous.fit(x[:4],v[:4],None,cfg)
        for mode in ['residual','all']:
            pred,new,nm=fit(x[:4],v[:4],None,dict(**cfg,credit_mode=mode))
            assert np.array_equal(state.samples,new.samples) and np.array_equal(state.anchor,new.anchor)
            assert rm['positive_mode_keys']==nm['positive_mode_keys'];states+=1
            bank,regs,meta,observer=discover(x[:4],v[:4],dict(**cfg,credit_mode=mode,capture_proofs=True))
            for k,proof in observer.proofs.items():
                with old.core.pipeline.discovery_box(.12):
                    exact=exact_certificate(x[:4],v[:4],np.frombuffer(k,np.uint8).reshape(4,4),proof['p'],proof['a'])
                rational=Fraction(int(exact['numerator']),int(exact['denominator']))
                assert rational>0 and Fraction(proof['lower'])<=rational;proofs+=1
            assert set(observer.rejected).isdisjoint(bytes.fromhex(k) for k in rm['positive_mode_keys']);sets+=1
    # A known old support context exercises actual positive certificates.
    rng=np.random.default_rng(5900000);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=base.forward(xx,truth)+np.random.default_rng(24900000).uniform(-base.EPS,base.EPS,24)
    cfg=dict(generator='alm',sweeps=16,restarts=64,split_proposals=True,credit_mode='all',capture_proofs=True)
    _,_,_,observer=discover(xx[:4],vv[:4],cfg)
    assert observer.proofs, 'The test must exercise positive cuts, not pass vacuously'
    for k,proof in observer.proofs.items():
        with old.core.pipeline.discovery_box(.12):
            exact=exact_certificate(xx[:4],vv[:4],np.frombuffer(k,np.uint8).reshape(4,4),proof['p'],proof['a'])
        rational=Fraction(int(exact['numerator']),int(exact['denominator']))
        assert rational>0 and Fraction(proof['lower'])<=rational;proofs+=1
    before=old.core.batched.refine;beforejac=base.forward_jacobian
    def forbidden(*args,**kwargs):raise AssertionError('BP entered local credit cache')
    try:
        old.core.batched.refine=forbidden;base.forward_jacobian=forbidden
        fit(x[:4],v[:4],None,dict(generator='alm',sweeps=12,restarts=8,split_proposals=True,credit_mode='all'))
    finally:old.core.batched.refine=before;base.forward_jacobian=beforejac
    return dict(passed=True,original_states_bitwise=states,positive_sets_unchanged=sets,exact_certificate_checks=proofs,
        local_candidate_no_global_bp=True,no_reference_or_query_in_fit=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
