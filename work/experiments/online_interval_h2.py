"""Causal H2 with outward interval certificates and exact fallback.

Frozen rational variants are retained as contemporaneous controls. Only the
verification of the already-selected direction changes; no extra queries.
"""
import time
import numpy as np
import online_factorized_h2 as original
import interval_credit_certificate as interval

prior=original.prior
base=original.base
neighbor=original.neighbor
RationalBank=original.factor.Bank


def configs():
    answer=[]
    for cfg in original.configs():
        answer.append(cfg)
        if cfg.get('factorized_full'):
            answer.append(dict(cfg,name=cfg['name']+'_interval',interval_verifier=True,rational_comparator=cfg['name']))
    return answer


class IntervalBank(RationalBank):
    def screen(self,regs):
        start=time.perf_counter();values=self.values(regs);float_seconds=time.perf_counter()-start
        reject=np.zeros(len(regs),bool);proofs=[];fallback=0;direct=0
        interval_seconds=0.;fraction_seconds=0.;gate_count=0
        if len(regs) and self.k:
            selected=values.argmax(1);rough=values[np.arange(len(regs)),selected]
            ids=np.flatnonzero(rough>1e-10*(1+abs(self.bank[selected]).sum((1,2))));gate_count=len(ids)
            if len(ids):
                start=time.perf_counter();enclosure=interval.enclose(self.x,self.v,regs[ids],self.bank[selected[ids]])
                interval_seconds=time.perf_counter()-start
                for k,i in enumerate(ids):
                    proof=dict(pattern=regs[i].tobytes().hex(),direction=int(selected[i]))
                    if enclosure['positive'][k]:
                        proof.update(kind='interval',lower=float(enclosure['lower'][k]),upper=float(enclosure['upper'][k]),
                            empty_shared_bias=bool(enclosure['empty_shared_bias'][k]));direct+=1
                    else:
                        start=time.perf_counter();exact=interval.original.exact_optimum(self.x,self.v,regs[i],self.bank[selected[i]])
                        fraction_seconds+=time.perf_counter()-start;fallback+=1
                        if not exact['positive']:continue
                        proof.update(kind='rational_fallback',exact=exact)
                    reject[i]=True;proofs.append(proof)
        return reject,proofs,dict(float_seconds=float_seconds,exact_seconds=fraction_seconds,interval_seconds=interval_seconds,
            rational_checks=fallback,interval_direct=direct,verification_proposals=gate_count,
            layer_solves=self.layer_solves,calls=self.calls,region_rows=self.region_rows,
            cached_values_numeric_bytes=sum(len(k)+v[0].nbytes+1 for cache in self.cache for k,v in cache.items()))


def fit(x,v,state,cfg,rng,count=2048):
    if not cfg.get('interval_verifier') or state is not None:return original.fit(x,v,state,cfg,rng,count)
    saved=original.factor.Bank
    try:
        original.factor.Bank=IntervalBank
        prediction,new,meta=original.fit(x,v,state,cfg,rng,count)
    finally:original.factor.Bank=saved
    # The original driver uses exact_checks as an accepted-proof count.
    # Here label actual Fraction calls, and retain a separate proof count.
    for call in meta['screen_calls']:
        credit=call['credit'];credit['certified_regions']=call['credit_rejected'];credit['exact_checks']=credit['rational_checks']
    meta['verification_kind']='outward_interval_then_exact_if_uncertain'
    return prediction,new,meta


def verify():
    from fractions import Fraction as F
    rng=np.random.default_rng(5900001);b=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=base.forward(xx,b)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    cases=0;proofs=0;reference=None;cfg_last=None
    for cfg in [c for c in configs() if c.get('interval_verifier')]:
        control=dict(cfg,interval_verifier=False,name=cfg['rational_comparator'])
        pred,ref,rm=original.fit(x,v,None,control,np.random.default_rng(482401),64)
        actual,state,meta=fit(x,v,None,cfg,np.random.default_rng(482401),64)
        assert np.array_equal(ref.samples,state.samples) and np.array_equal(ref.anchor,state.anchor)
        assert np.array_equal(pred(xx),actual(xx)) and meta['positive_mode_keys']==rm['positive_mode_keys']
        for key in ['discovery_bank_sha256','completion_proposals','geometry_calls']:assert meta[key]==rm[key]
        assert [c['input_pattern_hash'] for c in meta['screen_calls']]==[c['input_pattern_hash'] for c in rm['screen_calls']]
        assert [(p['pattern'],p['direction'],p['wave']) for p in meta['credit_proofs']]==[(p['pattern'],p['direction'],p['wave']) for p in rm['credit_proofs']]
        for pp,rr in zip(meta['credit_proofs'],rm['credit_proofs']):
            exact=rr['exact']
            if pp['kind']=='interval':
                assert pp['lower']>0
                if pp['empty_shared_bias']:assert 'empty_layer' in exact
                else:
                    value=F(int(exact['numerator']),int(exact['denominator']))
                    assert F(pp['lower'])<=value<=F(pp['upper'])
            else:assert pp['exact']==exact
            proofs+=1
        if cfg['name']=='alm_native_full_c5_interval':reference=(ref,rm);cfg_last=cfg
        cases+=1
    # Force an inconclusive interval to exercise the exact fallback branch.
    enclose=interval.enclose
    def inconclusive(x,v,regs,a):
        return dict(lower=np.full(len(regs),-np.inf),upper=np.full(len(regs),np.inf),
            empty_shared_bias=np.zeros(len(regs),bool),positive=np.zeros(len(regs),bool))
    try:
        interval.enclose=inconclusive
        _,state,meta=fit(x,v,None,cfg_last,np.random.default_rng(482401),64)
        assert np.array_equal(state.samples,reference[0].samples)
        assert [p['exact'] for p in meta['credit_proofs']]==[p['exact'] for p in reference[1]['credit_proofs']]
        fallback=sum(c['credit']['exact_checks'] for c in meta['screen_calls']);assert fallback>0
    finally:interval.enclose=enclose
    core=neighbor.previous.core;jac=base.forward_jacobian;refine=core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP in local interval H2')
    try:
        base.forward_jacobian=forbidden;core.batched.refine=forbidden;state=None
        for n in [4,8,16,24]:_,state,_=fit(xx[:n],vv[:n],state,cfg_last,np.random.default_rng(482401+n),64)
    finally:base.forward_jacobian=jac;core.batched.refine=refine
    return dict(passed=True,interval_configs_bitwise=cases,interval_proofs_against_exact=proofs,
        forced_fallback_checks=fallback,local_four_stage_no_global_bp=True,scope='old seed only; no task efficacy claim')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
