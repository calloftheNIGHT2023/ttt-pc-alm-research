"""Common batched safe credit verification and cross-mode direction banks.

Local candidates never use global BP credit. BP comparators reuse their own
adjoints, with all calculation/history/bank costs charged to the comparator.
No reference cells, teacher, query or query answer enters this module.
"""
import hashlib,time
import numpy as np
from scipy.optimize import minimize
import credit_cached_mode_memory as legacy
previous=legacy.previous;old=previous.old;base=previous.base;screen=legacy.screen
posterior=previous.posterior;trace=previous.trace


class Collector:
    def __init__(self,x,v,kind,capture=False):
        self.x=x;self.v=v;self.kind=kind;self.capture=capture
        self.forward={};self.split={};self.pending={};self.last=None
        self.current=None;self.history=None;self.events=0;self.rough_tests=0
        self.credit_seconds=0.;self.strict_seconds=0.;self.pending_updates=0
        _,_,self.hl,self.hh=screen.boxes(v,np.zeros((1,4,len(x)),dtype=np.uint8))

    def ensure_forward(self,b):
        if self.last is not None and np.array_equal(b,self.last):return
        r,d=b.shape;h=np.broadcast_to(self.x,(r,len(self.x)));codes=[];deriv=[]
        for j in range(d):
            z=h+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8))
            if self.kind=='bp':deriv.append(base.derivative(z))
            h=base.g(z)
        self.codes=np.array(codes).transpose(1,0,2);self.last=b.copy()
        if self.kind=='bp':
            raw=h-self.v;res=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0)
            alpha=np.empty((r,d,len(self.x)));alpha[:,-1]=-res
            for j in range(d-2,-1,-1):alpha[:,j]=deriv[j+1]*alpha[:,j+1]
            if self.current is None:self.history=np.zeros_like(alpha)
            else:self.history+=.5*self.current
            self.current=alpha

    def parameter(self,b,step=0):
        self.ensure_forward(b)
        for reg in self.codes:self.forward.setdefault(reg.tobytes(),None)

    def rough(self,regs,credit):
        p=base.SLOPES[regs]*credit;hc=credit.copy();hc[:,:-1]-=p[:,1:]
        value=-screen.B*np.abs(p.sum(2)).sum(1)+(hc*np.where(hc>=0,self.hl,self.hh)).sum((1,2))
        value-=(credit*base.INTERCEPTS[regs]).sum((1,2))+(p[:,0]*self.x).sum(1)
        scale=1+np.abs(p).sum((1,2))+np.abs(credit).sum((1,2))
        return value,scale

    def activity(self,b,h,u=None,step=0,phase=''):
        begin=time.perf_counter();self.ensure_forward(b);r,d=b.shape
        prev=np.broadcast_to(self.x,(r,len(self.x)));codes=[];residual=[]
        for j in range(d):
            z=prev+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8))
            residual.append(h[j]-base.g(z));prev=h[j]
        split=np.array(codes).transpose(1,0,2)
        for reg in split:self.split.setdefault(reg.tobytes(),None)
        res=np.array(residual).transpose(1,0,2);credits=[('residual',res)]
        if self.kind=='local' and u is not None and np.any(u):
            raw=u.transpose(1,0,2);credits.extend([('raw',raw),('augmented',raw+res)])
        elif self.kind=='bp':
            credits.extend([('current_bp',self.current),('history_bp',self.history),('combined_bp',self.current+self.history)])
        regs=np.concatenate([self.codes,split],axis=0)
        for label,a in credits:
            if not np.any(a):continue
            doubled=np.concatenate([a,a],axis=0);value,scale=self.rough(regs,doubled);self.rough_tests+=len(regs)
            for i in np.flatnonzero(value>1e-10*scale):
                key=regs[i].tobytes();ratio=float(value[i]/scale[i])
                if key not in self.pending or ratio>self.pending[key][0]:
                    self.pending[key]=(ratio,a[i%r].copy(),label);self.pending_updates+=1
        self.events+=1;self.credit_seconds+=time.perf_counter()-begin

    def finalize(self):
        """A rough candidate is NOT a rejection until this mandatory check."""
        begin=time.perf_counter();accepted={};proofs=[];keys=list(self.pending)
        if keys:
            regs=np.array([np.frombuffer(k,np.uint8).reshape(4,len(self.x)) for k in keys])
            aa=np.array([self.pending[k][1] for k in keys]);pp=base.SLOPES[regs]*aa
            lower=screen.certified_lower_bound(self.x,self.v,regs,pp,aa)
            for i,key in enumerate(keys):
                if lower[i]<=0:continue
                accepted[key]=(float(lower[i]),self.pending[key][1],self.pending[key][2])
                if self.capture:proofs.append(dict(pattern=key.hex(),a=aa[i].copy(),p=pp[i].copy(),lower=float(lower[i]),stage='matching'))
        self.strict_seconds=time.perf_counter()-begin
        return accepted,proofs


def direction_bank(accepted,limit):
    selected=[];seen=set()
    for key in sorted(accepted):
        a=accepted[key][1];scale=float(np.max(np.abs(a)))
        if scale==0:continue
        a=a/scale;token=a.tobytes()
        if token in seen:continue
        selected.append(a);seen.add(token)
        if len(selected)==limit:break
    shape=next(iter(accepted.values()))[1].shape if accepted else (4,4)
    return np.array(selected).reshape(-1,*shape)


def prepare(x,v,cfg):
    start=time.perf_counter();anchor=np.zeros(4);pool=old.interface.make_pool(4,'prior256')
    starts,meta=old.interface.select_pool(x,v,anchor,pool,cfg.get('restarts',64))
    collector=Collector(x,v,cfg['credit_kind'],cfg.get('capture_proofs',False))
    with old.core.pipeline.discovery_box(.12):
        gen=cfg['generator']
        if gen in ['alm','nodual','pc']:
            assert cfg['credit_kind']!='bp', 'No BP credit is allowed in local candidates'
            best=trace.refine(starts,x,v,anchor,gen,cfg['sweeps'],collector)
        elif gen=='adam':
            assert cfg['credit_kind']=='bp'
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
    discovery=time.perf_counter()-start
    accepted,proofs=collector.finalize();assert set(accepted)<=set(keys)
    begin=time.perf_counter();kept=[k for k in keys if k not in accepted]
    regs=np.array([np.frombuffer(k,np.uint8).reshape(4,len(x)) for k in kept],dtype=np.uint8).reshape(-1,4,len(x))
    rejected=screen.contract(x,v,regs,5);regs=regs[~rejected];screen_seconds=time.perf_counter()-begin
    begin=time.perf_counter();directions=np.empty((0,4,len(x)));cross={};array_subtotal=0;pair_count=0
    if cfg.get('bank_size',0)>0 and accepted:
        directions=direction_bank(accepted,cfg['bank_size'])
        if len(directions) and len(regs):
            k=len(directions);rr=np.repeat(regs,k,axis=0);aa=np.tile(directions,(len(regs),1,1))
            value,scale=collector.rough(rr,aa);values=value.reshape(len(regs),k);scales=scale.reshape(len(regs),k)
            selected=values.argmax(1);chosen=values[np.arange(len(regs)),selected];threshold=scales[np.arange(len(regs)),selected]
            ids=np.flatnonzero(chosen>1e-10*threshold);flat=ids*k+selected[ids];pp=base.SLOPES[rr[flat]]*aa[flat]
            lower=screen.certified_lower_bound(x,v,rr[flat],pp,aa[flat]) if len(ids) else []
            mask=np.ones(len(regs),bool)
            for i,fi,p,lb in zip(ids,flat,pp,lower):
                if lb<=0:continue
                key=regs[i].tobytes();cross[key]=float(lb);mask[i]=False
                if collector.capture:proofs.append(dict(pattern=key.hex(),a=aa[fi].copy(),p=p.copy(),lower=float(lb),stage='cross'))
            regs=regs[mask];pair_count=len(rr)
            array_subtotal=rr.nbytes+aa.nbytes+value.nbytes+scale.nbytes+pp.nbytes
    cross_seconds=time.perf_counter()-begin
    numeric_pending=sum(len(k)+8+a.nbytes for k,(ratio,a,label) in collector.pending.items())
    return bank,regs,dict(**meta,effective_generator=gen,effective_pool='prior256',
        original_pattern_count=len(keys),forward_patterns=len(forward),additional_split_patterns=len(extra),
        matching_candidate_patterns=len(collector.pending),matching_certified_patterns=len(accepted),cross_certified_patterns=len(cross),
        matching_mode_keys=[k.hex() for k in accepted],matching_lower_bounds=[r[0] for r in accepted.values()],
        cross_mode_keys=[k.hex() for k in cross],cross_lower_bounds=list(cross.values()),
        safe_screened_after_matching=int(rejected.sum()),retained_geometry_patterns=len(regs),
        pending_credit_numeric_bytes=numeric_pending,direction_bank_numeric_bytes=directions.nbytes,direction_bank_size=len(directions),
        main_cross_batch_array_bytes_subtotal=array_subtotal,cross_mode_direction_pairs=pair_count,
        discovery_seconds=discovery,credit_collection_seconds=collector.credit_seconds,strict_matching_seconds=collector.strict_seconds,
        safe_screen_seconds=screen_seconds,cross_credit_seconds=cross_seconds,
        matching_pending_updates=collector.pending_updates,credit_rough_row_tests=collector.rough_tests,
        direction_selection='one max normalized rough score per pattern; strict validation; canonical pattern order, normalized, first distinct K'),proofs


def materialize_retained(x,v,bank,regs,sample_count=512):
    """Same geometry/readout, with the common screening already done once."""
    start=time.perf_counter();polys={};notes=[];point=None;anchor=np.zeros(4)
    for reg in regs:
        _,_,g,rhs=previous.neighbor.pattern_matrix(x,v,reg);poly,note=posterior.polytope(g,rhs);notes.append(note)
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
    return posterior.make_predict(samples),old.State(point,samples),dict(geometry_seconds=geometry,sampling_seconds=sampling,
        geometry_calls=len(notes),positive_volume_regions=len(keys),positive_mode_keys=[k.hex() for k in keys],
        positive_cell_set_sha256=hashlib.sha256(b''.join(keys)).hexdigest(),anchor_output=point.tolist(),
        query_read_samples=len(samples),persistent_state_bytes=point.nbytes+samples.nbytes,
        geometry_numeric_arrays_subtotal=sum(a.nbytes for p in polys.values() for a in p.values() if isinstance(a,np.ndarray)),
        geometry_trace=notes,posterior_claim='same discovered positive union; all deletions have safe contractor or strict credit bound')


def fit(x,v,state,cfg):
    backend=cfg.get('backend','none')
    if state is not None or backend=='none':
        pred,new,meta=previous.fit(x,v,state,cfg)
        return pred,new,dict(**meta,batch_credit_active=False)
    if backend=='legacy':
        pred,new,meta=legacy.fit(x,v,state,dict(**cfg,credit_mode='all'))
        return pred,new,dict(**meta,batch_credit_active=False)
    assert backend=='batch'
    bank,regs,meta,proofs=prepare(x,v,cfg)
    pred,new,pm=materialize_retained(x,v,bank,regs,cfg.get('posterior_samples',512))
    return pred,new,dict(**meta,**pm,batch_credit_active=True,split_proposals_active=True)


def verify():
    from fractions import Fraction
    from certified_branch_solver import exact_certificate
    checks=0;proof_checks=0
    for seed in [5900000,5900001]:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        for gen,kind in [('alm','local'),('alm','residual'),('adam','bp')]:
            cfg=dict(generator=gen,sweeps=16,steps=16,restarts=64,split_proposals=True,backend='batch',credit_kind=kind)
            reference,state,rm=previous.fit(x[:4],v[:4],None,cfg)
            for size in [0,32]:
                pred,new,nm=fit(x[:4],v[:4],None,dict(**cfg,bank_size=size))
                assert np.array_equal(state.samples,new.samples) and np.array_equal(state.anchor,new.anchor)
                assert rm['positive_mode_keys']==nm['positive_mode_keys'];checks+=1
                _,_,meta,proofs=prepare(x[:4],v[:4],dict(**cfg,bank_size=size,capture_proofs=True))
                assert len(proofs)==meta['matching_certified_patterns']+meta['cross_certified_patterns']
                for proof in proofs:
                    with old.core.pipeline.discovery_box(.12):
                        exact=exact_certificate(x[:4],v[:4],np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4),proof['p'],proof['a'])
                    value=Fraction(int(exact['numerator']),int(exact['denominator']))
                    assert value>0 and Fraction(proof['lower'])<=value;proof_checks+=1
    assert proof_checks>0
    oldbp=old.core.batched.refine;oldjac=base.forward_jacobian
    def forbidden(*args,**kwargs):raise AssertionError('global BP entered local batch credit candidate')
    try:
        old.core.batched.refine=forbidden;base.forward_jacobian=forbidden
        fit(x[:4],v[:4],None,dict(generator='alm',sweeps=16,restarts=64,backend='batch',credit_kind='local',bank_size=32))
    finally:old.core.batched.refine=oldbp;base.forward_jacobian=oldjac
    return dict(passed=True,original_state_and_positive_set_bitwise_cases=checks,strict_exact_proofs=proof_checks,
        local_candidate_no_global_bp=True,query_or_reference_absent_from_fit=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
