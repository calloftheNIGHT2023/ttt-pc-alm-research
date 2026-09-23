"""Read-only bounded credit snapshots, used only before existing H2 geometry.

No activity/parameter repair, extra branch proposal or query-based selection.
Exact positive certificates are mandatory. H2 visited sets and finite proposal
budgets are determined before screening; empty cells cannot enter its frontier.
"""
import hashlib
import time
import numpy as np
import independent_hybrid_memory as prior
import split_activity_mode_trace as trace
import optimized_branch_dual as certificate

neighbor=prior.neighbor
base=prior.base


def config(learner,mode):
    assert learner in ['alm','adam60','pc','nodual','direct4096']
    assert mode in ['c5','c20','native','residual','axes','random']
    gen={'adam60':'adam','direct4096':'direct','nodual':'alm'}.get(learner,learner)
    return dict(name=f'{learner}_{mode}',learner=learner,credit_mode=mode,family='h2',generator=gen,
        dual_rate=0. if learner=='nodual' else .5,sweeps=80 if learner=='pc' else 16,steps=60,
        initial_features=4096 if learner=='direct4096' else 256,restarts=128 if learner=='direct4096' else 64,
        archive=True,pool='posterior_mix',completion_rounds=2,proposal_budget=1024,sampler='geometry',direction_budget=32)


def configs():
    answer=[]
    for learner in ['alm','adam60','pc','nodual','direct4096']:
        modes=['c5','c20','axes','random']
        if learner!='direct4096':modes+=['native']
        if learner in ['alm','adam60']:modes+=['residual']
        answer.extend(config(learner,mode) for mode in modes)
    return answer


class Snapshots:
    def __init__(self,x,v,cfg):
        self.x=x;self.v=v;self.cfg=cfg;self.values=[];self.events=[];self.seconds=0.;self.steps=0
        iterations=cfg['steps'] if cfg['learner']=='adam60' else cfg['sweeps']
        self.checkpoints={int(np.ceil(iterations*j/4)) for j in [1,2,3,4]}
        self.previous=None;self.history=None

    def add(self,array,label,step):
        self.values.append(np.ascontiguousarray(array).copy());self.events.append(dict(label=label,step=int(step),rows=len(array)))

    def parameter(self,b,step=0):pass

    def activity(self,b,h,u=None,step=0,phase=''):
        if phase!='after_parameters_before_dual' or step not in self.checkpoints:return
        start=time.perf_counter();prev=np.broadcast_to(self.x,(len(b),len(self.x)));rr=[]
        for j in range(len(h)):
            rr.append(h[j]-base.g(prev+b[:,j,None]));prev=h[j]
        residual=np.array(rr).transpose(1,0,2)
        self.add(residual,'residual',step)
        if self.cfg['credit_mode']=='native' and self.cfg['learner']=='alm':
            raw=u.transpose(1,0,2);self.add(raw,'raw',step);self.add(raw+residual,'augmented',step)
        self.seconds+=time.perf_counter()-start

    def evaluate(self,bank,x,v,with_jacobian=True):
        # The original forward-Jacobian arithmetic is unchanged. Downstream
        # adjoints reuse slopes from this pass; no second forward observer.
        r,d=bank.shape;n=len(x);h=np.broadcast_to(x,(r,n))
        jac=np.zeros((r,n,d)) if with_jacobian else None;slopes=[]
        for j in range(d):
            z=h+bank[:,j,None]
            slope=base.derivative(z);slopes.append(slope)
            if with_jacobian:jac*=slope[:,:,None];jac[:,:,j]+=slope
            h=base.g(z)
        raw=h-v;residual=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0)
        if with_jacobian:jac*=np.abs(raw[:,:,None])>base.EPS
        start=time.perf_counter();self.steps+=1
        if self.cfg['credit_mode']=='native':
            alpha=np.empty((r,d,n));alpha[:,-1]=-residual
            for j in range(d-2,-1,-1):alpha[:,j]=slopes[j+1]*alpha[:,j+1]
            if self.history is None:self.history=np.zeros_like(alpha)
            else:self.history+=.5*self.previous
            self.previous=alpha.copy()
        if self.steps in self.checkpoints:
            residual_only=np.zeros((r,d,n));residual_only[:,-1]=-residual
            self.add(residual_only,'residual',self.steps)
            if self.cfg['credit_mode']=='native':
                self.add(alpha,'current_bp',self.steps);self.add(self.history,'history_bp',self.steps)
                self.add(alpha+self.history,'combined_bp',self.steps)
        self.seconds+=time.perf_counter()-start
        return .5*np.mean(residual**2,axis=1),residual,jac,raw

    def directions(self):
        start=time.perf_counter();shape=(4,len(self.x));limit=self.cfg['direction_budget']
        values=np.concatenate(self.values) if self.values else np.empty((0,*shape))
        flat=values.reshape(len(values),int(np.prod(shape)))
        norms=np.linalg.norm(flat,axis=1);keep=norms>1e-14;flat=flat[keep];norms=norms[keep]
        if not len(flat):return np.empty((0,*shape)),dict(selection_seconds=time.perf_counter()-start,retained_rows=0,selected_rows=0,snapshot_events=self.events)
        unit=flat/norms[:,None]
        # Farthest-point traversal in signed normalized credit space. It uses
        # no H2 outcomes, region volumes, LP solutions, or query quantities.
        chosen=[int(np.argmax(norms))];distance=np.sum((unit-unit[chosen[0]])**2,axis=1)
        while len(chosen)<min(limit,len(unit)):
            index=int(np.argmax(distance))
            if distance[index]<=1e-24:break
            chosen.append(index);distance=np.minimum(distance,np.sum((unit-unit[index])**2,axis=1));distance[chosen]=0
        bank=flat[chosen].reshape(-1,*shape);bank/=np.max(abs(bank),axis=(1,2),keepdims=True)
        return bank,dict(selection_seconds=time.perf_counter()-start,retained_rows=len(values),nonzero_rows=len(flat),selected_rows=len(bank),
            retained_snapshot_bytes=sum(v.nbytes for v in self.values),selected_bank_bytes=bank.nbytes,
            snapshot_events=self.events,collection_seconds=self.seconds,checkpoints=sorted(self.checkpoints))


def discover(x,v,cfg):
    observer=Snapshots(x,v,cfg);mode=cfg['credit_mode'];active=mode in ['native','residual']
    core=neighbor.previous.core;old_local=core.contextual.refine_local;old_pc=neighbor.previous.refine_pc;old_eval=core.batched.evaluate
    def local(starts,x,v,anchor,sweeps=16,dual_rate=.5):
        learner='nodual' if dual_rate==0 else 'alm'
        best=trace.refine(starts,x,v,anchor,learner,sweeps,observer)
        r,d=starts.shape
        return best,dict(local_sweeps=sweeps,dual_rate=dual_rate,refinement_major_arrays_bytes=(2*r*d+2*d*r*len(x))*8)
    def pc(starts,x,v,anchor,sweeps=80):
        best=trace.refine(starts,x,v,anchor,'pc',sweeps,observer);r,d=starts.shape
        return best,dict(pc_gradient_sweeps=sweeps,dual_rate=0.,refinement_major_arrays_bytes=(2*r*d+d*r*len(x))*8)
    try:
        if active:
            core.contextual.refine_local=local;neighbor.previous.refine_pc=pc
            if cfg['learner']=='adam60':core.batched.evaluate=observer.evaluate
        bank,meta=neighbor.discover(x,v,None,cfg)
    finally:core.contextual.refine_local=old_local;neighbor.previous.refine_pc=old_pc;core.batched.evaluate=old_eval
    if active:directions,extra=observer.directions()
    elif mode=='axes':
        directions=np.r_[np.eye(4*len(x)),-np.eye(4*len(x))].reshape(-1,4,len(x))[:cfg['direction_budget']]
        extra=dict(selected_rows=len(directions),selected_bank_bytes=directions.nbytes)
    elif mode=='random':
        directions=np.random.default_rng(938111).normal(size=(cfg['direction_budget'],4,len(x)))
        directions/=np.max(abs(directions),axis=(1,2),keepdims=True);extra=dict(selected_rows=len(directions),selected_bank_bytes=directions.nbytes)
    else:directions=np.empty((0,4,len(x)));extra=dict(selected_rows=0,selected_bank_bytes=0)
    return bank,dict(meta,credit_snapshot=extra),directions


def prepare(x,v,cfg):
    saved_discover=neighbor.discover;saved_contract=neighbor.previous.screen.contract
    bank_holder=[];proofs=[];calls=[];discovery_hash=[]
    def wrapped_discover(x,v,state,cfg):
        assert state is None
        # discover() itself calls neighbor.discover; restore only during that
        # call, while retaining the independent H2 driver's scoped adapter.
        neighbor.discover=saved_discover
        try:bank,meta,directions=discover(x,v,cfg)
        finally:neighbor.discover=wrapped_discover
        bank_holder.append(directions);discovery_hash.append(hashlib.sha256(bank.tobytes()).hexdigest())
        return bank,meta
    def contract(x,v,regs,rounds=5):
        start=time.perf_counter();rounds=5 if cfg['credit_mode']=='c5' else 20
        mask=saved_contract(x,v,regs,rounds);ids=np.flatnonzero(~mask)
        preliminary=time.perf_counter()-start
        directions=bank_holder[0];p=[];detail=dict(seconds=0.,exact_checks=0)
        if len(ids) and len(directions):
            reject,p,detail=certificate.screen_bank(x,v,regs[ids],directions);mask[ids[reject]]=True
            for item in p:item['original_index']=int(ids[item['index']]);proofs.append(item)
        calls.append(dict(input_cells=len(regs),contract_rounds=rounds,contract_rejected=len(regs)-len(ids),
            credit_rejected=len(p),contraction_seconds=preliminary,credit=detail,
            input_pattern_hash=hashlib.sha256(regs.tobytes()).hexdigest()))
        return mask
    try:
        neighbor.discover=wrapped_discover;neighbor.previous.screen.contract=contract
        data,meta=prior.h2_prepare(x,v,cfg)
    finally:neighbor.discover=saved_discover;neighbor.previous.screen.contract=saved_contract
    return data,dict(meta,learner=cfg['learner'],credit_mode=cfg['credit_mode'],screen_calls=calls,
        credit_proofs=proofs,discovery_bank_sha256=discovery_hash[0],credit_bank=bank_holder[0].tolist())


def fit(x,v,state,cfg,rng,count=2048):
    if state is not None:return prior.fit(x,v,state,cfg,rng,count)
    data,meta=prepare(x,v,cfg);points,labels,sampling=prior.hybrid.sample(data,count,rng)
    anchor=points[0].copy();anchor.setflags(write=False);points.setflags(write=False);state=prior.State(anchor,points)
    return prior.posterior.make_predict(points),state,dict(meta,sampling=sampling,previous_state_digest=None,
        new_state_digest=prior.state_digest(state),persistent_state_bytes=anchor.nbytes+points.nbytes,
        actual_mode_keys=sorted({data['regs'][int(i)].tobytes().hex() for i in labels}))


def verify():
    rng=np.random.default_rng(5900001);b=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=base.forward(xx,b)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4];q=np.linspace(0,1,61)
    cases=0;proof_count=0;bp_exact=0
    for learner in ['alm','adam60','pc','nodual','direct4096']:
        baseline=config(learner,'c5');bank,_=neighbor.discover(x,v,None,baseline)
        reference,rm=prior.h2_prepare(x,v,baseline)
        points,labels,_=prior.hybrid.sample(reference,64,np.random.default_rng(482011))
        for cfg in [c for c in configs() if c['learner']==learner]:
            data,meta=prepare(x,v,cfg)
            assert meta['discovery_bank_sha256']==hashlib.sha256(bank.tobytes()).hexdigest()
            assert meta['positive_mode_keys']==rm['positive_mode_keys']
            assert meta['completion_proposals']==rm['completion_proposals']
            pp,ll,_=prior.hybrid.sample(data,64,np.random.default_rng(482011))
            assert np.array_equal(pp,points) and np.array_equal(ll,labels);cases+=1;proof_count+=len(meta['credit_proofs'])
    cfg=config('adam60','native');obs=Snapshots(x,v,cfg);bank=rng.uniform(-.12,.12,(7,4))
    for jac in [True,False]:
        expected=neighbor.previous.core.batched.evaluate(bank,x,v,jac);actual=obs.evaluate(bank,x,v,jac)
        for a,b in zip(expected,actual):assert (a is None and b is None) or np.array_equal(a,b)
        bp_exact+=1
    core=neighbor.previous.core;jac=base.forward_jacobian;refine=core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP in light local H2')
    try:
        base.forward_jacobian=forbidden;core.batched.refine=forbidden
        state=None
        for n in [4,8,16,24]:
            _,state,meta=fit(xx[:n],vv[:n],state,config('alm','native'),np.random.default_rng(482011+n),64)
            assert np.max(abs(prior.capture.model.light.forward_many(xx[:n],state.samples)[1][:,-1]-vv[:n]))<=.001+1e-8
    finally:base.forward_jacobian=jac;core.batched.refine=refine
    return dict(passed=True,discovery_and_h2_positive_set_and_samples_bitwise=cases,positive_proofs=proof_count,
        original_bp_evaluator_exact=bp_exact,local_four_stage_no_global_bp=True,
        scope='old seed preflight only; no efficacy or speed claim')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
