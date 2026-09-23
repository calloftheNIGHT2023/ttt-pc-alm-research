"""Fused actual online credit retention, exact matching and safe cross reuse."""
import hashlib,time
import numpy as np
import optimized_credit_memory as old
original=old.old.old.Collector
base=old.base
normal=old.normal


class Collector(original):
    def __init__(self,x,v,kind,trace=False):
        super().__init__(x,v,kind);self.retained={};self.retention_seconds=0.;self.retention_updates=0;self.nonzero_rows=0
        self.trajectory=hashlib.sha256() if trace else None

    def parameter(self,b,*args,**kwargs):
        super().parameter(b,*args,**kwargs)
        if self.trajectory is not None:self.trajectory.update(b'parameter');self.trajectory.update(b.tobytes())

    def activity(self,b,h,u=None,step=0,phase=''):
        begin=time.perf_counter();self.ensure_forward(b);r,d=b.shape;prev=np.broadcast_to(self.x,(r,len(self.x)));codes=[];residual=[]
        for j in range(d):
            z=prev+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8));residual.append(h[j]-base.g(z));prev=h[j]
        split=np.array(codes).transpose(1,0,2)
        for reg in split:self.split.setdefault(reg.tobytes(),None)
        res=np.array(residual).transpose(1,0,2);credits=[('residual',res)]
        if self.kind=='local' and u is not None and np.any(u):
            raw=u.transpose(1,0,2);credits.extend([('raw',raw),('augmented',raw+res)])
        elif self.kind=='bp':credits.extend([('current_bp',self.current),('history_bp',self.history),('combined_bp',self.current+self.history)])
        regs=np.concatenate([self.codes,split],axis=0)
        keys=np.ascontiguousarray(regs).reshape(len(regs),-1).view(np.dtype((np.void,d*len(self.x)))).ravel()
        for label,a in credits:
            if not np.any(a):continue
            aa=np.concatenate([a,a],axis=0);value,scale=self.rough(regs,aa);self.rough_tests+=len(regs);ratio=value/scale
            # Original positive pending selection and insertion order unchanged.
            for i in np.flatnonzero(value>1e-10*scale):
                key=regs[i].tobytes();score=float(ratio[i])
                if key not in self.pending or score>self.pending[key][0]:self.pending[key]=(score,a[i%r].copy(),label);self.pending_updates+=1
            start=time.perf_counter();indices=np.flatnonzero(np.any(aa!=0,axis=(1,2)));self.nonzero_rows+=len(indices)
            if len(indices):
                # Per-pattern maximum within this callback; stable first-row
                # tie reproduces the diagnostic's sequential updates exactly.
                order=np.lexsort((indices,-ratio[indices],keys[indices]));ordered=indices[order]
                selected=ordered[np.r_[True,keys[ordered[1:]]!=keys[ordered[:-1]]]]
                for i in selected:
                    key=(regs[i].tobytes(),label);score=float(ratio[i])
                    if key not in self.retained or score>self.retained[key]['ratio']:
                        self.retained[key]=dict(ratio=score,a=a[i%r].copy(),rough=float(value[i]),step=int(step),phase=phase);self.retention_updates+=1
            self.retention_seconds+=time.perf_counter()-start
        self.events+=1;self.credit_seconds+=time.perf_counter()-begin
        if self.trajectory is not None:
            self.trajectory.update(b'activity');self.trajectory.update(b.tobytes());self.trajectory.update(h.tobytes())
            if u is not None:self.trajectory.update(u.tobytes())
            if self.current is not None:self.trajectory.update(self.current.tobytes());self.trajectory.update(self.history.tobytes())

    def finalize(self):
        if not hasattr(self,'finished_credit'):self.finished_credit=super().finalize()
        return self.finished_credit


def matching(x,v,regs,collectors):
    begin=time.perf_counter();lookup={}
    for c in collectors:
        for (key,label),item in c.retained.items():lookup.setdefault(key,[]).append((c.kind,label,item))
    rr=[];aa=[];owners=[];records=[]
    for index,reg in enumerate(regs):
        seen=set()
        for kind,label,item in lookup.get(reg.tobytes(),[]):
            token=item['a'].tobytes()
            if token in seen:continue
            seen.add(token);rr.append(reg);aa.append(item['a']);owners.append(index);records.append((kind,label,item))
    reject=np.zeros(len(regs),bool);proofs=[];checks=0;best={}
    if rr:
        rr=np.array(rr);aa=np.array(aa);values=normal.float_optimum(x,v,rr,aa);scale=1+np.abs(aa).sum((1,2))
        for k in np.flatnonzero(values>1e-10*scale):
            index=owners[k]
            if index not in best or values[k]>values[best[index]]:best[index]=int(k)
        for index,k in best.items():
            exact=normal.exact_optimum(x,v,rr[k],aa[k]);checks+=1
            if not exact['positive']:continue
            kind,label,item=records[k];reject[index]=True
            proofs.append(dict(index=index,pattern=rr[k].tobytes().hex(),a=aa[k].tolist(),exact=exact,collector=kind,label=label,old_rough=item['rough'],old_normalized_score=item['ratio'],step=item['step'],phase=item['phase'],rough_optimized_value=float(values[k])))
    return reject,proofs,dict(seconds=time.perf_counter()-begin,matching_direction_pairs=len(records),exact_checks=checks)


def prepare(x,v,cfg,trace=False):
    collectors=[];saved=old.old.old.Collector
    def factory(x,v,kind):
        c=Collector(x,v,kind,trace);collectors.append(c);return c
    try:
        old.old.old.Collector=factory;bank,regs,meta=old.old.prepare(x,v,cfg)
    finally:old.old.old.Collector=saved
    directions=[];seen=set()
    for c in collectors:
        accepted,_=c.finalize()
        for a in old.old.old.direction_bank(accepted,32):
            if a.tobytes() not in seen:seen.add(a.tobytes());directions.append(a)
    directions=np.array(directions).reshape(-1,4,len(x));reject,normalproof,normalcost=normal.screen_bank(x,v,regs,directions);regs=regs[~reject]
    baseline=[r.tobytes().hex() for r in regs];reject,proofs,matchcost=matching(x,v,regs,collectors);regs=regs[~reject]
    matched=[r.tobytes().hex() for r in regs];newbank=[];seen=set();crossproof=[];crosscost={}
    if cfg.get('retention_mode','bank')=='bank':
        for proof in sorted(proofs,key=lambda p:p['pattern']):
            a=np.array(proof['a']);a/=np.max(np.abs(a));key=a.tobytes()
            if key not in seen:seen.add(key);newbank.append(a)
            if len(newbank)==32:break
        newbank=np.array(newbank).reshape(-1,4,len(x));reject,crossproof,crosscost=normal.screen_bank(x,v,regs,newbank);regs=regs[~reject]
    meta=dict(**meta,retention_active=True,normal_cost=normalcost,retained_matching_cost=matchcost,retained_cross_cost=crosscost,
        pre_retention_pattern_keys=baseline,post_matching_pattern_keys=matched,post_retention_pattern_keys=[r.tobytes().hex() for r in regs],
        retained_matching_keys=[p['pattern'] for p in proofs],retained_matching_values=[p['exact']['value'] for p in proofs],retained_cross_keys=[p['pattern'] for p in crossproof],
        retained_entries=sum(len(c.retained) for c in collectors),retained_numeric_bytes=sum(len(key[0])+8+item['a'].nbytes for c in collectors for key,item in c.retained.items()),
        retention_bookkeeping_seconds=sum(c.retention_seconds for c in collectors),retention_bookkeeping_updates=sum(c.retention_updates for c in collectors),
        extra_bank_directions=len(newbank),extra_bank_numeric_bytes=0 if isinstance(newbank,list) else newbank.nbytes)
    return bank,regs,meta,collectors


def fit(x,v,state,cfg):
    if state is not None or cfg.get('retention_mode','none')=='none' or cfg.get('passthrough',False):
        predict,new,meta=old.fit(x,v,state,cfg);return predict,new,dict(**meta,retention_active=False)
    bank,regs,meta,_=prepare(x,v,cfg);cap=cfg.get('geometry_budget');evaluated=regs if cap is None else regs[:cap]
    predict,new,detail=old.old.old.materialize_retained(x,v,bank,evaluated,cfg.get('posterior_samples',512))
    return predict,new,dict(**meta,**detail,optimized_normal=True,budget_active=cap is not None,geometry_budget=cap,evaluated_pattern_keys=[r.tobytes().hex() for r in evaluated],skipped_for_budget=len(regs)-len(evaluated))


def verify():
    import retained_credit_capture as diagnostic
    seed=5900000;rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4];cases=0
    for gen,kind in [('alm','local'),('alm','residual'),('adam','bp'),('pc','residual'),('nodual','residual')]:
        cfg=dict(generator=gen,credits=[kind],sweeps=16,steps=16,restarts=64,retention_mode='bank',optimized_normal=True)
        ref=diagnostic.capture(x,v,cfg,True);bank,regs,meta,collectors=prepare(x,v,cfg,True)
        assert np.array_equal(bank,ref[0]) and meta['pre_retention_pattern_keys']==[r.tobytes().hex() for r in ref[1]]
        for a,b in zip(collectors,ref[3]):
            assert a.trajectory.hexdigest()==b.trajectory.hexdigest() and set(a.retained)==set(b.retained) and list(a.pending)==list(b.pending)
            for key,item in a.retained.items():
                olditem=b.retained[key];assert np.array_equal(item['a'],olditem['a']);assert {k:v for k,v in item.items() if k!='a'}=={k:v for k,v in olditem.items() if k!='a'}
            for key in a.pending:assert a.pending[key][0]==b.pending[key][0] and np.array_equal(a.pending[key][1],b.pending[key][1]) and a.pending[key][2]==b.pending[key][2]
        _,s,_=fit(x,v,None,cfg);_,r,_=old.fit(x,v,None,cfg);assert np.array_equal(s.anchor,r.anchor) and np.array_equal(s.samples,r.samples);cases+=1
    beforejac=base.forward_jacobian;beforebp=old.old.old.old.core.batched.refine
    def forbidden(*args,**kwargs):raise AssertionError('global BP in fused local credit retention')
    try:
        base.forward_jacobian=forbidden;old.old.old.old.core.batched.refine=forbidden
        fit(x,v,None,dict(generator='alm',credits=['local'],sweeps=16,restarts=64,retention_mode='bank'))
    finally:base.forward_jacobian=beforejac;old.old.old.old.core.batched.refine=beforebp
    return dict(passed=True,full_trajectory_retained_original_pending_and_state_bitwise_cases=cases,pure_local_no_global_bp=True,
        stable_batch_maxima_same_selected_direction=True,scope='bookkeeping update counts intentionally collapse within-callback intermediate maxima')


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
