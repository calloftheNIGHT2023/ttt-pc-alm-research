"""Read-only per-pattern credit retention before the old positive-bound gate."""
import hashlib,time
import numpy as np
import optimized_credit_memory as model
original=model.old.old.Collector
base=model.base
normal=model.normal


class TracedCollector(original):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.trajectory=hashlib.sha256()

    def parameter(self,b,*args,**kwargs):
        super().parameter(b,*args,**kwargs);self.trajectory.update(b'parameter');self.trajectory.update(b.tobytes())

    def activity(self,b,h,u=None,step=0,phase=''):
        super().activity(b,h,u,step,phase)
        self.trajectory.update(b'activity');self.trajectory.update(b.tobytes());self.trajectory.update(h.tobytes())
        if u is not None:self.trajectory.update(u.tobytes())
        if self.current is not None:self.trajectory.update(self.current.tobytes());self.trajectory.update(self.history.tobytes())


class RetainingCollector(TracedCollector):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.retained={};self.retention_seconds=0.;self.retention_updates=0;self.nonzero_rows=0

    def activity(self,b,h,u=None,step=0,phase=''):
        super().activity(b,h,u,step,phase);begin=time.perf_counter();r,d=b.shape
        prev=np.broadcast_to(self.x,(r,len(self.x)));codes=[];residual=[]
        for j in range(d):
            z=prev+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8));residual.append(h[j]-base.g(z));prev=h[j]
        split=np.array(codes).transpose(1,0,2);res=np.array(residual).transpose(1,0,2);credits=[('residual',res)]
        if self.kind=='local' and u is not None and np.any(u):
            raw=u.transpose(1,0,2);credits.extend([('raw',raw),('augmented',raw+res)])
        elif self.kind=='bp':credits.extend([('current_bp',self.current),('history_bp',self.history),('combined_bp',self.current+self.history)])
        regs=np.concatenate([self.codes,split],axis=0)
        for label,a in credits:
            if not np.any(a):continue
            aa=np.concatenate([a,a],axis=0);values,scale=self.rough(regs,aa)
            # Zero is not an informative direction and would dominate every
            # negative score. Nonzero exclusion is declared before diagnosis.
            for i in np.flatnonzero(np.any(aa!=0,axis=(1,2))):
                key=(regs[i].tobytes(),label);ratio=float(values[i]/scale[i]);self.nonzero_rows+=1
                if key not in self.retained or ratio>self.retained[key]['ratio']:
                    self.retained[key]=dict(ratio=ratio,a=a[i%r].copy(),rough=float(values[i]),step=int(step),phase=phase);self.retention_updates+=1
        self.retention_seconds+=time.perf_counter()-begin


def capture(x,v,cfg,retain):
    collectors=[];saved=model.old.old.Collector
    def factory(x,v,kind):
        instance=(RetainingCollector if retain else TracedCollector)(x,v,kind);collectors.append(instance);return instance
    try:
        model.old.old.Collector=factory;bank,regs,meta=model.old.prepare(x,v,cfg)
    finally:model.old.old.Collector=saved
    directions=[];seen=set()
    for collector in collectors:
        accepted,_=collector.finalize()
        assert [k.hex() for k in accepted]==meta['credit_details'][collector.kind]['matching_keys']
        assert [v[0] for v in accepted.values()]==meta['credit_details'][collector.kind]['matching_bounds']
        for a in model.old.old.direction_bank(accepted,32):
            if a.tobytes() not in seen:seen.add(a.tobytes());directions.append(a)
    directions=np.array(directions).reshape(-1,4,len(x))
    rejected,proofs,cost=normal.screen_bank(x,v,regs,directions)
    return bank,regs[~rejected],meta,collectors,cost


def extra_matching(x,v,regs,collectors):
    begin=time.perf_counter();rr=[];aa=[];records=[];owners=[]
    for index,reg in enumerate(regs):
        key=reg.tobytes();seen=set()
        for collector in collectors:
            for (pattern,label),item in collector.retained.items():
                if pattern!=key or item['a'].tobytes() in seen:continue
                seen.add(item['a'].tobytes());rr.append(reg);aa.append(item['a']);owners.append(index)
                records.append(dict(collector=collector.kind,label=label,old_rough=item['rough'],old_normalized_score=item['ratio'],step=item['step'],phase=item['phase']))
    reject=np.zeros(len(regs),bool);proofs=[];best={};checks=0
    if rr:
        rr=np.array(rr);aa=np.array(aa);values=normal.float_optimum(x,v,rr,aa);scale=1+np.abs(aa).sum((1,2))
        for k in np.flatnonzero(values>1e-10*scale):
            index=owners[k]
            if index not in best or values[k]>values[best[index]]:best[index]=int(k)
        for index,k in best.items():
            exact=normal.exact_optimum(x,v,rr[k],aa[k]);checks+=1
            if not exact['positive']:continue
            reject[index]=True;proofs.append(dict(index=index,pattern=rr[k].tobytes().hex(),a=aa[k].tolist(),rough_optimized_value=float(values[k]),exact=exact,**records[k]))
    return reject,proofs,dict(seconds=time.perf_counter()-begin,matching_direction_pairs=len(records),exact_checks=checks,
        retained_entries=sum(len(c.retained) for c in collectors),retained_numeric_bytes=sum(len(key[0])+8+item['a'].nbytes for c in collectors for key,item in c.retained.items()),
        retention_seconds=sum(c.retention_seconds for c in collectors),retention_updates=sum(c.retention_updates for c in collectors),nonzero_rows=sum(c.nonzero_rows for c in collectors),
        scope='read-only diagnostic capture; numeric subtotal omits label keys and Python objects; matching lookup not optimized')


def verify():
    seed=5900000;rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24);checks=0
    for gen,kind in [('alm','local'),('alm','residual'),('adam','bp')]:
        cfg=dict(generator=gen,credits=[kind],sweeps=16,steps=16,restarts=64)
        before=capture(x[:4],v[:4],cfg,False);after=capture(x[:4],v[:4],cfg,True)
        assert np.array_equal(before[0],after[0]) and np.array_equal(before[1],after[1])
        for a,b in zip(before[3],after[3]):
            assert a.trajectory.hexdigest()==b.trajectory.hexdigest();assert a.forward==b.forward and a.split==b.split
            assert list(a.pending)==list(b.pending)
            for key in a.pending:assert a.pending[key][0]==b.pending[key][0] and np.array_equal(a.pending[key][1],b.pending[key][1])
        checks+=1
    oldjac=base.forward_jacobian;oldbp=model.old.old.old.core.batched.refine
    def forbidden(*args,**kwargs):raise AssertionError('global BP in local credit capture')
    try:
        base.forward_jacobian=forbidden;model.old.old.old.core.batched.refine=forbidden
        _,regs,_,collectors,_=capture(x[:4],v[:4],dict(generator='alm',credits=['local'],sweeps=16,restarts=64),True)
        extra_matching(x[:4],v[:4],regs,collectors)
    finally:base.forward_jacobian=oldjac;model.old.old.old.core.batched.refine=oldbp
    return dict(passed=True,trajectory_bank_original_pending_bitwise_cases=checks,local_capture_no_global_bp=True,zero_directions_excluded=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
