"""Read-only causal clause banks; no future directions or query information."""
import hashlib,time
import numpy as np
import retained_credit_memory as model
import credit_conflict as conflict
base=model.base;normal=model.normal
original=model.Collector


class Bank:
    def __init__(self,x,v,gate):
        self.x=x;self.v=v;self.gate=gate;self.clauses=[];self.tables=[];self.constants=[];self.rationals=[]
        self.proofs=[];self.events=[];self.hit_records={};self.exact_full_cache={};self.seen_clause=set()
        self.match_seconds=0.;self.full_seconds=0.;self.learning_seconds=0.;self.exact_proposals=0;self.refinements=0

    def observe(self,regs,c20,event,step,phase):
        start=time.perf_counter();n=len(regs);keys=[r.tobytes() for r in regs];c=conflict.clause_mask(regs,self.clauses) if self.clauses else np.zeros(n,bool)
        self.match_seconds+=time.perf_counter()-start;start=time.perf_counter();f=c.copy();witness=np.full(n,-1,int)
        if self.clauses:
            flat=regs.reshape(n,-1);table=np.array(self.tables);const=np.array(self.constants)
            # Explicit broadcasting: directions x rows x coordinates.
            values=table[np.arange(len(table))[:,None,None],np.arange(flat.shape[1])[None,None,:],flat[None]].sum(2)+const[:,None]
            for index in range(n):
                for ci in np.argsort(-values[:,index],kind='stable'):
                    if values[ci,index]<=1e-12:break
                    cachekey=int(ci),keys[index]
                    if cachekey not in self.exact_full_cache:
                        cc,tt=self.rationals[int(ci)];self.exact_full_cache[cachekey]=conflict.value(cc,tt,regs[index])>0
                    if self.exact_full_cache[cachekey]:f[index]=True;witness[index]=int(ci);break
        self.full_seconds+=time.perf_counter()-start
        for index in np.flatnonzero(c|f):
            key=keys[index];entry=self.hit_records.setdefault(key,dict(first_event=event,first_step=step,first_phase=phase,clause_visits=0,full_visits=0,
                beyond_c20_clause_visits=0,beyond_c20_full_visits=0,forward_clause_visits=0,forward_full_visits=0,cross_parent_clause=False,cross_parent_full=False))
            if c[index]:
                matches=[k for k,cl in enumerate(self.clauses) if np.all(regs[index].ravel()[cl['positions']]==cl['codes'])]
                assert matches
                for k in matches:
                    proof=self.proofs[k];assert proof['accepted_event']<event
                    if proof['first_future_clause_event'] is None:proof['first_future_clause_event']=event
                    if proof['first_future_full_event'] is None:proof['first_future_full_event']=event
                    proof['future_clause_visits']+=1
                entry['clause_visits']+=1;entry['beyond_c20_clause_visits']+=int(not c20[index]);entry['forward_clause_visits']+=int(index<n//2)
                entry['cross_parent_clause']|=any(self.proofs[k]['parent']!=key.hex() for k in matches)
                entry['cross_parent_full']|=any(self.proofs[k]['parent']!=key.hex() for k in matches)
            if f[index]:
                entry['full_visits']+=1;entry['beyond_c20_full_visits']+=int(not c20[index]);entry['forward_full_visits']+=int(index<n//2)
                k=int(witness[index])
                if k>=0:
                    proof=self.proofs[k];assert proof['accepted_event']<event
                    if proof['first_future_full_event'] is None:proof['first_future_full_event']=event
                    entry['cross_parent_full']|=proof['parent']!=key.hex()
        self.events.append(dict(event=event,step=step,phase=phase,visits=n,existing_clauses=len(self.clauses),
            clause_hits=int(c.sum()),full_hits=int(f.sum()),clause_beyond_c20=int(np.sum(c&~c20)),full_beyond_c20=int(np.sum(f&~c20)),
            forward_clause_hits=int(c[:n//2].sum()),forward_full_hits=int(f[:n//2].sum())))
        return c

    def learn(self,regs,credits,c20,covered,event,step,phase):
        if phase=='initial':return
        start=time.perf_counter();proposals=[]
        for label,a in credits:
            if not np.any(a):continue
            aa=np.concatenate([a,a],axis=0);values=normal.float_optimum(self.x,self.v,regs,aa);scale=1+np.abs(aa).sum((1,2))
            candidates=(values>1e-10*scale)&~covered
            if self.gate:candidates&=~c20
            for index in np.flatnonzero(candidates):proposals.append((regs[index].tobytes(),label,int(index),aa[index]))
        proposals.sort(key=lambda p:p[:3])
        for key,label,index,a in proposals:
            reg=regs[index];exact=normal.exact_optimum(self.x,self.v,reg,a);self.exact_proposals+=1
            if not exact['positive']:continue
            p,_,_=conflict.normal.refine(self.x,self.v,reg[None],a[None],4);self.refinements+=1
            clause=conflict.extract(self.x,self.v,reg,p[0],a)
            if clause is None:continue
            token=tuple(clause['positions']),tuple(clause['codes'])
            if token in self.seen_clause:continue
            self.seen_clause.add(token);cc,tt=conflict.rational_table(self.x,self.v,p[0],a)
            self.clauses.append(clause);self.rationals.append((cc,tt));self.constants.append(float(cc));self.tables.append(np.array([[float(v) for v in row] for row in tt]))
            self.proofs.append(dict(parent=key.hex(),label=label,restart=int(index%(len(regs)//2)),view='forward' if index<len(regs)//2 else 'split',
                accepted_event=event,accepted_step=step,accepted_phase=phase,c20_detects_parent=bool(c20[index]),optimized_exact=exact,
                first_future_clause_event=None,first_future_full_event=None,future_clause_visits=0,**clause))
            break
        self.learning_seconds+=time.perf_counter()-start

    def report(self):
        c=[p for p in self.proofs if p['first_future_clause_event'] is not None];hits=list(self.hit_records.values())
        return dict(gate_c20=self.gate,clauses=len(self.clauses),clauses_with_future_hits=len(c),
            maximum_available_future_callbacks=max([len(self.events)-p['accepted_event'] for p in self.proofs],default=0),
            mean_clause_cardinality=float(np.mean([p['cardinality'] for p in self.proofs])) if self.proofs else None,
            visits=sum(e['visits'] for e in self.events),clause_hits=sum(e['clause_hits'] for e in self.events),full_hits=sum(e['full_hits'] for e in self.events),
            clause_beyond_c20=sum(e['clause_beyond_c20'] for e in self.events),full_beyond_c20=sum(e['full_beyond_c20'] for e in self.events),
            forward_clause_hits=sum(e['forward_clause_hits'] for e in self.events),forward_full_hits=sum(e['forward_full_hits'] for e in self.events),
            unique_clause_hits=sum(r['clause_visits']>0 for r in hits),unique_full_hits=sum(r['full_visits']>0 for r in hits),
            unique_clause_beyond_c20=sum(r['beyond_c20_clause_visits']>0 for r in hits),unique_full_beyond_c20=sum(r['beyond_c20_full_visits']>0 for r in hits),
            unique_cross_parent_clause=sum(r['cross_parent_clause'] for r in hits),unique_cross_parent_full=sum(r['cross_parent_full'] for r in hits),
            exact_proposals=self.exact_proposals,refinements=self.refinements,match_seconds=self.match_seconds,full_seconds=self.full_seconds,learning_seconds=self.learning_seconds,
            library_numeric_bytes=sum(np.array(p['p']).nbytes+np.array(p['a']).nbytes+len(p['positions'])*16 for p in self.proofs)+sum(t.nbytes for t in self.tables),
            state_scope='numeric subtotal excludes Fraction objects, Python caches, original solver and trajectory observer')


class Collector(original):
    def __init__(self,x,v,kind,trace=False):
        super().__init__(x,v,kind,trace);self.banks=[Bank(x,v,True),Bank(x,v,False)];self.causal_events=0;self.screen_cache={};self.screen_seconds=0.;self.screen_tested=0
        self.core_trajectory=hashlib.sha256()

    def parameter(self,b,*args,**kwargs):
        super().parameter(b,*args,**kwargs);self.core_trajectory.update(b'parameter');self.core_trajectory.update(b.tobytes())

    def activity(self,b,h,u=None,step=0,phase=''):
        super().activity(b,h,u,step,phase);self.causal_events+=1;r,d=b.shape;prev=np.broadcast_to(self.x,(r,len(self.x)));codes=[];res=[]
        self.core_trajectory.update(b'activity');self.core_trajectory.update(b.tobytes());self.core_trajectory.update(h.tobytes())
        if u is not None:self.core_trajectory.update(u.tobytes())
        for j in range(d):
            z=prev+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8));res.append(h[j]-base.g(z));prev=h[j]
        split=np.array(codes).transpose(1,0,2);res=np.array(res).transpose(1,0,2);regs=np.concatenate([self.codes,split]);credits=[('residual',res)]
        if self.kind=='local' and u is not None and np.any(u):
            raw=u.transpose(1,0,2);credits.extend([('raw',raw),('augmented',raw+res)])
        elif self.kind=='bp':credits.extend([('current_bp',self.current),('history_bp',self.history),('combined_bp',self.current+self.history)])
        start=time.perf_counter();keys=[a.tobytes() for a in regs];new=list(dict.fromkeys(k for k in keys if k not in self.screen_cache))
        if new:
            rr=np.array([np.frombuffer(k,np.uint8).reshape(d,len(self.x)) for k in new]);c5=conflict.screen.contract(self.x,self.v,rr,5);c20=conflict.screen.contract(self.x,self.v,rr,20)
            for key,a,z in zip(new,c5,c20):self.screen_cache[key]=(bool(a),bool(z))
            self.screen_tested+=len(new)
        c20=np.array([self.screen_cache[k][1] for k in keys]);self.screen_seconds+=time.perf_counter()-start
        for bank in self.banks:
            covered=bank.observe(regs,c20,self.causal_events,int(step),phase);bank.learn(regs,credits,c20,covered,self.causal_events,int(step),phase)


def capture(x,v,cfg):
    saved=model.Collector
    try:model.Collector=Collector;return model.prepare(x,v,cfg,True)
    finally:model.Collector=saved


def compare(left,right):
    assert np.array_equal(left[0],right[0]) and np.array_equal(left[1],right[1])
    for field in ['pre_retention_pattern_keys','post_matching_pattern_keys','post_retention_pattern_keys']:assert left[2][field]==right[2][field]
    for a,b in zip(left[3],right[3]):
        assert a.trajectory.hexdigest()==b.trajectory.hexdigest() and a.forward==b.forward and a.split==b.split and list(a.pending)==list(b.pending) and set(a.retained)==set(b.retained)
        for k in a.pending:assert a.pending[k][0]==b.pending[k][0] and np.array_equal(a.pending[k][1],b.pending[k][1]) and a.pending[k][2]==b.pending[k][2]
        for k,item in a.retained.items():
            other=b.retained[k];assert np.array_equal(item['a'],other['a']);assert {k:v for k,v in item.items() if k!='a'}=={k:v for k,v in other.items() if k!='a'}
    return True


def verify():
    seed=5900000;rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4];checks=0
    for gen,kind,steps in [('alm','local',3),('adam','bp',3),('pc','residual',3)]:
        cfg=dict(generator=gen,sweeps=steps,steps=steps,restarts=8,credits=[kind],retention_mode='bank');before=model.prepare(x,v,cfg,True);after=capture(x,v,cfg);assert compare(before,after);checks+=1
    jac=base.forward_jacobian;bp=model.old.old.old.old.core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP entered local causal observation')
    try:
        base.forward_jacobian=forbidden;model.old.old.old.old.core.batched.refine=forbidden
        capture(x,v,dict(generator='alm',sweeps=3,restarts=8,credits=['local'],retention_mode='bank'))
    finally:base.forward_jacobian=jac;model.old.old.old.old.core.batched.refine=bp
    return dict(passed=True,original_trajectory_and_all_state_bitwise_cases=checks,no_global_bp_in_local=True,strictly_future_hits=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
