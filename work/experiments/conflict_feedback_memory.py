"""Causal local clause feedback; same enhanced mechanism available to BP/PC."""
import time
import numpy as np
import retained_credit_memory as old
import causal_conflict_observer as causal
import conflict_local_repair as repair
base=old.base;conflict=repair.conflict
original=old.Collector


class Collector(original):
    def __init__(self,x,v,kind,trace=False,mode='watch',capture_repairs=False):
        super().__init__(x,v,kind,trace);self.feedback_mode=mode;self.feedback_bank=causal.Bank(x,v,True);self.callback=0;self.screen_cache={}
        self.screen_seconds=0.;self.repair_seconds=0.;self.feedback_seconds=0.;self.capture_repairs=capture_repairs;self.repair_records=[]
        self.repairs_triggered=0;self.repairs_accepted=0;self.repair_candidates=0;self.activity_repairs=0;self.bias_repairs=0;self.max_energy_increase=0.;self.max_temporary_numeric_bytes=0

    def activity(self,b,h,u=None,step=0,phase=''):
        self.callback+=1;begin=time.perf_counter();d,r,n=h.shape
        if self.feedback_mode in ['repair','energy_only'] and self.feedback_bank.clauses and phase!='initial':
            regs=repair.split_many(self.x,b,h.transpose(1,0,2));hit=conflict.clause_mask(regs,self.feedback_bank.clauses)
            for i in np.flatnonzero(hit):
                before_b=b[i].copy();before_h=h[:,i].copy();uu=np.zeros_like(before_h) if u is None else u[:,i].copy()
                bb,hh,meta=repair.repair_one(self.x,before_b,before_h,uu,self.feedback_bank.clauses,enforce=self.feedback_mode!='energy_only')
                self.repairs_triggered+=int(meta['triggered']);self.repairs_accepted+=int(meta['accepted']);self.repair_candidates+=meta['candidates']
                self.max_temporary_numeric_bytes=max(self.max_temporary_numeric_bytes,meta['candidates']*(d+d*n+1)*8)
                if meta['accepted']:
                    assert np.array_equal(hh[-1],before_h[-1]) and np.max(np.abs(bb))<=conflict.screen.B and np.all((hh>=0)&(hh<=1))
                    assert max(p['accepted_event'] for p in self.feedback_bank.proofs)<self.callback
                    b[i]=bb;h[:,i]=hh;self.activity_repairs+=meta['selected_kind']=='activity';self.bias_repairs+=meta['selected_kind']=='bias'
                    self.max_energy_increase=max(self.max_energy_increase,meta['objective_after']-meta['objective_before'])
                if self.capture_repairs:
                    self.repair_records.append(dict(event=self.callback,step=int(step),phase=phase,restart=int(i),old_proofs=len(self.feedback_bank.proofs),
                        before_b=before_b.tolist(),before_h=before_h.tolist(),u=uu.tolist(),after_b=bb.tolist(),after_h=hh.tolist(),result=meta))
        self.repair_seconds+=time.perf_counter()-begin
        # Original collector sees the actual modified state, not a stale mode.
        super().activity(b,h,u,step,phase);begin=time.perf_counter();prev=np.broadcast_to(self.x,(r,n));codes=[];res=[]
        for j in range(d):
            z=prev+b[:,j,None];codes.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8));res.append(h[j]-base.g(z));prev=h[j]
        regs=np.concatenate([self.codes,np.array(codes).transpose(1,0,2)]);res=np.array(res).transpose(1,0,2);credits=[('residual',res)]
        if self.kind=='local' and u is not None and np.any(u):
            raw=u.transpose(1,0,2);credits.extend([('raw',raw),('augmented',raw+res)])
        elif self.kind=='bp':credits.extend([('current_bp',self.current),('history_bp',self.history),('combined_bp',self.current+self.history)])
        start=time.perf_counter();keys=[z.tobytes() for z in regs];new=list(dict.fromkeys(k for k in keys if k not in self.screen_cache))
        if new:
            rr=np.array([np.frombuffer(k,np.uint8).reshape(d,n) for k in new]);mask=conflict.screen.contract(self.x,self.v,rr,20)
            for key,reject in zip(new,mask):self.screen_cache[key]=bool(reject)
        c20=np.array([self.screen_cache[k] for k in keys]);self.screen_seconds+=time.perf_counter()-start
        covered=conflict.clause_mask(regs,self.feedback_bank.clauses) if self.feedback_bank.clauses else np.zeros(len(regs),bool)
        self.feedback_bank.learn(regs,credits,c20,covered,self.callback,int(step),phase);self.feedback_seconds+=time.perf_counter()-begin

    def feedback_metadata(self):
        return dict(kind=self.kind,mode=self.feedback_mode,callbacks=self.callback,clauses=len(self.feedback_bank.proofs),
            repairs_triggered=self.repairs_triggered,repairs_accepted=self.repairs_accepted,unrepaired_triggers=self.repairs_triggered-self.repairs_accepted,
            candidate_points=self.repair_candidates,activity_repairs=int(self.activity_repairs),bias_repairs=int(self.bias_repairs),max_energy_increase=self.max_energy_increase,
            repair_seconds=self.repair_seconds,feedback_seconds=self.feedback_seconds,screen_seconds=self.screen_seconds,
            exact_learning_seconds=self.feedback_bank.learning_seconds,screen_cache_patterns=len(self.screen_cache),
            library_numeric_bytes=sum(np.array(p['p']).nbytes+np.array(p['a']).nbytes+len(p['positions'])*16 for p in self.feedback_bank.proofs)+sum(t.nbytes for t in self.feedback_bank.tables),
            max_candidate_arrays_numeric_subtotal=self.max_temporary_numeric_bytes,proofs=self.feedback_bank.proofs,
            scope='numeric subtotals omit Fraction objects and Python allocation; feedback alters actual state, watch is read-only')


def prepare(x,v,cfg,trace=False):
    saved=old.Collector
    def factory(x,v,kind,trace=False):return Collector(x,v,kind,trace,cfg['feedback_mode'],cfg.get('capture_repairs',False))
    try:old.Collector=factory;bank,regs,meta,collectors=old.prepare(x,v,cfg,trace)
    finally:old.Collector=saved
    return bank,regs,dict(**meta,feedback_active=True,feedback_mode=cfg['feedback_mode'],feedback_details=[c.feedback_metadata() for c in collectors]),collectors


def fit(x,v,state,cfg):
    if state is not None or cfg.get('passthrough',False) or cfg.get('feedback_mode','none')=='none':
        pred,new,meta=old.fit(x,v,state,cfg);return pred,new,dict(**meta,feedback_active=False)
    bank,regs,meta,_=prepare(x,v,cfg);cap=cfg.get('geometry_budget');evaluated=regs if cap is None else regs[:cap]
    pred,new,detail=old.old.old.old.materialize_retained(x,v,bank,evaluated,cfg.get('posterior_samples',512))
    return pred,new,dict(**meta,**detail,geometry_budget=cap,evaluated_pattern_keys=[r.tobytes().hex() for r in evaluated],skipped_for_budget=len(regs)-len(evaluated))


def verify():
    rng=np.random.default_rng(5900001);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(base.forward(xx,truth)+np.random.default_rng(24900001).uniform(-base.EPS,base.EPS,24))[:4]
    checks=0;repairs=0;proofs=0
    for gen,kind,sweeps in [('alm','local',16),('alm','residual',16),('alm','bp',16),('nodual','residual',16),('pc','residual',20),('adam','bp',16)]:
        cfg=dict(generator=gen,credits=[kind],sweeps=sweeps,steps=sweeps,restarts=8,retention_mode='bank');before=old.prepare(x,v,cfg,True);watch=prepare(x,v,dict(**cfg,feedback_mode='watch'),True);assert causal.compare(before,watch);checks+=1
        audited=causal.capture(x,v,cfg)
        for a,b in zip(watch[3],audited[3]):
            original=b.banks[0].proofs
            assert len(a.feedback_bank.proofs)==len(original)
            for p,q in zip(a.feedback_bank.proofs,original):
                for key in p:
                    if key not in ['first_future_clause_event','first_future_full_event','future_clause_visits']:assert p[key]==q[key],key
        _,_,_,collectors=prepare(x,v,dict(**cfg,feedback_mode='repair',capture_repairs=True),True)
        for obs in collectors:
            proofs+=len(obs.feedback_bank.proofs)
            for record in obs.repair_records:
                clauses=obs.feedback_bank.clauses[:record['old_proofs']];b=np.array(record['before_b']);h=np.array(record['before_h']);u=np.array(record['u'])
                nb,nh,meta=repair.repair_one(x,b,h,u,clauses)
                assert np.array_equal(nb,np.array(record['after_b'])) and np.array_equal(nh,np.array(record['after_h'])) and meta==record['result']
                assert all(p['accepted_event']<record['event'] for p in obs.feedback_bank.proofs[:record['old_proofs']]);repairs+=1
    jac=base.forward_jacobian;bp=old.old.old.old.old.core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP entered pure local feedback')
    try:
        base.forward_jacobian=forbidden;old.old.old.old.old.core.batched.refine=forbidden
        fit(x,v,None,dict(generator='alm',sweeps=16,restarts=8,credits=['local'],retention_mode='bank',feedback_mode='repair'))
    finally:base.forward_jacobian=jac;old.old.old.old.old.core.batched.refine=bp
    return dict(passed=True,watch_trajectory_and_state_bitwise_cases=checks,watch_causal_proofs_identical=True,repair_records_replayed=repairs,new_dynamic_proofs=proofs,
        local_feedback_no_global_bp=True,primitive=repair.verify(),scope='preflight, not task superiority or timing result')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
