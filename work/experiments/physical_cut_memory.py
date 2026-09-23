"""Actual causal paths using exactly certified continuous residual cuts."""
import time
import numpy as np
import conflict_feedback_memory as feedback
import credit_residual_cut as repair

old=feedback.old;base=feedback.base;conflict=feedback.conflict


class Collector(feedback.Collector):
    def __init__(self,x,v,kind,trace=False,mode='physical',capture_repairs=False):
        # The inherited collector learns after the callback, without performing
        # its old canonical-label repair a second time.
        super().__init__(x,v,kind,trace,mode='watch',capture_repairs=False)
        self.physical_mode=mode;self.physical_capture=capture_repairs;self.physical_records=[]
        self.cut_seconds=0.;self.cut_triggered=0;self.cut_accepted=0;self.cut_candidates=0;self.cut_segments=0
        self.cut_empty_segments=0;self.cut_unrepresentable=0;self.cut_no_point=0;self.cut_checks=0;self.cut_activity=0;self.cut_bias=0

    def activity(self,b,h,u=None,step=0,phase=''):
        if self.physical_mode in ['physical','energy_only'] and self.feedback_bank.clauses and phase!='initial':
            begin=time.perf_counter();regs=feedback.repair.split_many(self.x,b,h.transpose(1,0,2))
            hit=conflict.clause_mask(regs,self.feedback_bank.clauses)
            for i in np.flatnonzero(hit):
                before_b=b[i].copy();before_h=h[:,i].copy();uu=np.zeros_like(before_h) if u is None else u[:,i].copy()
                bb,hh,meta=repair.repair_one(self.x,before_b,before_h,uu,self.feedback_bank.clauses,enforce=self.physical_mode=='physical')
                assert meta['triggered'];self.cut_triggered+=1;self.cut_accepted+=int(meta['accepted']);self.cut_candidates+=meta['candidates']
                self.cut_segments+=meta['segments'];self.cut_empty_segments+=meta['empty_segments'];self.cut_unrepresentable+=meta['unrepresentable_segments']
                self.cut_no_point+=meta.get('reason')=='no feasible representable one-coordinate point';self.cut_checks+=meta.get('exact_cut_checks',0)
                assert all(p['accepted_event']<self.callback+1 for p in self.feedback_bank.proofs)
                if meta['accepted']:
                    assert np.array_equal(hh[-1],before_h[-1]) and np.max(np.abs(bb))<=float(repair.B) and np.all((hh>=0)&(hh<=1))
                    b[i]=bb;h[:,i]=hh;self.cut_activity+=meta['selected_kind']=='activity';self.cut_bias+=meta['selected_kind']=='bias'
                if self.physical_capture:
                    self.physical_records.append(dict(event=self.callback+1,step=int(step),phase=phase,restart=int(i),old_proofs=len(self.feedback_bank.proofs),
                        before_b=before_b.tolist(),before_h=before_h.tolist(),u=uu.tolist(),after_b=bb.tolist(),after_h=hh.tolist(),result=meta))
            self.cut_seconds+=time.perf_counter()-begin
        # Actual modified activities/parameters enter the same archive and
        # generate only one new causal proof, as in the prior frozen version.
        super().activity(b,h,u,step,phase)

    def feedback_metadata(self):
        meta=super().feedback_metadata()
        return dict(**meta,physical_mode=self.physical_mode,physical_seconds=self.cut_seconds,physical_triggered=self.cut_triggered,
                    physical_accepted=self.cut_accepted,physical_candidates=self.cut_candidates,physical_segments=self.cut_segments,
                    physical_empty_segments=self.cut_empty_segments,physical_unrepresentable_segments=self.cut_unrepresentable,
                    physical_no_point=int(self.cut_no_point),physical_exact_cut_checks=self.cut_checks,
                    physical_activity=int(self.cut_activity),physical_bias=int(self.cut_bias),
                    physical_scope='all old cuts exact; fixed original canonical trigger; no feasible one-coordinate point leaves state unchanged')


def prepare(x,v,cfg,trace=False):
    saved=old.Collector
    def factory(x,v,kind,trace=False):return Collector(x,v,kind,trace,cfg['physical_mode'],cfg.get('capture_repairs',False))
    try:old.Collector=factory;bank,regs,meta,collectors=old.prepare(x,v,cfg,trace)
    finally:old.Collector=saved
    return bank,regs,dict(**meta,physical_active=True,physical_mode=cfg['physical_mode'],feedback_details=[c.feedback_metadata() for c in collectors]),collectors


def fit(x,v,state,cfg):
    if state is not None or cfg.get('passthrough',False) or cfg.get('physical_mode','none')=='none':
        pred,new,meta=old.fit(x,v,state,cfg);return pred,new,dict(**meta,physical_active=False)
    bank,regs,meta,_=prepare(x,v,cfg);cap=cfg.get('geometry_budget');evaluated=regs if cap is None else regs[:cap]
    pred,new,detail=old.old.old.old.materialize_retained(x,v,bank,evaluated,cfg.get('posterior_samples',512))
    return pred,new,dict(**meta,**detail,geometry_budget=cap,evaluated_pattern_keys=[r.tobytes().hex() for r in evaluated],skipped_for_budget=len(regs)-len(evaluated))


def verify():
    seed=5900001;rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
    v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4];checks=0;records=0;accepted=0
    for gen,kind,sweeps in [('alm','local',16),('alm','bp',16),('adam','bp',16),('pc','residual',20),('nodual','residual',16)]:
        cfg=dict(generator=gen,credits=[kind],sweeps=sweeps,steps=sweeps,restarts=8,retention_mode='bank')
        left=old.prepare(x,v,cfg,True);right=prepare(x,v,dict(**cfg,physical_mode='watch'),True);assert feedback.causal.compare(left,right);checks+=1
        _,_,_,collectors=prepare(x,v,dict(**cfg,physical_mode='physical',capture_repairs=True),True)
        for obs in collectors:
            for row in obs.physical_records:
                clauses=obs.feedback_bank.clauses[:row['old_proofs']]
                b,h,meta=repair.repair_one(x,np.array(row['before_b']),np.array(row['before_h']),np.array(row['u']),clauses)
                assert np.array_equal(b,np.array(row['after_b'])) and np.array_equal(h,np.array(row['after_h'])) and meta==row['result'];records+=1;accepted+=int(meta['accepted'])
                assert all(p['accepted_event']<row['event'] for p in obs.feedback_bank.proofs[:row['old_proofs']])
    jac=base.forward_jacobian;bp=old.old.old.old.old.core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP entered pure local physical cut adaptation')
    try:
        base.forward_jacobian=forbidden;old.old.old.old.old.core.batched.refine=forbidden
        fit(x,v,None,dict(generator='alm',sweeps=16,restarts=8,credits=['local'],retention_mode='bank',physical_mode='physical'))
    finally:base.forward_jacobian=jac;old.old.old.old.old.core.batched.refine=bp
    return dict(passed=True,watch_original_bitwise_cases=checks,dynamic_records_replayed=records,dynamic_accepted=accepted,no_global_bp_local=True,
                scope='preflight only; does not establish task superiority')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
