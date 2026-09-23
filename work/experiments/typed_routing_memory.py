"""Causal proposal side channel with unchanged original feedback trajectory.

Point updates remain the frozen original method. New modes are independently
screened and geometrically solved before contributing to predictive memory.
"""
import time
import numpy as np
import conflict_feedback_memory as feedback
import light_tied_proposals as light
from diagnose_light_tied_routing import route

old=feedback.old;base=feedback.base;conflict=feedback.conflict
original=feedback.Collector


class Collector(original):
    def __init__(self,x,v,kind,trace=False,mode='watch',capture_repairs=False,route_mode='none',route_use_dual=True):
        super().__init__(x,v,kind,trace,mode,capture_repairs)
        self.route_mode=route_mode;self.route_use_dual=route_use_dual;self.route_seen=set();self.route_kept={}
        self.route_stats=dict(candidate_mode_visits=0,distinct_modes=0,causal_clause_removed=0,c20_removed=0,routing_seconds=0.)
        self.route_counts=dict(inputs=0,candidates=0,segments=0,construction_seconds=0.,pattern_seconds=0.,trigger_seconds=0.)

    def activity(self,b,h,u=None,step=0,phase=''):
        if self.route_mode!='none' and self.feedback_bank.clauses and phase!='initial':
            start=time.perf_counter();regs=feedback.repair.split_many(self.x,b,h.transpose(1,0,2));hit=conflict.clause_mask(regs,self.feedback_bank.clauses)
            self.route_counts['trigger_seconds']+=time.perf_counter()-start
            for i in np.flatnonzero(hit):
                bb=b[i];hh=h[:,i];uu=np.zeros_like(hh) if u is None or not self.route_use_dual else u[:,i]
                assert all(p['accepted_event']<self.callback+1 for p in self.feedback_bank.proofs)
                start=time.perf_counter()
                if self.route_mode.startswith('tied'):
                    pb,ph,meta=light.tied_free(self.x,bb,hh,uu)
                else:
                    assert self.route_mode=='coordinate_forward';pb,meta=light.coordinate_scan(self.x,bb);ph=None
                self.route_counts['construction_seconds']+=time.perf_counter()-start
                self.route_counts['inputs']+=1;self.route_counts['candidates']+=meta['candidates'];self.route_counts['segments']+=meta['segments']
                start=time.perf_counter();fp,_=light.forward_many(self.x,pb)
                if self.route_mode=='tied_both':
                    sp=feedback.repair.split_many(self.x,pb,ph);proposed=np.concatenate([fp,sp])
                else:proposed=fp
                self.route_counts['pattern_seconds']+=time.perf_counter()-start
                route(self.x,self.v,proposed,self.feedback_bank.clauses,self.route_seen,self.route_kept,self.route_stats,
                      self.callback+1,int(i),len(self.feedback_bank.proofs))
        # This is the original state mutation and subsequent credit learning.
        # The proposal channel above did not mutate b, h, u or the proof bank.
        super().activity(b,h,u,step,phase)

    def feedback_metadata(self):
        result=super().feedback_metadata()
        return dict(**result,route_mode=self.route_mode,route_use_dual=self.route_use_dual,route_counts=self.route_counts,route_stats=self.route_stats,
                    route_mode_survivors=len(self.route_kept),route_key_numeric_bytes=sum(len(k) for k in self.route_seen)+sum(len(k) for k in self.route_kept),
                    route_scope='mode proposals only; key numeric subtotal omits Python set/dict allocations; full tracked resources measured separately')


def prepare(x,v,cfg,trace=False):
    saved=feedback.Collector
    def factory(x,v,kind,trace=False,mode='watch',capture_repairs=False):
        return Collector(x,v,kind,trace,mode,capture_repairs,cfg.get('route_mode','none'),cfg.get('route_use_dual',True))
    try:
        feedback.Collector=factory;bank,regs,meta,collectors=feedback.prepare(x,v,cfg,trace)
    finally:feedback.Collector=saved
    keys={r.tobytes() for r in regs};extra={}
    for c in collectors:
        for key,origin in c.route_kept.items():
            if key not in keys and key not in extra:extra[key]=origin
    cap=cfg.get('route_extra_budget');allkeys=list(extra);selected=allkeys if cap is None else allkeys[:cap]
    new=np.array([np.frombuffer(k,np.uint8).reshape(len(bank[0]),len(x)) for k in selected],np.uint8).reshape(-1,4,len(x))
    combined=np.concatenate([regs,new])
    return bank,combined,dict(**meta,routing_active=True,route_mode=cfg.get('route_mode','none'),route_extra_budget=cap,
        route_original_pattern_keys=[r.tobytes().hex() for r in regs],route_extra_all_keys=[k.hex() for k in allkeys],
        route_extra_pattern_keys=[k.hex() for k in selected],route_extra_geometry_calls=len(selected),route_extra_skipped=len(allkeys)-len(selected),
        route_extra_origins=[dict(pattern=k.hex(),**extra[k]) for k in selected]),collectors


def fit(x,v,state,cfg):
    if state is not None or cfg.get('route_mode','none')=='none' or cfg.get('passthrough',False):
        pred,new,meta=feedback.fit(x,v,state,cfg);return pred,new,dict(**meta,routing_active=False)
    bank,regs,meta,_=prepare(x,v,cfg)
    pred,new,detail=feedback.old.old.old.old.materialize_retained(x,v,bank,regs,cfg.get('posterior_samples',512))
    return pred,new,dict(**meta,**detail,evaluated_pattern_keys=[r.tobytes().hex() for r in regs],geometry_budget=None,skipped_for_budget=meta['route_extra_skipped'])


def verify():
    seed=5900001;rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
    v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4];checks=0;routed=0;extra=0
    for gen,kind,steps in [('alm','local',16),('alm','bp',16),('adam','bp',16),('pc','residual',20),('nodual','residual',16)]:
        cfg=dict(generator=gen,credits=[kind],sweeps=steps,steps=steps,restarts=8,retention_mode='bank',feedback_mode='repair')
        before=feedback.prepare(x,v,cfg,True)
        variants=[('tied_forward',True),('tied_both',True),('coordinate_forward',True)]
        if gen=='alm' and kind=='local':variants.append(('tied_forward',False))
        for variant,use_dual in variants:
            after=prepare(x,v,dict(**cfg,route_mode=variant,route_use_dual=use_dual),True)
            prefix=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,4) for k in after[2]['route_original_pattern_keys']])
            assert feedback.causal.compare(before,(after[0],prefix,after[2],after[3]));checks+=1
            for a,z in zip(before[3],after[3]):
                assert a.feedback_bank.proofs==z.feedback_bank.proofs
                assert z.route_counts['inputs']==z.repairs_triggered;routed+=z.route_counts['inputs']
                stats=z.route_stats;assert len(z.route_seen)==len(z.route_kept)+stats['causal_clause_removed']+stats['c20_removed']
            extra+=after[2]['route_extra_geometry_calls']
    jac=base.forward_jacobian;bp_module=old.old.old.old.old.core.batched;bp=bp_module.refine
    def forbidden(*a,**k):raise AssertionError('global BP entered the local typed routing mechanism')
    try:
        base.forward_jacobian=forbidden;bp_module.refine=forbidden
        fit(x,v,None,dict(generator='alm',credits=['local'],sweeps=16,restarts=8,retention_mode='bank',feedback_mode='repair',route_mode='tied_both'))
    finally:base.forward_jacobian=jac;bp_module.refine=bp
    return dict(passed=True,original_trajectory_archive_pending_retained_and_proofs_bitwise_cases=checks,old_trigger_inputs=routed,
                extra_geometries=extra,no_global_bp_local=True,scope='live side-channel preflight; task result and full costs require separate run')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
