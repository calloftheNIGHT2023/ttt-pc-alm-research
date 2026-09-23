"""Equivalent typed online memory with deferred row-local C20 routing.

The original point solver and causal proof learning are unchanged. Every
generator and comparator can use this same non-predictive optimization.
"""
import time
import numpy as np
import typed_routing_memory as original
from deferred_mode_router import Router

feedback=original.feedback;light=original.light;old=original.old;base=original.base;conflict=original.conflict


class Collector(original.Collector):
    def __init__(self,x,v,kind,trace=False,mode='watch',capture_repairs=False,route_mode='none',route_use_dual=True):
        super().__init__(x,v,kind,trace,mode,capture_repairs,route_mode,route_use_dual)
        self.deferred=Router(x,v,256)
        self.route_seen=self.deferred.seen;self.route_kept=self.deferred.kept;self.route_stats=self.deferred.stats

    def activity(self,b,h,u=None,step=0,phase=''):
        if self.route_mode!='none' and self.feedback_bank.clauses and phase!='initial':
            start=time.perf_counter();regs=feedback.repair.split_many(self.x,b,h.transpose(1,0,2));hit=conflict.clause_mask(regs,self.feedback_bank.clauses)
            self.route_counts['trigger_seconds']+=time.perf_counter()-start
            for i in np.flatnonzero(hit):
                bb=b[i];hh=h[:,i];uu=np.zeros_like(hh) if u is None or not self.route_use_dual else u[:,i]
                assert all(p['accepted_event']<self.callback+1 for p in self.feedback_bank.proofs)
                start=time.perf_counter()
                if self.route_mode.startswith('tied'):pb,ph,meta=light.tied_free(self.x,bb,hh,uu)
                else:
                    assert self.route_mode=='coordinate_forward';pb,meta=light.coordinate_scan(self.x,bb);ph=None
                self.route_counts['construction_seconds']+=time.perf_counter()-start
                self.route_counts['inputs']+=1;self.route_counts['candidates']+=meta['candidates'];self.route_counts['segments']+=meta['segments']
                start=time.perf_counter();fp,_=light.forward_many(self.x,pb)
                proposed=np.concatenate([fp,feedback.repair.split_many(self.x,pb,ph)]) if self.route_mode=='tied_both' else fp
                self.route_counts['pattern_seconds']+=time.perf_counter()-start
                self.deferred.observe(proposed,self.feedback_bank.clauses,self.callback+1,int(i),len(self.feedback_bank.proofs))
        # Deliberately call the original feedback mutation, not the eager
        # typed proposal callback (which would generate a second candidate pool).
        original.original.activity(self,b,h,u,step,phase)

    def feedback_metadata(self):
        result=super().feedback_metadata();d=self.deferred
        return dict(**result,deferred_route_flushed=d.flushed,deferred_c20_calls=d.c20_calls,deferred_c20_rows=d.c20_rows,
            deferred_max_c20_batch=d.max_c20_batch,deferred_peak_pending_keys=d.peak_pending_keys,
            deferred_pending_key_numeric_bytes=sum(len(k) for k in d.pending),
            deferred_resource_scope='pending key subtotal excludes containers and C20 arrays; fresh-process full tracking required')


def prepare(x,v,cfg,trace=False):
    saved=feedback.Collector
    def factory(x,v,kind,trace=False,mode='watch',capture_repairs=False):
        return Collector(x,v,kind,trace,mode,capture_repairs,cfg.get('route_mode','none'),cfg.get('route_use_dual',True))
    try:
        feedback.Collector=factory;bank,regs,meta,collectors=feedback.prepare(x,v,cfg,trace)
    finally:feedback.Collector=saved
    for collector in collectors:collector.deferred.flush()
    meta['feedback_details']=[c.feedback_metadata() for c in collectors]
    keys={r.tobytes() for r in regs};extra={}
    for collector in collectors:
        for key,origin in collector.route_kept.items():
            if key not in keys and key not in extra:extra[key]=origin
    cap=cfg.get('route_extra_budget');allkeys=list(extra);selected=allkeys if cap is None else allkeys[:cap]
    new=np.array([np.frombuffer(k,np.uint8).reshape(len(bank[0]),len(x)) for k in selected],np.uint8).reshape(-1,4,len(x))
    return bank,np.concatenate([regs,new]),dict(**meta,routing_active=True,deferred_routing_active=True,route_mode=cfg.get('route_mode','none'),route_extra_budget=cap,
        route_original_pattern_keys=[r.tobytes().hex() for r in regs],route_extra_all_keys=[k.hex() for k in allkeys],
        route_extra_pattern_keys=[k.hex() for k in selected],route_extra_geometry_calls=len(selected),route_extra_skipped=len(allkeys)-len(selected),
        route_extra_origins=[dict(pattern=k.hex(),**extra[k]) for k in selected]),collectors


def fit(x,v,state,cfg):
    if state is not None or cfg.get('route_mode','none')=='none' or cfg.get('passthrough',False):
        pred,new,meta=original.fit(x,v,state,cfg);return pred,new,dict(**meta,deferred_routing_active=False)
    bank,regs,meta,_=prepare(x,v,cfg)
    pred,new,detail=feedback.old.old.old.old.materialize_retained(x,v,bank,regs,cfg.get('posterior_samples',512))
    return pred,new,dict(**meta,**detail,evaluated_pattern_keys=[r.tobytes().hex() for r in regs],geometry_budget=None,skipped_for_budget=meta['route_extra_skipped'])


def verify():
    import json
    from pathlib import Path
    configs=json.loads(Path(__file__).with_name('typed_routing_development.json').read_text())['configs']
    seed=5900001;rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);q=rng.uniform(0,1,128)
    v=base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
    states=0;queues=0;proofs=0;later=0
    for config in configs:
        cfg=dict(**config,archive=True,pool='posterior_mix',posterior_samples=512,proposal_budget=1024)
        a=None;b=None
        for n in [4,8]:
            pa,a,ma=original.fit(xx[:n],v[:n],a,cfg);pb,b,mb=fit(xx[:n],v[:n],b,cfg)
            assert np.array_equal(a.anchor,b.anchor) and np.array_equal(a.samples,b.samples) and np.array_equal(pa(q),pb(q));states+=1
            assert ma['positive_mode_keys']==mb['positive_mode_keys']
            if n==4 and cfg.get('route_mode','none')!='none':
                for key in ['route_original_pattern_keys','route_extra_all_keys','route_extra_pattern_keys','route_extra_origins']:
                    assert ma[key]==mb[key],(config['name'],key)
                queues+=1
                for ea,eb in zip(ma['feedback_details'],mb['feedback_details']):
                    assert ea['proofs']==eb['proofs'];proofs+=1
                    for key in ['candidate_mode_visits','distinct_modes','causal_clause_removed','c20_removed']:
                        assert ea['route_stats'][key]==eb['route_stats'][key]
                    assert eb['deferred_route_flushed'] and eb['deferred_max_c20_batch']<=256
            if n==8:assert not mb['deferred_routing_active'];later+=1
    jac=base.forward_jacobian;bp_module=old.old.old.old.old.core.batched;bp=bp_module.refine
    def forbidden(*a,**k):raise AssertionError('global BP entered local deferred candidate')
    try:
        base.forward_jacobian=forbidden;bp_module.refine=forbidden
        fit(xx[:4],v[:4],None,dict(generator='alm',credits=['local'],sweeps=16,restarts=8,retention_mode='bank',feedback_mode='repair',route_mode='tied_both'))
    finally:base.forward_jacobian=jac;bp_module.refine=bp
    return dict(passed=True,all_original_configs=len(configs),states_and_predictions_bitwise=states,extra_queues_and_origins_exact=queues,
        proof_sequences_exact=proofs,later_own_state_passthroughs=later,pure_local_no_bp=True,
        scope='all 20 original configs, first and later write; same observations and original numerical outputs')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
