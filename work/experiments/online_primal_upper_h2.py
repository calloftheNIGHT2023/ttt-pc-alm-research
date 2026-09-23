"""Causal, cached primal upper gate with unchanged region decisions.

Only the full endpoint credit bank can be bypassed. A skipped region is
retained for the original geometry, never excluded. All six credit learners
can use the same gate, with their ungated comparators left available.
"""
import time
import numpy as np
import bounded_error_primal_gate as upper
import common_prior_recovery as recovery
import run_recovered_online_comparison as previous

memory=recovery.memory
EndpointBank=memory.EndpointIntervalBank
PRIMARY=previous.PRIMARY+'_primal_gate'


class GatedBank(EndpointBank):
    def __init__(self,x,v,directions):
        super().__init__(x,v,directions)
        start=time.perf_counter();self.upper=upper.Bank(self.x,self.v,self.bank)
        self.upper_construction_seconds=time.perf_counter()-start

    def screen(self,regs):
        start=time.perf_counter();values=self.upper.values(regs);skip=np.all(values<=0,axis=1)
        gate_seconds=time.perf_counter()-start
        reject,proofs,meta=super().screen(regs[~skip]);result=np.zeros(len(regs),bool);result[~skip]=reject
        meta['primal_gate']=dict(seconds=gate_seconds,construction_seconds=self.upper_construction_seconds,
            input_regions=len(regs),skipped=int(skip.sum()),skipped_patterns=[r.tobytes().hex() for r in regs[skip]],
            upper_layer_solves=self.upper.layer_solves,fast_bounded_error=self.upper.fast,
            layer_absolute_error_allowance=self.upper.layer_error,
            cached_upper_numeric_bytes=sum(len(k)+v[0].nbytes+1 for cache in self.upper.cache for k,v in cache.items()),
            upper_output_numeric_bytes=self.upper.output.nbytes,
            action='skip credit computation only; retain region for original geometry')
        return result,proofs,meta


def configs():
    controls=previous.configs()
    return controls+[dict(c,name=c['name']+'_primal_gate',primal_upper_gate=True,ungated_comparator=c['name'])
        for c in controls if c.get('endpoint_bound')]


def fit(x,v,state,cfg,rng,count=2048):
    saved=memory.EndpointIntervalBank
    try:
        if cfg.get('primal_upper_gate') and state is None:memory.EndpointIntervalBank=GatedBank
        pred,new,meta=recovery.fit(x,v,state,cfg,rng,count)
    finally:memory.EndpointIntervalBank=saved
    return pred,new,meta


def verify():
    xx,vv=previous.olddriver.observations(5900001);checks=dict(cold_success_states=0,cold_matched_exhaustion=0,
        stream_states=0,unchanged_proof_records=0,screen_hashes=0,skipped_regions=0)
    for cfg in [c for c in configs() if c.get('primal_upper_gate')]:
        control=dict(cfg,name=cfg['ungated_comparator'],primal_upper_gate=False)
        for n in [4,8,16,24]:
            outputs=[]
            for c in [control,cfg]:
                try:outputs.append(fit(xx[:n],vv[:n],None,c,np.random.default_rng(483301+n),64))
                except recovery.RecoveryExhausted as exc:outputs.append(exc)
            if isinstance(outputs[0],recovery.RecoveryExhausted):
                assert isinstance(outputs[1],recovery.RecoveryExhausted)
                assert [(e['role'],e['success'],e.get('features')) for e in outputs[0].events]==[(e['role'],e['success'],e.get('features')) for e in outputs[1].events]
                checks['cold_matched_exhaustion']+=1;continue
            rp,rs,rm=outputs[0];pred,state,meta=outputs[1]
            assert rs.samples.tobytes()==state.samples.tobytes() and rs.anchor.tobytes()==state.anchor.tobytes()
            assert rp(xx).tobytes()==pred(xx).tobytes()
            for key in ['credit_proofs','credit_bank','positive_mode_keys','discovery_bank_sha256','completion_proposals','geometry_calls']:
                assert meta.get(key)==rm.get(key),key
            for key in ['input_pattern_hash','screened_pattern_hash']:
                assert [c[key] for c in meta.get('screen_calls',[])]==[c[key] for c in rm.get('screen_calls',[])]
                checks['screen_hashes']+=len(meta.get('screen_calls',[]))
            checks['cold_success_states']+=1;checks['unchanged_proof_records']+=len(meta.get('credit_proofs',[]))
            checks['skipped_regions']+=sum(c['credit'].get('primal_gate',{}).get('skipped',0) for c in meta.get('screen_calls',[]))
        states=[None,None]
        for n in [4,8,16,24]:
            outputs=[fit(xx[:n],vv[:n],s,c,np.random.default_rng(483311+n),64) for s,c in zip(states,[control,cfg])]
            assert outputs[0][1].samples.tobytes()==outputs[1][1].samples.tobytes()
            assert outputs[0][0](xx).tobytes()==outputs[1][0](xx).tobytes()
            states=[r[1] for r in outputs];checks['stream_states']+=1
    core=memory.neighbor.previous.core;jac=memory.base.forward_jacobian;refine=core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP used by gated local method')
    try:
        memory.base.forward_jacobian=forbidden;core.batched.refine=forbidden;state=None
        cfg=next(c for c in configs() if c['name']==PRIMARY)
        for n in [4,8,16,24]:_,state,_=fit(xx[:n],vv[:n],state,cfg,np.random.default_rng(483321+n),64)
    finally:memory.base.forward_jacobian=jac;core.batched.refine=refine
    return dict(passed=True,checks=checks,local_four_stage_no_global_bp=True,
        scope='Old preflight seed; all six learners; cold and own-state paths; no efficacy or timing inference')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
