"""Actual online use of analytic branch-normal credit screening."""
import numpy as np
import budgeted_credit_memory as old
import optimized_branch_dual as normal
base=old.base


def prepare(x,v,cfg):
    if not cfg.get('optimized_normal',False):return old.prepare(x,v,cfg)
    original=old.old.Collector;collectors=[]
    class Cached(original):
        def finalize(self):
            if not hasattr(self,'finished_credit'):self.finished_credit=super().finalize()
            return self.finished_credit
    def factory(x,v,kind):
        instance=Cached(x,v,kind);collectors.append(instance);return instance
    try:
        old.old.Collector=factory
        bank,regs,meta=old.prepare(x,v,cfg)
    finally:old.old.Collector=original
    directions=[];seen=set()
    for collector in collectors:
        accepted,_=collector.finalize()
        for a in old.old.direction_bank(accepted,32):
            key=a.tobytes()
            if key not in seen:seen.add(key);directions.append(a)
    directions=np.array(directions).reshape(-1,4,len(x))
    reject,proofs,cost=normal.screen_bank(x,v,regs,directions);kept=regs[~reject]
    return bank,kept,dict(**meta,optimized_normal=True,normal_cost=cost,
        pre_normal_pattern_keys=[r.tobytes().hex() for r in regs],normal_certified_keys=[p['pattern'] for p in proofs],
        normal_exact_values=[p['exact']['value'] for p in proofs],post_normal_pattern_keys=[r.tobytes().hex() for r in kept])


def fit(x,v,state,cfg):
    if state is not None or cfg.get('passthrough',False):
        predict,new,meta=old.fit(x,v,state,cfg);return predict,new,dict(**meta,optimized_normal=False)
    bank,regs,meta=prepare(x,v,cfg);cap=cfg.get('geometry_budget');evaluated=regs if cap is None else regs[:cap]
    predict,new,detail=old.old.materialize_retained(x,v,bank,evaluated,cfg.get('posterior_samples',512))
    return predict,new,dict(**meta,**detail,budget_active=cap is not None,geometry_budget=cap,evaluated_pattern_keys=[r.tobytes().hex() for r in evaluated],skipped_for_budget=len(regs)-len(evaluated))


def verify():
    checked=0;sequences=0
    for seed in [5900000,5900001]:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        for gen,kind,steps in [('alm','local',16),('alm','residual',16),('adam','bp',16),('nodual','residual',16),('pc','residual',80)]:
            cfg=dict(generator=gen,credits=[kind],sweeps=steps,steps=steps,restarts=64)
            _,expected,_=old.fit(x[:4],v[:4],None,cfg);_,new,meta=fit(x[:4],v[:4],None,dict(**cfg,optimized_normal=True))
            assert np.array_equal(expected.anchor,new.anchor) and np.array_equal(expected.samples,new.samples);checked+=1
            for cap in [0,24]:
                _,s,m=fit(x[:4],v[:4],None,dict(**cfg,optimized_normal=True,geometry_budget=cap))
                assert m['evaluated_pattern_keys']==meta['evaluated_pattern_keys'][:cap];sequences+=1
    before_jac=base.forward_jacobian;before_bp=old.old.old.core.batched.refine
    def forbidden(*args,**kwargs):raise AssertionError('Global BP entered pure local optimized-credit candidate')
    try:
        base.forward_jacobian=forbidden;old.old.old.core.batched.refine=forbidden
        fit(x[:4],v[:4],None,dict(generator='alm',credits=['local'],sweeps=16,restarts=64,optimized_normal=True))
    finally:base.forward_jacobian=before_jac;old.old.old.core.batched.refine=before_bp
    return dict(passed=True,unlimited_state_bitwise_cases=checked,prefix_cases=sequences,pure_local_no_global_bp=True)


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
