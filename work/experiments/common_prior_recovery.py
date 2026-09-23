"""Shared bounded recovery from an empty discovered feasible-cell union.

All observed support pairs are retained. Only approximate particles are
discarded. The same fixed prior4096/16384 search is available to every
posterior method, and failed attempts are charged. No query target or seed
selects the recovery. This does not prove global posterior completeness.
"""
import time
import numpy as np
import online_endpoint_h2 as memory

EMPTY='Geometry found no nonzero cell'
FEATURE_BUDGETS=(4096,16384)


class RecoveryExhausted(RuntimeError):
    def __init__(self,events):
        super().__init__('Common prior recovery exhausted after '+str([r['features'] for r in events if r['role']=='recovery']))
        self.events=events


def fit(x,v,state,cfg,rng,count=2048):
    previous=memory.prior.state_digest(state);events=[];start=time.perf_counter()
    try:
        prediction,new,meta=memory.fit(x,v,state,cfg,rng,count)
        elapsed=time.perf_counter()-start
        return prediction,new,dict(meta,recovery=dict(triggered=False,attempts=[dict(role='original',success=True,seconds=elapsed)]))
    except RuntimeError as exc:
        if str(exc)!=EMPTY:raise
        events.append(dict(role='original',features=None,success=False,seconds=time.perf_counter()-start,error=str(exc)))
    for features in FEATURE_BUDGETS:
        reset=memory.original.original.light.config('direct4096','c5')
        reset.update(name=f'common_prior{features}_recovery',initial_features=features,recovery_role=True)
        start=time.perf_counter()
        try:
            prediction,new,meta=memory.fit(x,v,None,reset,rng,count)
        except RuntimeError as exc:
            if str(exc)!=EMPTY:raise
            events.append(dict(role='recovery',features=features,success=False,seconds=time.perf_counter()-start,error=str(exc)))
            continue
        events.append(dict(role='recovery',features=features,success=True,seconds=time.perf_counter()-start,
            geometry_calls=meta['geometry_calls'],positive_cells=len(meta['positive_mode_keys'])))
        return prediction,new,dict(meta,previous_state_digest=previous,recovery=dict(triggered=True,attempts=events,
            selected_features=features,reset_approximate_particles=True,all_observations_reused=len(x),
            common_search='fixed prior seed731, direct128, C5, H2 rounds2 budget1024',
            claim='uniform posterior conditional on recovered discovered union, not globally complete'))
    raise RecoveryExhausted(events)


def verify():
    rng=np.random.default_rng(5900001);b=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=memory.base.forward(xx,b)+np.random.default_rng(24900001).uniform(-.001,.001,24)
    cfg=next(c for c in memory.configs() if c['name']=='alm_native_full_c5_interval_endpoints')
    ref=None;state=None;cases=0
    for n in [4,8,16,24]:
        predict,ref,_=memory.fit(xx[:n],vv[:n],ref,cfg,np.random.default_rng(482601+n),64)
        actual,state,meta=fit(xx[:n],vv[:n],state,cfg,np.random.default_rng(482601+n),64)
        assert ref.samples.tobytes()==state.samples.tobytes() and predict(xx).tobytes()==actual(xx).tobytes()
        assert not meta['recovery']['triggered'];cases+=1
    original=memory.fit
    def fail_original(x,v,state,cfg,rng,count):
        if not cfg.get('recovery_role'):raise RuntimeError(EMPTY)
        return original(x,v,state,cfg,rng,count)
    core=memory.neighbor.previous.core;jac=memory.base.forward_jacobian;bp=core.batched.refine
    def forbidden(*args,**kwargs):raise AssertionError('global BP in shared direct recovery')
    try:
        memory.fit=fail_original;memory.base.forward_jacobian=forbidden;core.batched.refine=forbidden
        _,new,meta=fit(xx[:4],vv[:4],None,cfg,np.random.default_rng(482605),64)
        assert meta['recovery']['triggered'] and meta['recovery']['selected_features']==4096
        assert np.max(abs(memory.prior.capture.model.light.forward_many(xx[:4],new.samples)[1][:,-1]-vv[:4]))<=.001+1e-8
    finally:memory.fit=original;memory.base.forward_jacobian=jac;core.batched.refine=bp
    # Exhaustion is explicit and bounded; unrelated errors must propagate.
    def always_empty(*args,**kwargs):raise RuntimeError(EMPTY)
    try:
        memory.fit=always_empty
        try:fit(xx[:4],vv[:4],None,cfg,np.random.default_rng(482605),64);raise AssertionError('expected exhaustion')
        except RecoveryExhausted as exc:assert len(exc.events)==3;exhausted=len(exc.events)
    finally:memory.fit=original
    def other_error(*args,**kwargs):raise RuntimeError('unrelated error')
    try:
        memory.fit=other_error
        try:fit(xx[:4],vv[:4],None,cfg,np.random.default_rng(482605),64);raise AssertionError('expected original error')
        except RuntimeError as exc:assert str(exc)=='unrelated error'
    finally:memory.fit=original
    return dict(passed=True,successful_four_stage_paths_bytewise=cases,forced_recovery_no_global_bp=True,
        bounded_exhaustion_attempts=exhausted,unrelated_errors_propagate=True,
        scope='old preflight only; no failed development case examined')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
