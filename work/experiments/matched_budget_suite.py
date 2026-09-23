"""262 budget catalogue and fresh outputs; no evaluator or truth access."""
import hashlib
import time
import traceback
import numpy as np
import minimum_dual_resource_suite as original
import expanded_cold_readout as expanded
import region_posterior_memory as prior
oldfit=original.oldfit
cold=original.live.cold
PRIMARY='minimum_dual_alm64'

def catalogue():
    existing={c['name']:c for c in original.configs()};ans=[]
    for method,short,steps in [('adam','adam',[240,480,960]),('gauss_newton','gn',[40,80,160])]:
        for n in steps:
            name=f'{short}{n}';ans.append(existing[name] if name in existing else dict(name=name,family='warm_plus',group='warm_'+short,config=dict(name=name,initial='unchanged',method=method,steps=n,bp=True)))
    for method,short,n in [('adam','adam',240),('gauss_newton','gn',40)]:
        for r in [17,33,65]:
            name=f'cold__{short}{n}'+(f'_r{r}' if r!=17 else '')
            ans.append(existing[name] if name in existing else dict(name=name,family='cold_plus',group='cold_'+short,config=dict(name=name,family='bp',method=method,steps=n,restarts=r)))
    for method in ['alm','pc','nodual']:
        for n in [128,256,512]:
            name=f'cold__{method}{n}';ans.append(existing[name] if name in existing else dict(name=name,family='cold_plus',group='cold_'+method,config=dict(name=name,family='local',method=method,steps=n,restarts=17)))
    for n in [4096,16384,65536]:
        name=f'cold__prior{n}_ridge';ans.append(existing[name] if name in existing else dict(name=name,family='prior_plus',group='prior_ridge',config=dict(name=name,features=n)))
    ans.append(existing[PRIMARY]);assert len(ans)==25 and len({c['name'] for c in ans})==25
    return ans

def prior_bank(n):return np.random.default_rng(731).uniform(-prior.PRIOR_BOUND,prior.PRIOR_BOUND,(n,4))
def prior_fit(n,x,v,q):
    start=time.perf_counter();predict,meta=prior.prior_moments(x,v,4,n);write=time.perf_counter()-start;start=time.perf_counter();prediction=predict(q);read=time.perf_counter()-start
    support=predict(x);arrays=dict(x=x,v=v,q=q,prediction=prediction,point_prediction=prediction.copy(),support_prediction=support)
    return arrays,dict(write_seconds=write,read_seconds=read,total_seconds=write+read,features=n,noise_variance=meta['noise_variance'],persistent_predictor_bytes=meta['persistent_state_bytes'],support_max_error=float(np.max(abs(support-v))),feature_seed=731)

def fit(cfg,x,v,q,seed,loaded):
    start=time.perf_counter();family=cfg['family']
    if family=='warm_plus':a,m=original.oldwarm.fit(cfg['config'],x,v,q,seed)
    elif family=='cold_plus':a,m=expanded.fit(cfg['config'],x,v,q,seed)
    elif family=='prior_plus':a,m=prior_fit(cfg['config']['features'],x,v,q)
    else:a,m=original.fit(cfg,x,v,q,seed,loaded)
    m['complete_call_seconds']=time.perf_counter()-start
    return a,m

RECOVERABLE=(AssertionError,ArithmeticError,ValueError,RuntimeError)
def guarded_fit(cfg,x,v,q,seed,loaded,runner=None):
    """Predeclared numerical failure rule; I/O, memory and coding errors stop."""
    start=time.perf_counter()
    try:
        a,m=(fit if runner is None else runner)(cfg,x,v,q,seed,loaded)
        assert a['prediction'].shape==q.shape
        for k,z in a.items():
            if isinstance(z,np.ndarray) and np.issubdtype(z.dtype,np.number):assert np.isfinite(z).all(),k
        m={**m,'execution_failed':False,'charged_complete_seconds':time.perf_counter()-start}
        return a,m
    except RECOVERABLE as exc:
        note=traceback.format_exc();b=np.zeros(4);prediction=cold.base.forward(q,b)
        a=dict(prediction=prediction,point_prediction=prediction.copy(),selected_b=b,points=b[None],allocation=np.empty(0,dtype=int),best_bank=b[None])
        m=dict(execution_failed=True,failure_type=type(exc).__name__,failure_message=str(exc),failure_traceback=note,
            failure_policy='charge failed attempt then fixed original zero-bias model; no retry and no task removal',charged_complete_seconds=time.perf_counter()-start)
        return a,m

def check_existing(root,seed,cfg,a,m):
    if cfg['name'] in {c['name'] for c in original.configs()}:
        assert not m['execution_failed'];original.check(root,seed,cfg,a,m);return True
    return False

def hashes_of_priors():
    return dict(starts={str(n):hashlib.sha256(expanded.starts(n).tobytes()).hexdigest() for n in [17,33,65]},
        feature_banks={str(n):hashlib.sha256(prior_bank(n).tobytes()).hexdigest() for n in [4096,16384,65536]})
