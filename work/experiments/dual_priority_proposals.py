"""Support-only shared-pool ranking: local dual scores and strong controls.

No query, teacher, reference geometry or posterior-region volume enters here.
Only the explicitly named BP comparator calls the full forward Jacobian.
"""
import hashlib
import time
import numpy as np
import light_tied_proposals as light

common=light.common
base=light.base
SCORERS=['dual_gain','zero_gain','support_band','bp_gain','movement','fifo','hash']


def residuals(x,b,h):
    prev=np.broadcast_to(x,(len(b),len(x)));parts=[]
    for j in range(b.shape[1]):
        parts.append(h[:,j]-base.g(prev+b[:,j,None]));prev=h[:,j]
    return np.array(parts).transpose(1,0,2)


def shared_pool(x,b,h,u):
    """Actual-u then zero-u candidates; exact binary64 (b,h) deduplication."""
    start=time.perf_counter();bb=[];hh=[];seen=set();segments=0;raw=0
    for credit in [u,np.zeros_like(u)]:
        pb,ph,meta=light.tied_free(x,b,h,credit);segments+=meta['segments'];raw+=len(pb)
        for q,z in zip(pb,ph):
            key=q.tobytes()+z.tobytes()
            if key not in seen:seen.add(key);bb.append(q);hh.append(z)
    return np.array(bb).reshape(-1,len(b)),np.array(hh).reshape(-1,*h.shape),dict(
        raw_candidates=raw,candidates=len(bb),segments=segments,construction_seconds=time.perf_counter()-start)


def local_scores(x,b,h,u,pb,ph):
    """Lower is better. Subtract each callback's own current-state energy."""
    n=len(x);r0=residuals(x,b[None],h[None])[0];r=residuals(x,pb,ph)
    move=np.sum((ph-h)**2,axis=(1,2))+n*np.sum((pb-b)**2,axis=1)
    zero=(np.sum(r*r,axis=(1,2))-np.sum(r0*r0)+light.TAU*move)/n
    credit=2*np.sum(u*(r-r0),axis=(1,2))/n
    return dict(dual_gain=zero+credit,zero_gain=zero,movement=move/n)


def bp_scores(x,v,b,pb):
    """Comparator only: first-order change of the true forward band loss."""
    predicted,jac=base.forward_jacobian(x,b)
    residual=np.sign(predicted-v)*np.maximum(np.abs(predicted-v)-base.EPS,0)
    gradient=2*np.mean(residual[:,None]*jac,axis=0)
    return (pb-b)@gradient


def scores(x,v,b,h,u,pb,ph):
    times={};start=time.perf_counter();result=local_scores(x,b,h,u,pb,ph)
    times['local_score_seconds']=time.perf_counter()-start
    start=time.perf_counter();patterns,activities=light.forward_many(x,pb)
    times['forward_pattern_seconds']=time.perf_counter()-start
    start=time.perf_counter();error=np.maximum(np.abs(activities[:,-1]-v)-base.EPS,0)
    result['support_band']=np.mean(error*error,axis=1)
    times['support_score_seconds']=time.perf_counter()-start
    start=time.perf_counter();result['bp_gain']=bp_scores(x,v,b,pb)
    times['bp_score_seconds']=time.perf_counter()-start
    assert all(np.isfinite(z).all() for z in result.values())
    return patterns,result,times


def rank(keys,features,scorer):
    if scorer=='fifo':return list(keys)
    if scorer=='hash':return sorted(keys,key=lambda k:hashlib.sha256(b'dual-priority-v1'+k).digest())
    position={k:i for i,k in enumerate(keys)}
    return sorted(keys,key=lambda k:(features[k][scorer],position[k]))


def canonical_select(keys,features,scorer,budget):
    ordered=rank(keys,features,scorer);chosen=set(ordered if budget is None else ordered[:budget])
    # Fixed first-arrival materialization order removes RNG/permutation effects.
    return [k for k in keys if k in chosen]


def verify():
    rng=np.random.default_rng(734601);checks=0;bp_checks=0;maxerr=0.;maxbp=0.
    for d,n in [(2,2),(4,4),(4,8)]:
        for _ in range(3):
            x=rng.uniform(.02,.98,n);b=rng.uniform(-light.B,light.B,d)
            h=rng.uniform(0,1,(d,n));u=rng.normal(0,.1,(d,n));v=rng.uniform(0,1,n)
            pb,ph,_=shared_pool(x,b,h,u);ss=local_scores(x,b,h,u,pb,ph)
            move=np.sum((ph-h)**2,axis=(1,2))+n*np.sum((pb-b)**2,axis=1)
            full=(common.energy_many(x,pb,ph,u)-common.energy_many(x,b[None],h[None],u)[0]+light.TAU*move)/n
            err=float(np.max(np.abs(full-ss['dual_gain'])));assert err<1e-12;maxerr=max(maxerr,err);checks+=len(pb)
            jac=base.forward_jacobian
            def forbidden(*a,**k):raise AssertionError('local rank called global BP')
            try:
                base.forward_jacobian=forbidden;local_scores(x,b,h,u,pb,ph)
            finally:base.forward_jacobian=jac
            delta=rng.normal(0,1,(5,d));step=1e-8;analytic=bp_scores(x,v,b,b+step*delta)/step
            def loss(q):return np.mean(np.maximum(np.abs(base.forward(x,q)-v)-base.EPS,0)**2)
            numeric=np.array([(loss(b+step*z)-loss(b-step*z))/(2*step) for z in delta])
            err=float(np.max(np.abs(analytic-numeric)));assert err<1e-5;maxbp=max(maxbp,err);bp_checks+=len(delta)
    keys=[bytes([i]) for i in range(7)];features={k:{s:float(6-i) for s in SCORERS[:5]} for i,k in enumerate(keys)}
    for scorer in SCORERS:
        assert canonical_select(keys,features,scorer,None)==keys
        assert canonical_select(keys,features,scorer,3)==[k for k in keys if k in set(rank(keys,features,scorer)[:3])]
    return dict(passed=True,local_objective_checks=checks,max_objective_error=maxerr,bp_direction_checks=bp_checks,
        max_bp_finite_difference_error=maxbp,local_forbidden_bp_passed=True,full_budget_canonical_order=True,
        scope='score identities and fixed-order selection; no ranking-to-query-risk guarantee')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
