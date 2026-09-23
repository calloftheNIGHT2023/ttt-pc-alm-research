"""Whole-network BP discovery banks with the same support-only selection rule."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize,least_squares
import streaming_branch_projection as base


def loss_gradient(b,x,v,band=False):
    h=x; derivatives=[]
    for bias in b:
        z=h+bias; derivatives.append(base.derivative(z)); h=base.g(z)
    raw=h-v
    error=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0) if band else raw
    adj=error/len(x); grad=np.empty_like(b)
    for j in reversed(range(len(b))):
        adj=adj*derivatives[j]; grad[j]=adj.sum()
    return .5*np.mean(error**2),grad,raw


def discover(x,v,anchor,*,restarts=64,solver="lbfgs",band=True):
    starts=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(restarts-1,len(anchor)))])
    best=starts.copy(); errors,moves=base.score(best,x,v,anchor)
    calls=0
    for i,start in enumerate(starts):
        cache_b=None; cache=None
        def evaluate(b):
            nonlocal calls,cache_b,cache
            if cache_b is not None and np.array_equal(cache_b,b): return cache
            calls+=1
            if solver=="trf":
                pred,jac=base.forward_jacobian(x,b); raw=pred-v
                residual=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0) if band else raw
                if band: jac=jac*(np.abs(raw)>base.EPS)[:,None]
                cache=(residual,jac)
            else:
                loss,grad,raw=loss_gradient(b,x,v,band); cache=(loss,grad)
            err=float(np.max(np.abs(raw))); move=.5*float(np.sum((b-anchor)**2))
            if base.better(np.array([err]),np.array([move]),errors[i:i+1],moves[i:i+1])[0]:
                best[i]=b; errors[i]=err; moves[i]=move
            cache_b=b.copy()
            return cache
        if solver=="trf":
            res=least_squares(lambda z:evaluate(z)[0],start,jac=lambda z:evaluate(z)[1],
                bounds=(-base.BOUND,base.BOUND),method="trf",max_nfev=300,ftol=1e-12,xtol=1e-12,gtol=1e-10)
        else:
            res=minimize(evaluate,start,jac=True,method="L-BFGS-B",bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                options={"maxiter":300,"ftol":1e-13,"gtol":1e-9,"maxls":40})
        evaluate(res.x)
    feasible=errors<=base.EPS+base.TOL
    order=sorted(range(restarts),key=lambda i:(0,moves[i]) if feasible[i] else (1,errors[i]))
    unique=[]; seen=set()
    for i in order:
        key=base.pattern(x,best[i]).tobytes()
        if key not in seen: unique.append(i); seen.add(key)
    return best[unique].copy(),{"discovery_solver":solver,"band_loss":band,"function_derivative_evaluations":calls,
        "discovery_restarts":restarts,"unique_regions":len(unique),"retained_bank_bytes":best[unique].nbytes}


def verify_gradient():
    rng=np.random.default_rng(58913); errors=[]
    for band in [False,True]:
        for _ in range(10):
            x=rng.uniform(0,1,24); b=rng.uniform(-.1,.1,4); v=rng.uniform(0,1,24)
            value,grad,_=loss_gradient(b,x,v,band)
            numeric=np.array([(loss_gradient(b+np.eye(4)[j]*1e-7,x,v,band)[0]-loss_gradient(b-np.eye(4)[j]*1e-7,x,v,band)[0])/2e-7 for j in range(4)])
            error=float(np.max(np.abs(grad-numeric))); assert error<1e-6,error; errors.append(error)
    return {"passed":True,"cases":len(errors),"max_fd_error":max(errors)}
