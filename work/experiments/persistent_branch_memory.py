"""Append-only candidate memory; local ALM and BP have identical bank retention.

No query values enter this module. Refinement is a shared GLOBAL affine LP/QP.
The local discovery path never computes a chain derivative.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog, minimize, least_squares
import streaming_branch_projection as base
from matched_discovery_baselines import loss_gradient


def initial(x, anchor, restarts, state, mode):
    depth=len(anchor)
    b=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(restarts-1,depth))])
    if state is not None and mode!="cold":
        assert np.array_equal(x[:len(state["x"])],state["x"]), "Only append-only contexts are supported"
        assert len(state["b"])==restarts
        b[1:]=state["b"][1:]
    h=np.empty((depth,restarts,len(x))); prev=x
    for j in range(depth):
        h[j]=base.g(prev+b[:,j,None]); prev=h[j]
    u=np.zeros_like(h)
    if state is not None and mode in ["activities","full"]:
        nold=len(state["x"])
        h[:,1:,:nold]=state["h"][:,1:]
        if mode=="full": u[:,1:,:nold]=state["u"][:,1:]
    return b,h,u


def local_bank(x,v,anchor,cfg,state):
    restarts=cfg.get("restarts",64); depth=len(anchor); trust=.01
    b,h,u=initial(x,anchor,restarts,state,cfg["memory"])
    best=b.copy(); besth=h.copy(); bestu=u.copy()
    errors,moves=base.score(b,x,v,anchor)
    sweeps=cfg.get("initial_sweeps",120) if state is None else cfg["sweeps"]
    for _ in range(sweeps):
        for j in reversed(range(depth)):
            prev=x if j==0 else h[j-1]
            a=base.g(prev+b[:,j,None])-u[j]; old=h[j].copy()
            if j==depth-1:
                h[j]=np.clip((a+trust*old)/(1+trust),np.maximum(0,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                nb=b[:,j+1]
                lo=np.maximum(0,np.array([-np.inf,0,.5,1.])[:,None]-nb)
                hi=np.minimum(1,np.array([0.,.5,1.,np.inf])[:,None]-nb)
                slope=base.SLOPES[:,None,None]
                offset=(base.SLOPES[:,None]*nb+base.INTERCEPTS[:,None])[:,:,None]
                target=h[j+1]+u[j+1]
                cand=(a[None]+slope*(target[None]-offset)+trust*old[None])/(1+slope**2+trust)
                cand=np.minimum(np.maximum(cand,lo[:,:,None]),hi[:,:,None])
                energy=(cand-a)**2+(base.g(cand+nb[None,:,None])-target)**2+trust*(cand-old)**2
                energy=np.where((lo>hi)[:,:,None],np.inf,energy)
                h[j]=np.take_along_axis(cand,np.argmin(energy,axis=0)[None],axis=0)[0]
        for j in range(depth):
            prev=np.broadcast_to(x,(restarts,len(x))) if j==0 else h[j-1]
            b[:,j]=base.bias_solve(prev,h[j]+u[j],b[:,j].copy(),anchor[j],float("inf"),trust)
        prev=x
        for j in range(depth):
            residual=h[j]-base.g(prev+b[:,j,None]); u[j]+=cfg.get("dual_rate",.5)*residual; prev=h[j]
        err,mov=base.score(b,x,v,anchor); update=base.better(err,mov,errors,moves)
        best[update]=b[update]; besth[:,update]=h[:,update]; bestu[:,update]=u[:,update]
        errors[update]=err[update]; moves[update]=mov[update]
    saved={"x":x.copy(),"b":best.copy()}
    if cfg["memory"] in ["full","activities"]: saved["h"]=besth.copy()
    if cfg["memory"]=="full": saved["u"]=bestu.copy()
    return best,errors,moves,saved,{"discovery_sweeps":sweeps,"discovery_restarts":restarts,
        "discovery_primary_arrays_bytes":sum(a.nbytes for a in [b,h,u,best,besth,bestu])}


def bp_bank(x,v,anchor,cfg,state):
    restarts=cfg.get("restarts",64); solver=cfg["generator"]
    starts,_,_=initial(x,anchor,restarts,state,cfg["memory"])
    best=starts.copy(); errors,moves=base.score(best,x,v,anchor); calls=0
    for i,start in enumerate(starts):
        cache_b=None; cache=None
        def evaluate(b):
            nonlocal cache_b,cache,calls
            if cache_b is not None and np.array_equal(b,cache_b): return cache
            calls+=1
            if solver=="trf":
                pred,jac=base.forward_jacobian(x,b); raw=pred-v
                residual=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0)
                cache=(residual,jac*(np.abs(raw)>base.EPS)[:,None])
            else:
                loss,grad,raw=loss_gradient(b,x,v,band=True); cache=(loss,grad)
            err=np.max(np.abs(raw)); mov=.5*np.sum((b-anchor)**2)
            if base.better(np.array([err]),np.array([mov]),errors[i:i+1],moves[i:i+1])[0]:
                best[i]=b; errors[i]=err; moves[i]=mov
            cache_b=b.copy(); return cache
        if solver=="trf":
            res=least_squares(lambda z:evaluate(z)[0],start,jac=lambda z:evaluate(z)[1],bounds=(-base.BOUND,base.BOUND),
                method="trf",max_nfev=300,ftol=1e-12,xtol=1e-12,gtol=1e-10)
        else:
            res=minimize(evaluate,start,jac=True,method="L-BFGS-B",bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                options={"maxiter":300,"ftol":1e-13,"gtol":1e-9,"maxls":40})
        evaluate(res.x)
    return best,errors,moves,{"x":x.copy(),"b":best.copy()},{"function_derivative_evaluations":calls,"discovery_restarts":restarts}


def project(bank,x,v,anchor,max_regions=8):
    best=bank[0].copy(); err0,mov0=base.score(best[None],x,v,anchor); tried=feasible=0
    for start in bank[:max_regions]:
        tried+=1; _,_,mat,rhs=base.branch_polytope(x,v,start)
        lp=linprog(np.zeros(len(anchor)),A_ub=mat,b_ub=rhs,bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
            options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        if not lp.success: continue
        feasible+=1
        res=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),lp.x,jac=True,method="SLSQP",
            constraints={"type":"ineq","fun":lambda b:rhs-mat@b,"jac":lambda b:-mat},
            bounds=[(-base.BOUND,base.BOUND)]*len(anchor),options={"ftol":1e-12,"maxiter":300})
        for b in [lp.x,res.x]:
            err,mov=base.score(b[None],x,v,anchor)
            if base.better(err,mov,err0,mov0)[0]: best,err0,mov0=b.copy(),err,mov
        if err0[0]<=base.EPS+base.TOL: break
    return best,{"regions_attempted":tried,"lp_feasible_regions":feasible}


def fit(x,v,anchor,cfg,state=None):
    if np.max(np.abs(base.forward(x,anchor)-v))<=base.EPS:
        return anchor.copy(),state,{"skipped":True,"persistent_state_bytes":0 if state is None else sum(a.nbytes for a in state.values())}
    function=local_bank if cfg["generator"]=="alm" else bp_bank
    best,errors,moves,saved,meta=function(x,v,anchor,cfg,state)
    order=sorted(range(len(best)),key=lambda i:(0,moves[i]) if errors[i]<=base.EPS+base.TOL else (1,errors[i]))
    seen=set(); unique=[]
    for i in order:
        key=base.pattern(x,best[i]).tobytes()
        if key not in seen: seen.add(key); unique.append(i)
    result,projection=project(best[unique],x,v,anchor,cfg.get("max_regions",8))
    return result,saved,{**meta,**projection,"unique_regions":len(unique),
        "persistent_state_bytes":sum(a.nbytes for a in saved.values())}


def verify():
    from hybrid_discovery_bank import discover as old_local
    from matched_discovery_baselines import discover as old_bp
    rng=np.random.default_rng(44991); x=rng.uniform(0,1,8); v=base.forward(x,rng.uniform(-.12,.12,4)); anchor=np.zeros(4)
    cfg={"generator":"alm","memory":"cold","restarts":8,"initial_sweeps":12,"sweeps":12}
    best,errors,moves,state,_=local_bank(x,v,anchor,cfg,None)
    old,_=old_local(x,v,anchor,restarts=8,sweeps=12)
    order=sorted(range(8),key=lambda i:(0,moves[i]) if errors[i]<=base.EPS+base.TOL else (1,errors[i]))
    assert np.allclose(best[order[0]],old[0],atol=1e-13), (best[order[0]],old[0])
    cfg.update(memory="full")
    _,_,_,state,_=local_bank(x,v,anchor,cfg,None)
    xx=np.r_[x,rng.uniform(0,1,4)]; b,h,u=initial(xx,anchor,8,state,"full")
    assert np.array_equal(u[:,1:,:8],state["u"][:,1:]); assert np.all(u[:,:,8:]==0)
    assert np.array_equal(h[:,1:,:8],state["h"][:,1:]); assert np.all(u[:,0]==0)
    for solver in ["trf","lbfgs"]:
        cfg.update(generator=solver,memory="cold")
        best,errors,moves,_,_=bp_bank(x,v,anchor,cfg,None)
        old,_=old_bp(x,v,anchor,restarts=8,solver=solver,band=True)
        order=sorted(range(8),key=lambda i:(0,moves[i]) if errors[i]<=base.EPS+base.TOL else (1,errors[i]))
        assert np.allclose(best[order[0]],old[0],atol=1e-13)
    return {"passed":True,"cold_local_and_bp_equivalence":True,"append_zero_duals_and_anchor_reset":True}
