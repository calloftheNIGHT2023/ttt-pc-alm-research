"""Support-conditioned prior proposals and equally seeded local/BP refinement."""
from __future__ import annotations
import numpy as np
from scipy.optimize import minimize,least_squares
import streaming_branch_projection as base
from matched_discovery_baselines import loss_gradient


def deduplicate(bank,x,v,anchor):
    err,mov=base.score(bank,x,v,anchor)
    order=sorted(range(len(bank)),key=lambda i:(0,mov[i]) if err[i]<=base.EPS+base.TOL else (1,err[i]))
    chosen=[]; seen=set()
    for i in order:
        key=base.pattern(x,bank[i]).tobytes()
        if key not in seen: chosen.append(i); seen.add(key)
    return bank[chosen].copy()


def proposals(x,v,anchor,features=65536,restarts=64):
    """Nearest distinct activation patterns, never nearest unseen-query answers."""
    bank=np.random.default_rng(731).uniform(-.12,.12,(features,len(anchor)))
    h=np.broadcast_to(x,(features,len(x))); packed=[]
    for j in range(len(anchor)):
        z=h+bank[:,j,None]; reg=np.searchsorted(base.KNOTS,z,side="right").astype(np.uint8)
        bits=np.stack([reg&1,reg>>1],axis=-1).reshape(features,2*len(x))
        packed.append(np.packbits(bits,axis=1,bitorder="little")); h=base.g(z)
    raw=(h-v)**2; distance=raw.mean(axis=1)
    signatures=np.ascontiguousarray(np.concatenate(packed,axis=1))
    tokens=signatures.view(np.dtype((np.void,signatures.shape[1]))).reshape(-1)
    order=np.argsort(distance,kind="stable")
    _,first=np.unique(tokens[order],return_index=True)
    candidates=order[np.sort(first)][:restarts-1]
    starts=np.vstack([anchor,bank[candidates]])
    return starts,{"proposal_prior_tasks":features,"proposal_distinct_patterns":len(first),"proposal_restarts":len(starts),
        "proposal_bank_parameter_bytes":bank.nbytes,"proposal_support_array_bytes":h.nbytes,
        "proposal_signature_bytes":signatures.nbytes,"proposal_nearest_support_mse":float(distance[order[0]])}


def refine_local(starts,x,v,anchor,sweeps=60,dual_rate=.5):
    b=starts.copy(); restarts,depth=b.shape; trust=.01
    h=np.empty((depth,restarts,len(x))); u=np.zeros_like(h); prev=x
    for j in range(depth): h[j]=base.g(prev+b[:,j,None]); prev=h[j]
    best=b.copy(); besterr,bestmove=base.score(b,x,v,anchor)
    for _ in range(sweeps):
        for j in reversed(range(depth)):
            prev=x if j==0 else h[j-1]; old=h[j].copy()
            a=base.g(prev+b[:,j,None])-u[j]
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
            residual=h[j]-base.g(prev+b[:,j,None]); u[j]+=dual_rate*residual; prev=h[j]
        err,mov=base.score(b,x,v,anchor); update=base.better(err,mov,besterr,bestmove)
        best[update]=b[update]; besterr[update]=err[update]; bestmove[update]=mov[update]
    return best,{"local_sweeps":sweeps,"dual_rate":dual_rate,"refinement_major_arrays_bytes":sum(a.nbytes for a in [b,h,u,best])}


def refine_bp(starts,x,v,anchor,solver="lbfgs"):
    best=starts.copy(); errors,moves=base.score(best,x,v,anchor); calls=0
    for i,start in enumerate(starts):
        cache_b=None; cache=None
        def evaluate(b):
            nonlocal calls,cache_b,cache
            if cache_b is not None and np.array_equal(cache_b,b): return cache
            calls+=1
            if solver=="trf":
                pred,jac=base.forward_jacobian(x,b); raw=pred-v
                residual=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0)
                cache=(residual,jac*(np.abs(raw)>base.EPS)[:,None])
            else:
                loss,grad,raw=loss_gradient(b,x,v,True); cache=(loss,grad)
            err=float(np.max(np.abs(raw))); move=.5*float(np.sum((b-anchor)**2))
            if base.better(np.array([err]),np.array([move]),errors[i:i+1],moves[i:i+1])[0]:
                best[i]=b; errors[i]=err; moves[i]=move
            cache_b=b.copy(); return cache
        if solver=="trf":
            res=least_squares(lambda z:evaluate(z)[0],start,jac=lambda z:evaluate(z)[1],bounds=(-base.BOUND,base.BOUND),
                method="trf",max_nfev=300,ftol=1e-12,xtol=1e-12,gtol=1e-10)
        else:
            res=minimize(evaluate,start,jac=True,method="L-BFGS-B",bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                options={"maxiter":300,"ftol":1e-13,"gtol":1e-9,"maxls":40})
        evaluate(res.x)
    return best,{"function_derivative_evaluations":calls,"refinement_solver":solver}


def verify():
    from hybrid_discovery_bank import discover as old_local
    from matched_discovery_baselines import discover as old_bp
    rng=np.random.default_rng(40552); x=rng.uniform(0,1,8); v=base.forward(x,rng.uniform(-.12,.12,4)); anchor=np.zeros(4)
    starts=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(7,4))])
    actual,_=refine_local(starts,x,v,anchor,sweeps=12); actual=deduplicate(actual,x,v,anchor)
    expected,_=old_local(x,v,anchor,sweeps=12,restarts=8)
    assert np.allclose(actual,expected,atol=1e-13)
    for solver in ["trf","lbfgs"]:
        actual,_=refine_bp(starts,x,v,anchor,solver); actual=deduplicate(actual,x,v,anchor)
        expected,_=old_bp(x,v,anchor,restarts=8,solver=solver,band=True)
        assert np.allclose(actual,expected,atol=1e-13)
    p,_=proposals(x,v,anchor,features=512,restarts=16)
    bank=np.random.default_rng(731).uniform(-.12,.12,(512,4)); h=np.stack([base.forward(x,b) for b in bank]); distance=np.mean((h-v)**2,axis=1)
    reference=[]; seen=set()
    for i in np.argsort(distance,kind="stable"):
        key=base.pattern(x,bank[i]).tobytes()
        if key not in seen: seen.add(key); reference.append(i)
    assert np.array_equal(p,np.vstack([anchor,bank[reference[:15]]]))
    return {"passed":True,"old_refiners_unchanged_at_equal_starts":True,"packed_signature_selection_matches_reference":True}
