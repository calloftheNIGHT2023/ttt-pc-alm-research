"""Local discovery bank plus auditable branch scheduling. No query access."""
from __future__ import annotations
import numpy as np
import streaming_branch_projection as base
import certified_branch_solver_v2 as regional


def discover(x,v,anchor,sweeps=120,restarts=64,dual_rate=.5,gradient_blocks=False):
    depth=len(anchor); trust=.01
    b=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(restarts-1,depth))])
    h=np.empty((depth,restarts,len(x))); u=np.zeros_like(h)
    prev=np.broadcast_to(x,(restarts,len(x)))
    for j in range(depth):
        h[j]=base.g(prev+b[:,j,None]); prev=h[j]
    best=b.copy(); besterr,bestmove=base.score(b,x,v,anchor)
    for it in range(sweeps):
        for j in reversed(range(depth)):
            prev=x if j==0 else h[j-1]
            a=base.g(prev+b[:,j,None])-u[j]; old=h[j].copy()
            if j==depth-1:
                h[j]=np.clip((a+trust*old)/(1+trust),np.maximum(0,v-base.EPS),np.minimum(1,v+base.EPS))
            elif gradient_blocks:
                zz=old+b[:,j+1,None]
                grad=old-a+base.derivative(zz)*(base.g(zz)-h[j+1]-u[j+1])
                h[j]=np.clip(old-grad/(5+trust),0,1)
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
            if gradient_blocks:
                old=b[:,j].copy(); zz=prev+old[:,None]
                grad=np.mean((base.g(zz)-(h[j]+u[j]))*base.derivative(zz),axis=1)
                b[:,j]=np.clip(old-grad/(4+trust),-base.BOUND,base.BOUND)
            else:
                b[:,j]=base.bias_solve(prev,h[j]+u[j],b[:,j].copy(),anchor[j],float("inf"),trust)
        prev=x; residual=np.empty_like(h)
        for j in range(depth):
            residual[j]=h[j]-base.g(prev+b[:,j,None]); prev=h[j]
        u+=dual_rate*residual
        err,move=base.score(b,x,v,anchor)
        update=base.better(err,move,besterr,bestmove)
        best[update],besterr[update],bestmove[update]=b[update],err[update],move[update]
    feasible=besterr<=base.EPS+base.TOL
    order=sorted(range(restarts),key=lambda i:(0,bestmove[i]) if feasible[i] else (1,besterr[i]))
    # Keep the old winner first; equivalent signatures have the same convex
    # feasible set, though their finite-step iterates need not be identical.
    seen=set(); unique=[]
    for i in order:
        key=base.pattern(x,best[i]).tobytes()
        if key not in seen: unique.append(i); seen.add(key)
    return best[unique].copy(),{"unique_regions":len(unique),"discovery_sweeps":sweeps,
        "discovery_restarts":restarts,"discovery_main_arrays_bytes":sum(a.nbytes for a in [b,h,u,best]),
        "retained_bank_bytes":best[unique].nbytes}
