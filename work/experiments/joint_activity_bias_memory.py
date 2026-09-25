"""Joint shared-bias / per-sample-activity local blocks, without global BP."""
from __future__ import annotations
import numpy as np
import streaming_branch_projection as base


def profile(a,target,oldh,oldb,candidates,trust=.01):
    """Exact activity elimination for each proposed shared bias (R,G)."""
    ss=base.SLOPES[None,None,:,None]
    cc=base.INTERCEPTS[None,None,:,None]
    bias=candidates[:,:,None,None]
    lo=np.maximum(0,np.array([-np.inf,0,.5,1])[None,None,:,None]-bias)
    hi=np.minimum(1,np.array([0,.5,1,np.inf])[None,None,:,None]-bias)
    aa=a[:,None,None,:]; tt=target[:,None,None,:]; hh=oldh[:,None,None,:]
    cand=(aa+ss*(tt-ss*bias-cc)+trust*hh)/(1+ss**2+trust)
    cand=np.minimum(np.maximum(cand,lo),hi)
    energy=(cand-aa)**2+(base.g(cand+bias)-tt)**2+trust*(cand-hh)**2
    energy=np.where(lo<=hi,energy,np.inf)
    which=np.argmin(energy,axis=2)[:,:,None,:]
    besth=np.take_along_axis(cand,which,axis=2)[:,:,0,:]
    value=np.take_along_axis(energy,which,axis=2)[:,:,0,:].mean(axis=2)
    value+=trust*(candidates-oldb[:,None])**2
    return besth,value


def joint_block(a,target,oldh,oldb,grid_size=17,refine_size=9,trust=.01):
    r=len(oldb)
    hsep,_=profile(a,target,oldh,oldb,oldb[:,None],trust)
    bsep=base.bias_solve(hsep[:,0,:],target,oldb,0.,float("inf"),trust)
    candidates=np.column_stack([np.broadcast_to(np.linspace(-base.BOUND,base.BOUND,grid_size),(r,grid_size)),oldb,bsep])
    hs,values=profile(a,target,oldh,oldb,candidates,trust)
    which=np.argmin(values,axis=1); rb=np.arange(r)
    bestb=candidates[rb,which]; besth=hs[rb,which]; bestvalue=values[rb,which]
    spacing=2*base.BOUND/(grid_size-1)
    fine=np.clip(bestb[:,None]+spacing*np.linspace(-1,1,refine_size),-base.BOUND,base.BOUND)
    candidates=np.column_stack([fine,bestb,oldb,bsep])
    hs,values=profile(a,target,oldh,oldb,candidates,trust)
    which=np.argmin(values,axis=1)
    bestb=candidates[rb,which]; besth=hs[rb,which]; bestvalue=values[rb,which]
    initial=np.mean((oldh-a)**2+(base.g(oldh+oldb[:,None])-target)**2,axis=1)
    sepvalue=np.mean((hsep[:,0,:]-a)**2+(base.g(hsep[:,0,:]+bsep[:,None])-target)**2+trust*(hsep[:,0,:]-oldh)**2,axis=1)+trust*(bsep-oldb)**2
    assert np.all(bestvalue<=initial+1e-10)
    assert np.all(bestvalue<=sepvalue+1e-10)
    return besth,bestb,{"max_increase_over_old":float(np.max(bestvalue-initial)),
                      "max_increase_over_separate":float(np.max(bestvalue-sepvalue))}


def discover(x,v,anchor,sweeps=40,restarts=8,dual_rate=.5):
    depth=len(anchor); trust=.01
    b=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(restarts-1,depth))])
    h=np.empty((depth,restarts,len(x))); u=np.zeros_like(h)
    prev=np.broadcast_to(x,(restarts,len(x)))
    for j in range(depth): h[j]=base.g(prev+b[:,j,None]); prev=h[j]
    best=b.copy(); besterr,bestmove=base.score(b,x,v,anchor)
    max_old=-np.inf; max_sep=-np.inf
    for _ in range(sweeps):
        for j in reversed(range(depth)):
            prev=x if j==0 else h[j-1]
            a=base.g(prev+b[:,j,None])-u[j]; old=h[j].copy()
            if j==depth-1:
                h[j]=np.clip((a+trust*old)/(1+trust),np.maximum(0,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                h[j],b[:,j+1],audit=joint_block(a,h[j+1]+u[j+1],old,b[:,j+1].copy())
                max_old=max(max_old,audit["max_increase_over_old"])
                max_sep=max(max_sep,audit["max_increase_over_separate"])
        b[:,0]=base.bias_solve(np.broadcast_to(x,(restarts,len(x))),h[0]+u[0],b[:,0].copy(),anchor[0],float("inf"),trust)
        prev=x; residual=np.empty_like(h)
        for j in range(depth): residual[j]=h[j]-base.g(prev+b[:,j,None]); prev=h[j]
        u+=dual_rate*residual
        err,move=base.score(b,x,v,anchor)
        update=base.better(err,move,besterr,bestmove)
        best[update],besterr[update],bestmove[update]=b[update],err[update],move[update]
    feasible=besterr<=base.EPS+base.TOL
    order=sorted(range(restarts),key=lambda i:(0,bestmove[i]) if feasible[i] else (1,besterr[i]))
    seen=set(); unique=[]
    for i in order:
        key=base.pattern(x,best[i]).tobytes()
        if key not in seen: unique.append(i); seen.add(key)
    return best[unique].copy(),{"unique_regions":len(unique),"discovery_sweeps":sweeps,"discovery_restarts":restarts,
        "max_local_energy_increase_over_old":max_old,"max_local_energy_increase_over_separate":max_sep,
        "discovery_main_arrays_bytes":sum(a.nbytes for a in [b,h,u,best]),"retained_bank_bytes":best[unique].nbytes,
        "bias_profile_evaluations_per_joint_block":32}
