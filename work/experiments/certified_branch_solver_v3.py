"""Regional PDHG with exact box-separation checks of dual displacement rays.

PDHG infeasibility detection is prior art (Applegate et al., 2021). Here any
proposed dual vector is accepted ONLY after an exact rational certificate.
"""
from __future__ import annotations
import numpy as np
import streaming_branch_projection as base
from certified_branch_solver import exact_certificate


def solve(x,v,anchor,start,steps=8000,certificate=True,check_every=20,pattern_override=None):
    depth,n=len(anchor),len(x)
    regs=base.pattern(x,start) if pattern_override is None else np.asarray(pattern_override,dtype=int).copy()
    assert regs.shape==(depth,n) and np.all((regs>=0)&(regs<=3))
    s=base.SLOPES[regs]; c=base.INTERCEPTS[regs]
    lo=np.maximum(-base.BOUND,np.array([-np.inf,0,.5,1])[regs])
    hi=np.minimum(1+base.BOUND,np.array([0,.5,1,np.inf])[regs])
    hlow=np.zeros((depth,n)); hhigh=np.ones((depth,n))
    hlow[-1]=np.maximum(0,v-base.EPS); hhigh[-1]=np.minimum(1,v+base.EPS)
    dlo=hlow.copy(); dhi=hhigh.copy()
    dlo[-1]=np.maximum(0,v-base.EPS-base.TOL)
    dhi[-1]=np.minimum(1,v+base.EPS+base.TOL)
    b=start.copy(); z=np.empty((depth,n)); h=np.empty_like(z)
    prev=x
    for j in range(depth):
        z[j]=prev+b[j]; h[j]=base.g(z[j]); prev=h[j]
    up=np.zeros_like(h); ua=np.zeros_like(h)
    bb,zb,hb=b.copy(),z.copy(),h.copy()
    eta=.99; tb=eta/n
    tz=eta/(1+np.abs(s)); th=np.full((depth,1),eta/2); th[-1]=eta
    sp=np.full((depth,1),eta/3); sp[0]=eta/2; sa=eta/(1+np.abs(s))
    best=start.copy(); besterr,bestmove=base.score(best[None],x,v,anchor)
    gap=None; cert=None; reason="iteration_budget"
    recent=[]
    def components(p,a):
        bc=-p.sum(axis=1); zc=p-s*a
        hc=a.copy(); hc[:-1]-=p[1:]
        rest=float(np.sum(zc*np.where(zc>=0,lo,hi))+np.sum(hc*np.where(hc>=0,dlo,dhi))-p[0]@x-np.sum(a*c))
        return bc,rest
    for it in range(steps):
        prev=np.vstack([x,hb[:-1]])
        up+=sp*(zb-prev-bb[:,None]); ua+=sa*(hb-s*zb-c)
        oldb,oldz,oldh=b,z,h
        b=np.clip((oldb+tb*(up.sum(axis=1)+anchor))/(1+tb),-base.BOUND,base.BOUND)
        z=np.clip(oldz-tz*(up-s*ua),lo,hi)
        hc=ua.copy(); hc[:-1]-=up[1:]
        h=np.clip(oldh-th*hc,hlow,hhigh)
        bb,zb,hb=2*b-oldb,2*z-oldz,2*h-oldh
        if (it+1)%check_every: continue
        err,move=base.score(b[None],x,v,anchor)
        if base.better(err,move,besterr,bestmove)[0]: best,besterr,bestmove=b.copy(),err,move
        bc,rest=components(up,ua)
        if certificate:
            candidates=[("raw_dual",up,ua)]
            for lag in [1,10]:
                if len(recent)>=lag:
                    prevp,preva=recent[-lag]
                    candidates.append((f"displacement_{lag*check_every}",up-prevp,ua-preva))
            for label,p,a in candidates:
                cb,cr=components(p,a)
                infeas=float(np.sum(cb*np.where(cb>=0,-base.BOUND,base.BOUND))+cr)
                scale=1+float(np.sum(np.abs(p))+np.sum(np.abs(a)))
                if infeas<=1e-10*scale: continue
                exact=exact_certificate(x,v,regs,p,a)
                if exact["positive"]:
                    cert={**exact,"dual_proposal":label,"up":p.tolist(),"ua":a.tolist()}
                    reason="certified_infeasible"; break
            if cert is not None: break
            recent.append((up.copy(),ua.copy()))
            if len(recent)>10: recent.pop(0)
        bmin=np.clip(anchor-bc,-base.BOUND,base.BOUND)
        dual=float(.5*np.sum((bmin-anchor)**2)+bc@bmin+rest)
        read=x; inside=True
        for j in range(depth):
            pre=read+best[j]
            inside=inside and bool(np.all(pre>=lo[j]-1e-10) and np.all(pre<=hi[j]+1e-10))
            read=base.g(pre)
        if besterr[0]<=base.EPS+base.TOL and inside:
            gap=float(bestmove[0]-dual)
            assert gap>=-2e-8,gap
            if gap<1e-7: reason="regional_gap"; break
    return best,{"polish_steps":it+1,"fixed_branch_duality_gap":gap,
                 "stop_reason":reason,"infeasibility_certificate":cert,
                 "support_max_error":float(besterr[0]),"regional_pattern":regs.tolist()}
