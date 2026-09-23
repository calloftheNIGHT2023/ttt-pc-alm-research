"""Local fixed-region primal-dual solver with rational infeasibility certificates.

No global Jacobian/LP is used in this solver. Audits may separately call LP.
"""
from __future__ import annotations
from fractions import Fraction as F
import numpy as np
import streaming_branch_projection as base


def exact_certificate(x,v,regs,up,ua):
    """Strict rational lower bound for the accepted EPS+TOL domain.

    All supplied floats become exact binary rationals, and all operations in
    this proof calculation are exact. Complexity is linear in local state size.
    """
    depth,n=regs.shape
    p=[[F(float(up[j,i])) for i in range(n)] for j in range(depth)]
    a=[[F(float(ua[j,i])) for i in range(n)] for j in range(depth)]
    bound=F(base.BOUND); eps=F(base.EPS)+F(base.TOL)
    lows=[-bound,F(0),F(1,2),F(1)]
    highs=[F(0),F(1,2),F(1),1+bound]
    slopes=[0,2,-2,0]; offsets=[0,0,2,0]
    total=F(0)
    def at_min(coef,lo,hi): return coef*(lo if coef>=0 else hi)
    for j in range(depth):
        bc=-sum(p[j],F(0))
        total+=at_min(bc,-bound,bound)
        for i in range(n):
            reg=int(regs[j,i])
            zc=p[j][i]-slopes[reg]*a[j][i]
            hc=a[j][i]-(p[j+1][i] if j<depth-1 else 0)
            total+=at_min(zc,lows[reg],highs[reg])
            hlo=max(F(0),F(float(v[i]))-eps) if j==depth-1 else F(0)
            hhi=min(F(1),F(float(v[i]))+eps) if j==depth-1 else F(1)
            total+=at_min(hc,hlo,hhi)-a[j][i]*offsets[reg]
            if j==0: total-=p[j][i]*F(float(x[i]))
    return {"positive":total>0,"value":float(total),"numerator":str(total.numerator),
            "denominator":str(total.denominator),"arithmetic":"exact binary-rational inputs"}


def solve(x,v,anchor,start,steps=8000,certificate=True,check_every=20):
    depth,n=len(anchor),len(x)
    regs=base.pattern(x,start)
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
        bc=-up.sum(axis=1); zc=up-s*ua
        rest=float(np.sum(zc*np.where(zc>=0,lo,hi))+np.sum(hc*np.where(hc>=0,dlo,dhi))-up[0]@x-np.sum(ua*c))
        infeas=float(np.sum(bc*np.where(bc>=0,-base.BOUND,base.BOUND))+rest)
        scale=1+float(np.sum(np.abs(up))+np.sum(np.abs(ua)))
        if certificate and infeas>1e-10*scale:
            exact=exact_certificate(x,v,regs,up,ua)
            if exact["positive"]:
                cert=exact; reason="certified_infeasible"; break
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
