"""Batched local interval contraction and PDHG box-separation screening.

PDHG / interval contractors are established algorithms, not PC-ALM novelty.
Only outward-rounded proofs reject; all other regions go to the original LP.
Every float input is interpreted as its exact binary-rational value.
"""
import numpy as np
import streaming_branch_projection as base

B=.12
down=lambda x:np.nextafter(x,-np.inf)
up=lambda x:np.nextafter(x,np.inf)


def boxes(v,regs):
    r,d,n=regs.shape
    zl=np.array([-B,0,.5,1.])[regs]
    zh=np.array([0,.5,1.,up(1+B)])[regs]
    hl=np.zeros((r,d,n)); hh=np.ones_like(hl)
    e=up(base.EPS+base.TOL)
    hl[:,-1]=np.maximum(0,down(v-e)); hh[:,-1]=np.minimum(1,up(v+e))
    return zl,zh,hl,hh


def forward_intervals(x,v,regs):
    zl,zh,hl,hh=boxes(v,regs); s=base.SLOPES[regs]; c=base.INTERCEPTS[regs]
    low=np.broadcast_to(x,regs.shape[::2]); high=low.copy(); reject=np.zeros(len(regs),dtype=bool)
    for j in range(regs.shape[1]):
        lo=np.maximum(zl[:,j],down(low-B)); hi=np.minimum(zh[:,j],up(high+B))
        reject|=np.any(lo>hi,axis=1)
        a=s[:,j]*lo; b=s[:,j]*hi
        low=np.maximum(hl[:,j],down(down(np.minimum(a,b))+c[:,j]))
        high=np.minimum(hh[:,j],up(up(np.maximum(a,b))+c[:,j]))
        reject|=np.any(low>high,axis=1)
    return reject


def contract(x,v,regs,rounds=5):
    """Bound consistency includes the shared bias across ALL observations."""
    zl,zh,hl,hh=boxes(v,regs); r,d,n=regs.shape
    bl=np.full((r,d),-B); bh=np.full((r,d),B)
    s=base.SLOPES[regs]; c=base.INTERCEPTS[regs]; reject=np.zeros(r,dtype=bool)
    for _ in range(rounds):
        for j in range(d):
            pl=x if j==0 else hl[:,j-1]; ph=x if j==0 else hh[:,j-1]
            zl[:,j]=np.maximum(zl[:,j],down(pl+bl[:,j,None]))
            zh[:,j]=np.minimum(zh[:,j],up(ph+bh[:,j,None]))
            a=s[:,j]*zl[:,j]; b=s[:,j]*zh[:,j]
            hl[:,j]=np.maximum(hl[:,j],down(down(np.minimum(a,b))+c[:,j]))
            hh[:,j]=np.minimum(hh[:,j],up(up(np.maximum(a,b))+c[:,j]))
        for j in reversed(range(d)):
            sj=s[:,j]; safe=np.where(sj==0,1,sj)
            al=down(hl[:,j]-c[:,j]); ah=up(hh[:,j]-c[:,j])
            il=down(np.where(sj>=0,al,ah)/safe); ih=up(np.where(sj>=0,ah,al)/safe)
            zl[:,j]=np.maximum(zl[:,j],np.where(sj!=0,il,-np.inf))
            zh[:,j]=np.minimum(zh[:,j],np.where(sj!=0,ih,np.inf))
            pl=x if j==0 else hl[:,j-1]; ph=x if j==0 else hh[:,j-1]
            bl[:,j]=np.maximum(bl[:,j],np.max(down(zl[:,j]-ph),axis=1))
            bh[:,j]=np.minimum(bh[:,j],np.min(up(zh[:,j]-pl),axis=1))
            if j:
                hl[:,j-1]=np.maximum(hl[:,j-1],down(zl[:,j]-bh[:,j,None]))
                hh[:,j-1]=np.minimum(hh[:,j-1],up(zh[:,j]-bl[:,j,None]))
        reject|=np.any(bl>bh,axis=1)|np.any(zl>zh,axis=(1,2))|np.any(hl>hh,axis=(1,2))
        # Retired rows no longer need tightening; reset to prevent overflow.
        if np.any(reject):
            zl[reject],zh[reject],hl[reject],hh[reject]=boxes(v,regs[reject])
            bl[reject]=-B; bh[reject]=B
    return reject


def float_bound(x,v,regs,p,a):
    zl,zh,hl,hh=boxes(v,regs); s=base.SLOPES[regs]; c=base.INTERCEPTS[regs]
    bc=-p.sum(axis=2); zc=p-s*a; hc=a.copy(); hc[:,:-1]-=p[:,1:]
    return -B*np.abs(bc).sum(1)+(zc*np.where(zc>=0,zl,zh)).sum((1,2))+(hc*np.where(hc>=0,hl,hh)).sum((1,2))-(p[:,0]*x).sum(1)-(a*c).sum((1,2))


def min_product(cl,ch,lo,hi):
    return down(np.minimum.reduce([cl*lo,cl*hi,ch*lo,ch*hi]))


def certified_lower_bound(x,v,regs,p,a):
    """IEEE754 nextafter lower enclosure; no loss backward or global Jacobian.

    Sequential sums deliberately avoid assumptions about BLAS/reduction order.
    NaNs / overflow never certify. Audited independently with Fraction.
    """
    zl,zh,hl,hh=boxes(v,regs); r,d,n=regs.shape
    s=base.SLOPES[regs]; c=base.INTERCEPTS[regs]
    pslo=np.zeros((r,d)); pshi=pslo.copy()
    for i in range(n): pslo=down(pslo+p[:,:,i]); pshi=up(pshi+p[:,:,i])
    bt=min_product(-pshi,-pslo,-B,B)
    sal=down(s*a); sah=up(s*a)
    zcl=down(p-sah); zch=up(p-sal)
    pcl=np.zeros_like(p); pcl[:,:-1]=p[:,1:]
    hcl=down(a-pcl); hch=up(a-pcl)
    zt=min_product(zcl,zch,zl,zh); ht=min_product(hcl,hch,hl,hh)
    ct=down(-a*c); xt=down(-p[:,0]*x)
    total=np.zeros(r)
    for j in range(d):
        total=down(total+bt[:,j])
        for i in range(n):
            total=down(total+zt[:,j,i]); total=down(total+ht[:,j,i]); total=down(total+ct[:,j,i])
    for i in range(n): total=down(total+xt[:,i])
    total[~np.isfinite(total)]=-np.inf
    return total


def pdhg(x,v,starts,regs,steps=60,check_every=20):
    r,d=starts.shape; n=len(x); zl,zh,hl,hh=boxes(v,regs)
    s=base.SLOPES[regs]; c=base.INTERCEPTS[regs]
    b=np.clip(starts,-B,B); z=np.empty((r,d,n)); h=np.empty_like(z); prev=x
    for j in range(d): z[:,j]=prev+b[:,j,None]; h[:,j]=base.g(z[:,j]); prev=h[:,j]
    p=np.zeros_like(h); a=p.copy(); bb=b.copy(); zb=z.copy(); hb=h.copy()
    tb=.99/n; tz=.99/(1+np.abs(s)); th=np.full((1,d,1),.99/2); th[:,-1]=.99
    sp=np.full((1,d,1),.99/3); sp[:,0]=.99/2; sa=.99/(1+np.abs(s))
    reject=np.zeros(r,dtype=bool); lower=np.full(r,-np.inf); proofs={}; lastp=p.copy(); lasta=a.copy()
    snapshots={}; byte_subtotal=sum(t.nbytes for t in [b,z,h,p,a,bb,zb,hb,zl,zh,hl,hh,s,c,tz,sa,lastp,lasta])
    for it in range(steps):
        prev=np.concatenate([np.broadcast_to(x,(r,1,n)),hb[:,:-1]],axis=1)
        p+=sp*(zb-prev-bb[:,:,None]); a+=sa*(hb-s*zb-c)
        oldb,oldz,oldh=b,z,h
        b=np.clip(oldb+tb*p.sum(axis=2),-B,B)
        z=np.clip(oldz-tz*(p-s*a),zl,zh)
        hc=a.copy(); hc[:,:-1]-=p[:,1:]; h=np.clip(oldh-th*hc,hl,hh)
        bb,zb,hb=2*b-oldb,2*z-oldz,2*h-oldh
        if (it+1)%check_every and it+1!=steps: continue
        for label,pp,aa in [("raw",p,a),("displacement",p-lastp,a-lasta)]:
            rough=float_bound(x,v,regs,pp,aa); ids=np.flatnonzero((rough>1e-10*(1+np.abs(pp).sum((1,2))+np.abs(aa).sum((1,2))))&~reject)
            if len(ids):
                lb=certified_lower_bound(x,v,regs[ids],pp[ids],aa[ids]); accepted=ids[lb>0]
                for idx,val in zip(accepted,lb[lb>0]):
                    proofs[int(idx)]={"p":pp[idx].copy(),"a":aa[idx].copy(),"lower":float(val),"step":it+1,"proposal":label}
                    lower[idx]=val
                reject[accepted]=True
        lastp=p.copy(); lasta=a.copy(); snapshots[it+1]=int(reject.sum())
    return reject,{"certificates":proofs,"steps":steps,"rejected_by_step":snapshots,"main_arrays_bytes_subtotal":byte_subtotal,
        "state_note":"temporary array subtotal, not peak native/Python allocation"}


def verify():
    from fractions import Fraction as F
    from certified_branch_solver import exact_certificate
    from scipy.optimize import linprog
    saved=base.BOUND; base.BOUND=B; rng=np.random.default_rng(16333)
    exact_checks=0; min_slack=np.inf; no_false=0
    try:
        for d,n in [(1,2),(2,4),(4,8),(4,24)]:
            x=rng.uniform(0,1,n); truth=rng.uniform(-B,B,d); v=base.forward(x,truth)
            starts=np.vstack([truth,rng.uniform(-B,B,(12,d))]); regs=np.stack([base.pattern(x,b) for b in starts])
            for _ in range(3):
                p=rng.normal(size=regs.shape); a=rng.normal(size=regs.shape); lower=certified_lower_bound(x,v,regs,p,a)
                for i in range(len(starts)):
                    exact=exact_certificate(x,v,regs[i],p[i],a[i]); ef=F(int(exact["numerator"]),int(exact["denominator"]))
                    assert F(float(lower[i]))<=ef
                    min_slack=min(min_slack,float(ef-F(float(lower[i])))); exact_checks+=1
            masks=[forward_intervals(x,v,regs),contract(x,v,regs,5),contract(x,v,regs,20),pdhg(x,v,starts,regs,60)[0]]
            for mask in masks:
                assert not mask[0]
                for i in np.flatnonzero(mask):
                    _,_,g,rhs=base.branch_polytope(x,v,starts[i]); lp=linprog(np.zeros(d),A_ub=g,b_ub=rhs,bounds=[(-B,B)]*d)
                    assert lp.status==2; no_false+=1
        # Per-observation intervals miss an elementary shared-parameter conflict.
        x=np.array([.45,.47]); v=np.array([.98,.98]); regs=np.ones((1,1,2),dtype=int)
        p=np.array([[[2.,-2.]]]); a=np.array([[[1.,-1.]]])
        lb=certified_lower_bound(x,v,regs,p,a)[0]; ex=exact_certificate(x,v,regs[0],p[0],a[0])
        assert not forward_intervals(x,v,regs)[0] and contract(x,v,regs,5)[0]
        assert .03799<lb<=ex["value"]
        # Crucially, this is NOT an example against the stronger contractor.
        return {"passed":True,"exact_enclosure_cases":exact_checks,"minimum_exact_minus_lower":min_slack,
            "rejections_independently_lp_checked":no_false,"simple_shared_conflict":{"lower":float(lb),"exact":ex,
                "forward_ibp_detects":False,"shared_interval_contractor_detects":True},
            "scope":"Local PDHG is established prior art; no PC-ALM specificity claimed."}
    finally: base.BOUND=saved


if __name__=="__main__":
    import json
    print(json.dumps(verify()))
