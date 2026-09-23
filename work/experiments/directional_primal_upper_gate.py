"""Tighter safe upper gate: midpoint shared biases, directional activities.

Fixing each layer bias leaves all activity minimizers analytic. This is an
upper bound on the full minimum, NOT a positive infeasibility certificate.
"""
import time
import numpy as np
import primal_upper_gate as shared

iv=shared.iv
original=iv.original


def pair_sum(a):
    """Outward-rounded fixed binary-tree reduction, without BLAS assumptions."""
    lo,hi=a
    while lo.shape[-1]>1:
        m=lo.shape[-1]//2
        ll=iv.down(lo[...,:2*m:2]+lo[...,1:2*m:2]);hh=iv.up(hi[...,:2*m:2]+hi[...,1:2*m:2])
        if lo.shape[-1]%2:ll=np.concatenate([ll,lo[...,-1:]],-1);hh=np.concatenate([hh,hi[...,-1:]],-1)
        lo,hi=ll,hh
    return lo[...,0],hi[...,0]


def float_upper(x,v,regs,bank):
    rows,d,n=regs.shape;k=len(bank);zl,zh,hl,hh=original.screen.boxes(v,regs)
    total=np.minimum(bank[None,:,-1]*hl[:,None,-1],bank[None,:,-1]*hh[:,None,-1]).sum(2)
    total-=(bank[None]*original.base.INTERCEPTS[regs][:,None]).sum((2,3));valid=np.all(hl<=hh,axis=(1,2))
    for j in range(d):
        pl=np.broadcast_to(x,(rows,n)) if j==0 else hl[:,j-1];ph=np.broadcast_to(x,(rows,n)) if j==0 else hh[:,j-1]
        low=np.maximum(-original.screen.B,(zl[:,j]-ph).max(1));high=np.minimum(original.screen.B,(zh[:,j]-pl).min(1))
        valid&=low<=high;b=(low+high)/2
        lo=np.maximum(pl,zl[:,j]-b[:,None]);hi=np.minimum(ph,zh[:,j]-b[:,None])
        sa=original.base.SLOPES[regs[:,j]][:,None]*bank[None,:,j]
        c=(np.zeros((k,n)) if j==0 else bank[:,j-1])[None]-sa
        total+=(np.minimum(c*lo[:,None],c*hi[:,None])-sa*b[:,None,None]).sum(2)
    total[~valid]=np.inf
    return total,valid


def strict_upper(x,v,regs,bank):
    rows,d,n=regs.shape;k=len(bank);zl,zh,hl,hh=original.screen.boxes(v,regs)
    total=pair_sum(iv.minimum(iv.mul(iv.point(bank[None,:,-1]),iv.point(hl[:,None,-1])),
        iv.mul(iv.point(bank[None,:,-1]),iv.point(hh[:,None,-1]))))
    constant=iv.mul(iv.point(bank[None]),iv.point(original.base.INTERCEPTS[regs][:,None]))
    total=iv.sub(total,pair_sum((constant[0].reshape(rows,k,-1),constant[1].reshape(rows,k,-1))))
    valid=np.all(hl<=hh,axis=(1,2))
    for j in range(d):
        pl=np.broadcast_to(x,(rows,n)) if j==0 else hl[:,j-1];ph=np.broadcast_to(x,(rows,n)) if j==0 else hh[:,j-1]
        low=iv.maximum(iv.point(-original.screen.B),iv.reduce_max(iv.sub(iv.point(zl[:,j]),iv.point(ph)),1))
        high=iv.minimum(iv.point(original.screen.B),iv.reduce_min(iv.sub(iv.point(zh[:,j]),iv.point(pl)),1))
        valid&=low[1]<=high[0];b=shared.middle(low,high)
        lo=iv.maximum(iv.point(pl),iv.sub(iv.point(zl[:,j]),iv.expand(b,1)))
        hi=iv.minimum(iv.point(ph),iv.sub(iv.point(zh[:,j]),iv.expand(b,1)))
        sa=iv.mul(iv.point(original.base.SLOPES[regs[:,j]][:,None]),iv.point(bank[None,:,j]))
        prev=np.zeros((k,n)) if j==0 else bank[:,j-1];c=iv.sub(iv.point(prev[None]),sa)
        value=iv.sub(iv.minimum(iv.mul(c,iv.expand(lo,1)),iv.mul(c,iv.expand(hi,1))),iv.mul(sa,iv.expand(iv.expand(b,1),2)))
        total=iv.add(total,pair_sum(value))
    lower,upper=total;lower[~valid]=-np.inf;upper[~valid]=np.inf
    return lower,upper,valid


def gate(x,v,regs,bank):
    start=time.perf_counter()
    if not len(regs) or not len(bank):return np.ones(len(regs),bool),np.empty((len(regs),len(bank))),dict(seconds=time.perf_counter()-start,skipped=len(regs),rough_candidates=len(regs))
    rough,valid=float_upper(x,v,regs,bank);ids=np.flatnonzero(valid&np.all(rough<=0,axis=1));upper=np.full_like(rough,np.inf)
    if len(ids):_,upper[ids],_=strict_upper(x,v,regs[ids],bank)
    skip=np.all(upper<=0,axis=1)
    return skip,upper,dict(seconds=time.perf_counter()-start,valid_witness_rows=int(valid.sum()),rough_candidates=len(ids),skipped=int(skip.sum()))
