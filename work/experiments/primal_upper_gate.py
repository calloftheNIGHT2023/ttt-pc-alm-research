"""Safe cost gate for a fixed credit bank using relaxed-primal witnesses.

The gate never rejects a network region. It skips credit computation only
when every supplied direction has a certified nonpositive upper bound.
All uncertain cases retain the existing full computation.
"""
import time
import numpy as np
import interval_credit_certificate as iv

original=iv.original


def middle(a,b):return iv.mul(iv.add(a,b),iv.point(.5))


def witnesses(x,v,regs):
    """Enclose three exact, deterministic relaxed-primal witnesses.

Bounds are interpreted as binary rationals, as in exact_optimum. The exact
witness uses low / midpoint / high bias and midpoint previous activities.
"""
    rows,d,n=regs.shape;zl,zh,hl,hh=original.screen.boxes(v,regs)
    bs=[np.empty((3,rows,d)),np.empty((3,rows,d))]
    hs=[np.empty((3,rows,d,n)),np.empty((3,rows,d,n))]
    zs=[np.empty_like(hs[0]),np.empty_like(hs[1])]
    valid=np.all(hl<=hh,axis=(1,2))
    out=middle(iv.point(hl[:,-1]),iv.point(hh[:,-1]))
    for side in [0,1]:hs[side][:,:,-1]=out[side][None]
    for j in range(d):
        pl=np.broadcast_to(x,(rows,n)) if j==0 else hl[:,j-1]
        ph=np.broadcast_to(x,(rows,n)) if j==0 else hh[:,j-1]
        low=iv.maximum(iv.point(-original.screen.B),iv.reduce_max(iv.sub(iv.point(zl[:,j]),iv.point(ph)),1))
        high=iv.minimum(iv.point(original.screen.B),iv.reduce_min(iv.sub(iv.point(zh[:,j]),iv.point(pl)),1))
        valid&=low[1]<=high[0]
        mid=middle(low,high)
        b=(np.stack([low[0],mid[0],high[0]]),np.stack([low[1],mid[1],high[1]]))
        lo=iv.maximum(iv.point(pl[None]),iv.sub(iv.point(zl[None,:,j]),iv.expand(b,2)))
        hi=iv.minimum(iv.point(ph[None]),iv.sub(iv.point(zh[None,:,j]),iv.expand(b,2)))
        prev=iv.point(np.broadcast_to(x,(3,rows,n))) if j==0 else middle(lo,hi)
        z=iv.add(prev,iv.expand(b,2))
        for side in [0,1]:
            bs[side][:,:,j]=b[side];zs[side][:,:,j]=z[side]
            if j:hs[side][:,:,j-1]=prev[side]
    linear=iv.add(iv.mul(iv.point(original.base.SLOPES[regs][None]),tuple(zs)),iv.point(original.base.INTERCEPTS[regs][None]))
    residual=iv.sub(tuple(hs),linear)
    return dict(valid=valid,b=tuple(bs),h=tuple(hs),z=tuple(zs),residual=residual)


def upper_values(residual,bank):
    """Sequential outward upper dot products, then min over witnesses."""
    w,rows,d,n=residual[0].shape;flat=bank.reshape(len(bank),d*n)
    low=residual[0].reshape(w,rows,d*n);high=residual[1].reshape(w,rows,d*n)
    total=np.zeros((w,rows,len(bank)))
    for j in range(d*n):
        a=flat[:,j];chosen=np.where(a[None,None,:]>=0,high[:,:,j,None],low[:,:,j,None])
        term=iv.up(a[None,None,:]*chosen)
        term=np.where(a[None,None,:]==0,0.,term)
        total=iv.up(total+term)
    return total.min(0)


def gate(x,v,regs,bank):
    start=time.perf_counter();count=len(regs)
    if not count or not len(bank):
        return np.ones(count,bool),np.empty((count,len(bank))),dict(seconds=time.perf_counter()-start,valid_witness_rows=count,rough_candidates=count,skipped=count)
    witness=witnesses(x,v,regs);residual=witness['residual']
    midpoint=(residual[0]+residual[1])*.5
    rough=np.einsum('wrdn,kdn->wrk',midpoint,bank,optimize=True).min(0)
    # Floating arithmetic only nominates rows for strict verification.
    ids=np.flatnonzero(witness['valid']&np.all(rough<=0,axis=1))
    upper=np.full((count,len(bank)),np.inf)
    if len(ids):upper[ids]=upper_values((residual[0][:,ids],residual[1][:,ids]),bank)
    skip=witness['valid']&np.all(upper<=0,axis=1)
    return skip,upper,dict(seconds=time.perf_counter()-start,valid_witness_rows=int(witness['valid'].sum()),
        rough_candidates=len(ids),skipped=int(skip.sum()),residual_interval_bytes=residual[0].nbytes+residual[1].nbytes,
        scope='Skip fixed credit bank only; region still requires original geometry')
