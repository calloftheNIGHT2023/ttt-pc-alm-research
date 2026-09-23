"""Upper-only arithmetic and layer-pattern reuse for midpoint primal gate.

All arithmetic that can round is directed upward as needed. Slope/intercept
multiplication by 0 or +/-2 is exact for finite bounded float64 inputs.
No unverified floating gate grants a skip; every returned value is an upper.
"""
import time
import numpy as np
import directional_primal_upper_gate as parent

iv=parent.iv
original=parent.original


def upper_sum(value):
    while value.shape[-1]>1:
        m=value.shape[-1]//2
        reduced=iv.up(value[...,:2*m:2]+value[...,1:2*m:2])
        if value.shape[-1]%2:reduced=np.concatenate([reduced,value[...,-1:]],-1)
        value=reduced
    return value[...,0]


class Bank:
    def __init__(self,x,v,directions):
        self.x=np.asarray(x);self.v=np.asarray(v);self.bank=np.asarray(directions);self.k,self.d,self.n=self.bank.shape
        assert np.all(np.isfinite(self.bank)) and np.max(abs(self.bank),initial=0)<1e100
        _,_,hl,hh=original.screen.boxes(v,np.zeros((1,self.d,self.n),np.uint8))
        aa=self.bank[:,-1];chosen=np.where(aa>=0,hl[0,-1],hh[0,-1])
        term=np.where(aa==0,0.,iv.up(aa*chosen));self.output=upper_sum(term)
        self.output_valid=bool(np.all(hl<=hh));self.cache=[{} for _ in range(self.d)];self.layer_solves=0

    def layer_upper(self,j,patterns):
        u=len(patterns);B=original.screen.B
        zl=np.array([-B,0.,.5,1.])[patterns];zh=np.array([0.,.5,1.,original.screen.up(1+B)])[patterns]
        pl=self.x if j==0 else np.zeros(self.n);ph=self.x if j==0 else np.ones(self.n)
        low=iv.maximum(iv.point(-B),iv.reduce_max(iv.sub(iv.point(zl),iv.point(ph)),1))
        high=iv.minimum(iv.point(B),iv.reduce_min(iv.sub(iv.point(zh),iv.point(pl)),1))
        valid=low[1]<=high[0];b=parent.shared.middle(low,high)
        lo=iv.maximum(iv.point(pl),iv.sub(iv.point(zl),iv.expand(b,1)))
        hi=iv.minimum(iv.point(ph),iv.sub(iv.point(zh),iv.expand(b,1)))
        # All exact previous activities are nonnegative. Intersect the
        # interval enclosures with their known [pl,ph] domain.
        lo=(np.maximum(lo[0],pl),np.minimum(lo[1],ph));hi=(np.maximum(hi[0],pl),np.minimum(hi[1],ph))
        aa=self.bank[None,:,j,:]
        sa=original.base.SLOPES[patterns][:,None,:]*aa
        prev=np.zeros((self.k,self.n)) if j==0 else self.bank[:,j-1]
        nonnegative=prev[None]>=sa;zero=prev[None]==sa
        cup=iv.up(prev[None]-sa);cup=np.where(zero,0.,cup)
        # Positive c chooses the lower activity endpoint; negative c chooses
        # upper. The upper product uses opposite interval endpoints when c<0.
        hbound=np.where(nonnegative,lo[1][:,None,:],hi[0][:,None,:])
        activity=iv.up(cup*hbound);activity=np.where(zero,0.,activity)
        bias_coefficient=-sa
        bbound=np.where(bias_coefficient>=0,b[1][:,None,None],b[0][:,None,None])
        bias=iv.up(bias_coefficient*bbound);bias=np.where(bias_coefficient==0,0.,bias)
        constant=aa*original.base.INTERCEPTS[patterns][:,None,:]
        terms=iv.up(iv.up(activity+bias)-constant)
        values=upper_sum(terms);values[~valid]=np.inf
        return values,valid

    def values(self,regs):
        if not len(regs):return np.empty((0,self.k))
        assert regs.shape[1:]==(self.d,self.n)
        total=np.broadcast_to(self.output,(len(regs),self.k)).copy()
        valid=np.full(len(regs),self.output_valid)
        for j in range(self.d):
            keys=[r[j].tobytes() for r in regs];new=list(dict.fromkeys(k for k in keys if k not in self.cache[j]))
            if new:
                patterns=np.array([np.frombuffer(k,np.uint8) for k in new]);values,flags=self.layer_upper(j,patterns)
                for key,row,flag in zip(new,values,flags):self.cache[j][key]=(row.copy(),bool(flag))
                self.layer_solves+=len(new)
            total=iv.up(total+np.array([self.cache[j][key][0] for key in keys]));valid&=np.array([self.cache[j][key][1] for key in keys])
        total[~valid]=np.inf
        return total


def gate(x,v,regs,bank):
    start=time.perf_counter();solver=Bank(x,v,bank);upper=solver.values(regs);skip=np.all(upper<=0,axis=1)
    return skip,upper,dict(seconds=time.perf_counter()-start,skipped=int(skip.sum()),rough_candidates=None,
        upper_layer_solves=solver.layer_solves,cached_upper_bytes=sum(len(k)+val[0].nbytes+1 for cache in solver.cache for k,val in cache.items()))
