"""Cached primal upper with explicit absolute roundoff allowance.

Fast domain: |a|<=1, x in [0,1], fixed tent boxes B=.12, n<=4096.
One layer has pointwise error <=256*n*eps +16*n*n*eps; the deliberately
loose allowance 1024*(n+1)^2*eps dominates it. Fixed binary-tree sums avoid
assumptions about BLAS reduction internals. Unsupported input falls back.
"""
import time
import numpy as np
import factorized_primal_upper_gate as parent

iv=parent.iv
original=parent.original


def tree_sum(value):
    while value.shape[-1]>1:
        m=value.shape[-1]//2;reduced=value[...,:2*m:2]+value[...,1:2*m:2]
        if value.shape[-1]%2:reduced=np.concatenate([reduced,value[...,-1:]],-1)
        value=reduced
    return value[...,0]


class Bank(parent.Bank):
    def __init__(self,x,v,directions):
        super().__init__(x,v,directions)
        self.fast=bool(0<self.n<=4096 and np.all((self.x>=0)&(self.x<=1)) and np.max(abs(self.bank),initial=0)<=1 and original.screen.B==.12)
        self.layer_error=1024.*(self.n+1)**2*np.finfo(np.float64).eps

    def layer_upper(self,j,patterns):
        if not self.fast:return super().layer_upper(j,patterns)
        B=original.screen.B;zl=np.array([-B,0.,.5,1.])[patterns];zh=np.array([0.,.5,1.,original.screen.up(1+B)])[patterns]
        pl=self.x if j==0 else np.zeros(self.n);ph=self.x if j==0 else np.ones(self.n)
        lower_dif=zl-ph;upper_dif=zh-pl
        # Separately certify that the EXACT shared-bias interval is nonempty.
        safe_low=np.maximum(-B,iv.up(lower_dif).max(1));safe_high=np.minimum(B,iv.down(upper_dif).min(1))
        valid=safe_low<=safe_high
        low=np.maximum(-B,lower_dif.max(1));high=np.minimum(B,upper_dif.min(1));b=(low+high)*.5
        lo=np.maximum(pl,zl-b[:,None]);hi=np.minimum(ph,zh-b[:,None])
        aa=self.bank[None,:,j];sa=original.base.SLOPES[patterns][:,None]*aa
        prev=np.zeros((self.k,self.n)) if j==0 else self.bank[:,j-1];c=prev[None]-sa
        terms=np.minimum(c*lo[:,None],c*hi[:,None])-sa*b[:,None,None]-aa*original.base.INTERCEPTS[patterns][:,None]
        upper=iv.up(tree_sum(terms)+self.layer_error);upper[~valid]=np.inf
        return upper,valid


def gate(x,v,regs,bank):
    start=time.perf_counter();solver=Bank(x,v,bank);upper=solver.values(regs);skip=np.all(upper<=0,axis=1)
    return skip,upper,dict(seconds=time.perf_counter()-start,skipped=int(skip.sum()),rough_candidates=None,
        upper_layer_solves=solver.layer_solves,fast_bounded_error=solver.fast,layer_absolute_error_allowance=solver.layer_error,
        cached_upper_bytes=sum(len(k)+value[0].nbytes+1 for cache in solver.cache for k,value in cache.items()))
