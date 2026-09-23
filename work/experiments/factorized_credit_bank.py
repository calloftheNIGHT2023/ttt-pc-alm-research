"""Exact same optimized float bound, sharing repeated layer-pattern solves.

Each optimized layer term depends only on that layer's branch row and the
fixed adjacent credit rows. Whole-region assembly preserves the original
reduction order. Positive float values still require rational certification.
"""
import time
import numpy as np
import optimized_branch_dual as original


class Bank:
    def __init__(self,x,v,directions):
        self.x=np.array(x,copy=True);self.v=np.array(v,copy=True);self.bank=np.array(directions,copy=True)
        self.k,self.d,self.n=self.bank.shape
        _,_,hl,hh=original.screen.boxes(v,np.zeros((1,self.d,self.n),np.uint8))
        self.output=np.minimum(self.bank[:,-1]*hl[0,-1],self.bank[:,-1]*hh[0,-1]).sum(1)
        self.cache=[{} for _ in range(self.d)];self.layer_solves=0;self.calls=0;self.region_rows=0

    def layer_values(self,j,patterns):
        u=len(patterns);B=original.screen.B
        zl=np.array([-B,0.,.5,1.])[patterns]
        zh=np.array([0.,.5,1.,original.screen.up(1+B)])[patterns]
        pl=self.x if j==0 else np.zeros(self.n);ph=self.x if j==0 else np.ones(self.n)
        previous=np.zeros((self.k,self.n)) if j==0 else self.bank[:,j-1]
        sa=original.base.SLOPES[patterns][:,None,:]*self.bank[None,:,j,:];c=previous[None]-sa
        low=np.maximum(-B,(zl-ph).max(1));high=np.minimum(B,(zh-pl).min(1));invalid=low>high
        kink=np.where(c>=0,zl[:,None,:]-pl,zh[:,None,:]-ph)
        b=np.concatenate([np.broadcast_to(low[:,None,None],(u,self.k,1)),
            np.broadcast_to(high[:,None,None],(u,self.k,1)),kink],axis=2)
        b=np.clip(b,low[:,None,None],high[:,None,None])
        lo=np.maximum(pl,zl[:,None,None,:]-b[:,:,:,None])
        hi=np.minimum(ph,zh[:,None,None,:]-b[:,:,:,None])
        values=(c[:,:,None,:]*np.where(c[:,:,None,:]>=0,lo,hi)-sa[:,:,None,:]*b[:,:,:,None]).sum(3)
        return values.min(2),invalid

    def values(self,regs):
        if not len(regs):return np.empty((0,self.k))
        assert regs.shape[1:]==(self.d,self.n)
        # Keep the original contiguous (layer, observation) summation order.
        constant=(self.bank[None]*original.base.INTERCEPTS[regs][:,None]).sum((2,3))
        total=self.output[None]-constant;invalid=np.zeros(len(regs),bool)
        for j in range(self.d):
            keys=[r[j].tobytes() for r in regs];new=list(dict.fromkeys(k for k in keys if k not in self.cache[j]))
            if new:
                patterns=np.array([np.frombuffer(k,np.uint8) for k in new]);mu,bad=self.layer_values(j,patterns)
                for key,row,flag in zip(new,mu,bad):self.cache[j][key]=(row.copy(),bool(flag))
                self.layer_solves+=len(new)
            total+=np.array([self.cache[j][key][0] for key in keys])
            invalid|=np.array([self.cache[j][key][1] for key in keys])
        total[invalid]=np.inf;self.calls+=1;self.region_rows+=len(regs)
        return total

    def screen(self,regs):
        start=time.perf_counter();values=self.values(regs);float_seconds=time.perf_counter()-start
        if not len(regs) or not self.k:return np.zeros(len(regs),bool),[],dict(float_seconds=float_seconds,exact_seconds=0.)
        selected=values.argmax(1);rough=values[np.arange(len(regs)),selected]
        scale=1+abs(self.bank[selected]).sum((1,2));reject=np.zeros(len(regs),bool);proofs=[];start=time.perf_counter()
        for i in np.flatnonzero(rough>1e-10*scale):
            a=self.bank[selected[i]];exact=original.exact_optimum(self.x,self.v,regs[i],a)
            if not exact['positive']:continue
            reject[i]=True;proofs.append(dict(pattern=regs[i].tobytes().hex(),direction=int(selected[i]),exact=exact))
        return reject,proofs,dict(float_seconds=float_seconds,exact_seconds=time.perf_counter()-start,
            layer_solves=self.layer_solves,calls=self.calls,region_rows=self.region_rows,
            cached_values_numeric_bytes=sum(len(k)+v[0].nbytes+1 for cache in self.cache for k,v in cache.items()))


def verify():
    rng=np.random.default_rng(482117);checks=0;cache_checks=0;max_error=0.
    for d,n in [(1,2),(2,3),(4,4),(6,5)]:
        x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);directions=rng.normal(size=(7,d,n))
        b=rng.uniform(-.12,.12,(21,d));regs=np.array([original.base.pattern(x,z) for z in b],np.uint8)
        regs=np.r_[regs,rng.integers(0,4,(11,d,n),dtype=np.uint8)]
        bb=Bank(x,v,directions);actual=bb.values(regs)
        expected=original.float_optimum(x,v,np.repeat(regs,7,axis=0),np.tile(directions,(len(regs),1,1))).reshape(len(regs),7)
        finite=np.isfinite(expected);max_error=max(max_error,float(np.max(abs(actual[finite]-expected[finite]),initial=0)))
        assert np.array_equal(actual,expected),(d,n,max_error)
        before=bb.layer_solves;assert np.array_equal(bb.values(regs[::-1]),expected[::-1]) and bb.layer_solves==before;cache_checks+=1
        cc=Bank(x,v,directions);first=cc.values(regs[:13]);second=cc.values(regs[13:])
        assert np.array_equal(np.r_[first,second],expected);checks+=len(regs)*7
    return dict(passed=True,float_values_bitwise=checks,reverse_cache_no_new_solves=cache_checks,
        maximum_error=max_error,scope='generic bound algebra and repeated-row cache, not online timing or task result')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
