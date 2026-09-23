"""Remove duplicate bias candidates without changing a single float bound.

Specific to fixed [0,1] hidden boxes, tent branches and 0 < B < 1/2. The
first layer needs the two interval endpoints only; hidden layers additionally
need zero iff an original kink equals zero. Other clipped kinks are endpoints.
"""
import numpy as np
import factorized_credit_bank as parent


class Bank(parent.Bank):
    def layer_values(self,j,patterns):
        u=len(patterns);B=parent.original.screen.B;assert 0<B<.5
        zl=np.array([-B,0.,.5,1.])[patterns]
        zh=np.array([0.,.5,1.,parent.original.screen.up(1+B)])[patterns]
        pl=self.x if j==0 else np.zeros(self.n);ph=self.x if j==0 else np.ones(self.n)
        previous=np.zeros((self.k,self.n)) if j==0 else self.bank[:,j-1]
        sa=parent.original.base.SLOPES[patterns][:,None,:]*self.bank[None,:,j,:];c=previous[None]-sa
        low=np.maximum(-B,(zl-ph).max(1));high=np.minimum(B,(zh-pl).min(1));invalid=low>high
        candidates=[np.broadcast_to(low[:,None,None],(u,self.k,1)),np.broadcast_to(high[:,None,None],(u,self.k,1))]
        if j:
            kink=np.where(c>=0,zl[:,None,:]-pl,zh[:,None,:]-ph)
            middle=np.where(np.any(kink==0.,axis=2),0.,low[:,None])
            candidates.append(middle[:,:,None])
        b=np.concatenate(candidates,axis=2);b=np.clip(b,low[:,None,None],high[:,None,None])
        lo=np.maximum(pl,zl[:,None,None,:]-b[:,:,:,None])
        hi=np.minimum(ph,zh[:,None,None,:]-b[:,:,:,None])
        values=(c[:,:,None,:]*np.where(c[:,:,None,:]>=0,lo,hi)-sa[:,:,None,:]*b[:,:,:,None]).sum(3)
        return values.min(2),invalid


def verify():
    rng=np.random.default_rng(482431);checks=0;candidate_sets=0
    for d,n in [(1,2),(2,3),(4,4),(6,5),(4,24)]:
        x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);a=rng.normal(size=(7,d,n))
        bs=rng.uniform(-.12,.12,(25,d))
        regs=np.r_[np.array([parent.original.base.pattern(x,b) for b in bs],np.uint8),rng.integers(0,4,(17,d,n),dtype=np.uint8)]
        expected=parent.Bank(x,v,a).values(regs);actual=Bank(x,v,a).values(regs)
        assert actual.tobytes()==expected.tobytes();checks+=actual.size
        # Check candidate-set equality directly, including infeasible boxes.
        B=parent.original.screen.B
        for j in range(d):
            rr=regs[:,j];zl=np.array([-B,0.,.5,1.])[rr];zh=np.array([0.,.5,1.,parent.original.screen.up(1+B)])[rr]
            pl=x if j==0 else np.zeros(n);ph=x if j==0 else np.ones(n)
            previous=np.zeros((len(a),n)) if j==0 else a[:,j-1]
            c=previous[None]-parent.original.base.SLOPES[rr][:,None]*a[None,:,j]
            low=np.maximum(-B,(zl-ph).max(1));high=np.minimum(B,(zh-pl).min(1))
            kink=np.where(c>=0,zl[:,None]-pl,zh[:,None]-ph)
            old=np.clip(np.concatenate([np.broadcast_to(low[:,None,None],(len(rr),len(a),1)),np.broadcast_to(high[:,None,None],(len(rr),len(a),1)),kink],2),low[:,None,None],high[:,None,None])
            mid=np.where(np.any(kink==0,axis=2),0.,low[:,None]) if j else np.broadcast_to(low[:,None],(len(rr),len(a)))
            new=np.clip(np.stack([np.broadcast_to(low[:,None],mid.shape),np.broadcast_to(high[:,None],mid.shape),mid],2),low[:,None,None],high[:,None,None])
            for aa,bb in zip(old.reshape(-1,n+2),new.reshape(-1,3)):
                assert set(aa)==set(bb);candidate_sets+=1
    return dict(passed=True,float_values_bitwise=checks,candidate_sets_exact=candidate_sets,
        scope='specific fixed-box tent algebra; no timing or online claim')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
