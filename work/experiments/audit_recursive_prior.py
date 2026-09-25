"""Noise-aware kernel recursive least squares equals batch prior-moment readout."""
import json
import numpy as np
import streaming_branch_projection as base
import region_posterior_memory as posterior


def verify():
    rng=np.random.default_rng(89543); bank=np.random.default_rng(731).uniform(-.12,.12,(1024,4))
    x=rng.uniform(0,1,24); v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,24)
    q=np.linspace(0,1,1024)
    def values(inputs):
        h=np.broadcast_to(inputs,(len(bank),len(inputs)))
        for j in range(4): h=base.g(h+bank[:,j,None])
        return h
    phi=values(x); centered=phi-phi.mean(axis=0)
    inverse=np.zeros((0,0)); alpha=np.empty(0); errors=[]; min_schur=float("inf")
    for n in range(1,25):
        new=centered[:,n-1]; covariance=centered[:,:n-1].T@new/len(bank)
        diagonal=float(new@new/len(bank)+base.EPS**2/3)
        z=inverse@covariance; schur=diagonal-covariance@z; assert schur>0
        min_schur=min(min_schur,schur)
        innovation=v[n-1]-phi[:,n-1].mean()-covariance@alpha
        alpha=np.r_[alpha-z*innovation/schur,innovation/schur]
        inverse=np.block([[inverse+np.outer(z,z)/schur,-z[:,None]/schur],[-z[None,:]/schur,np.array([[1/schur]])]])
        if n not in [4,8,16,24]: continue
        weights=(1+centered[:,:n]@alpha)/len(bank)
        direct,_=posterior.prior_moments(x[:n],v[:n],4,features=1024)
        err=float(np.max(np.abs(weights@values(q)-direct(q)))); assert err<1e-8,err
        errors.append({"n_context":n,"max_prediction_difference":err})
    return {"passed":True,"checks":errors,"minimum_positive_schur":float(min_schur),
        "interpretation":"This task's noise-aware fixed-prior-feature ridge can be implemented recursively; it is not limited to batch closed-form adaptation."}


if __name__=="__main__": print(json.dumps(verify()))
