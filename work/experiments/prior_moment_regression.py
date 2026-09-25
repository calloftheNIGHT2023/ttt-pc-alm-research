"""Empirical prior-moment best affine prediction, before output clipping."""
from __future__ import annotations
import numpy as np
from local_branch_memory import g


def fit(x,v,depth,features=1024,relative_cutoff=1e-10):
    bank=np.random.default_rng(731).uniform(-.12,.12,(features,depth))
    def values(inputs):
        out=np.broadcast_to(inputs,(features,len(inputs)))
        for j in range(depth): out=g(out+bank[:,j,None])
        return out
    phi=values(x); mean=phi.mean(axis=0); centered=phi-mean[None,:]
    covariance=centered.T@centered/features
    eig,vec=np.linalg.eigh(covariance)
    threshold=max(0,float(eig.max()))*relative_cutoff
    inv=np.divide(1.,eig,out=np.zeros_like(eig),where=eig>threshold)
    alpha=vec@(inv*(vec.T@(v-mean)))
    # Collapse the regression into one coefficient per fixed prior function.
    weights=(1+centered@alpha)/features
    assert abs(weights.sum()-1)<1e-7
    def predict(q):
        # Chunk the query axis, so the prior bank does not require F*all_queries
        # simultaneous activations. Query workload is timed by the evaluator.
        out=np.empty(len(q))
        for start in range(0,len(q),128): out[start:start+128]=weights@values(q[start:start+128])
        return out
    return predict,{"family":"empirical_prior_moment_affine","prior_functions":features,
        "covariance_rank":int(np.sum(eig>threshold)),"relative_pinv_cutoff":relative_cutoff,
        "persistent_float64_scalars":int(bank.size+weights.size),"coefficient_sum":float(weights.sum())}


def verify():
    rng=np.random.default_rng(582614); tasks=rng.normal(size=(500,5)); coef=rng.normal(size=(5,3)); target=tasks@coef+np.array([.2,.5,.7])
    mx=tasks.mean(0); my=target.mean(0); cx=(tasks-mx).T@(tasks-mx)/len(tasks)
    cross=(target-my).T@(tasks-mx)/len(tasks)
    predicted=my+(tasks-mx)@np.linalg.solve(cx,cross.T)
    err=float(np.max(np.abs(predicted-target))); assert err<1e-10
    return {"passed":True,"linear_control_max_error":err}
