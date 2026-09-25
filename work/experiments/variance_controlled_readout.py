"""Geometry-only bounds and stratified posterior sampling.

Classical stratification / Neyman allocation, not a PC-specific invention.
Bounds concern finite readout error relative to the discovered-region posterior,
not missing modes, Bayes risk, or every random sample's realized error.
"""
import time
import numpy as np
import region_posterior_memory as geometry


def moments(poly):
    d=len(poly["center"]); k=d+1
    local=np.concatenate([poly["facets"],np.broadcast_to(poly["interior"],(len(poly["facets"]),1,d))],axis=1)
    vertices=poly["center"]+poly["scale"]*local
    means=vertices.mean(axis=1); centered=vertices-means[:,None,:]
    within=np.sum(centered**2,axis=1)/(k*(k+1))
    weights=poly["simplex_probs"].copy(); weights/=weights.sum()
    mean=weights@means
    diagonal=weights@(within+(means-mean)**2)
    assert np.all(diagonal>=-1e-14)
    return mean,np.maximum(diagonal,0)


def uniform_query_variance_bound(poly):
    mean,var=moments(poly); depth=len(mean)
    lipschitz=2.**np.arange(depth,0,-1)
    # Small numerical padding is not formal interval arithmetic. The theorem
    # assumes exact moments; geometry and reported computed bounds are float64.
    bound=min(.25,float(np.sum(lipschitz*np.sqrt(var+1e-18))**2))
    return bound,mean,var


def allocate(probabilities,bounds,budget,tolerance=None):
    assert len(probabilities)<=budget
    a2=probabilities**2*bounds; counts=np.ones(len(a2),dtype=int)
    while counts.sum()<budget:
        current=float(np.sum(a2/counts))
        if tolerance is not None and current<=tolerance: break
        gain=a2/(counts*(counts+1)); counts[int(np.argmax(gain))]+=1
    return counts,float(np.sum(a2/counts))


def weighted_predict(bank,weights):
    assert np.all(weights>=0) and abs(weights.sum()-1)<1e-12
    def predict(q):
        answer=np.empty(len(q))
        for first in range(0,len(q),256):
            h=np.broadcast_to(q[first:first+256],(len(bank),min(256,len(q)-first)))
            for j in range(bank.shape[1]): h=geometry.base.g(h+bank[:,j,None])
            answer[first:first+256]=np.einsum("i,ij->j",weights,h,optimize=False)
        return answer
    return predict


def build(polys,point,mode,tolerance=1e-6):
    begin=time.perf_counter()
    if not polys:
        return geometry.make_predict(point[None]),{"readout_seconds":time.perf_counter()-begin,"readout_samples":1,
            "readout_mc_variance_bound":None,"readout_tolerance_met":False,"readout_state_bytes":point.nbytes,"readout_fallback":True}
    volumes=np.array([p["volume"] for p in polys]); probability=volumes/volumes.sum()
    rng=np.random.default_rng(6173)
    if mode.startswith("iid") or mode=="adaptive_iid":
        if mode=="adaptive_iid":
            mv=[moments(p) for p in polys]; means=np.stack([a[0] for a in mv]); variances=np.stack([a[1] for a in mv])
            mean=probability@means; totalvar=probability@(variances+(means-mean)**2)
            lipschitz=2.**np.arange(len(mean),0,-1)
            totalbound=min(.25,float(np.sum(lipschitz*np.sqrt(totalvar+1e-18))**2))
            budget=max(1,min(512,int(np.ceil(totalbound/tolerance))))
        else: budget=int(mode[3:]); totalbound=.25
        counts=rng.multinomial(budget,probability)
        bank=np.concatenate([geometry.sample(p,int(n),rng) for p,n in zip(polys,counts) if n])
        predict=geometry.make_predict(bank); bound=totalbound/budget; state=bank.nbytes
    else:
        bounds=np.array([uniform_query_variance_bound(p)[0] for p in polys])
        budget=512 if mode=="adaptive" else int(mode[5:])
        assert len(polys)<=budget, "More modes than strata budget: protocol must address this explicitly"
        counts,bound=allocate(probability,bounds,budget,tolerance if mode=="adaptive" else None)
        bank=np.concatenate([geometry.sample(p,int(n),rng) for p,n in zip(polys,counts)])
        weights=np.concatenate([np.full(n,p/n) for p,n in zip(probability,counts)])
        predict=weighted_predict(bank,weights); state=bank.nbytes+weights.nbytes
    return predict,{"readout_seconds":time.perf_counter()-begin,"readout_samples":int(counts.sum()),
        "readout_mc_variance_bound":bound,"readout_tolerance_met":bool(bound<=tolerance),
        "readout_state_bytes":state,"readout_fallback":False,"readout_allocation":counts.tolist()}


def verify():
    import itertools
    rng=np.random.default_rng(59448); moment_errors=[]; allocation_cases=0
    for d in [2,3,4]:
        p,_=geometry.polytope(np.empty((0,d)),np.empty(0)); mean,var=moments(p)
        err=max(np.max(np.abs(mean)),np.max(np.abs(var-.12**2/3))); assert err<1e-12; moment_errors.append(float(err))
    for _ in range(30):
        probability=rng.dirichlet(np.ones(3)); bounds=rng.uniform(0,.25,3)
        for budget in range(3,9):
            counts,value=allocate(probability,bounds,budget)
            exact=min(np.sum(probability**2*bounds/np.array(c)) for c in itertools.product(range(1,budget+1),repeat=3) if sum(c)==budget)
            assert abs(value-exact)<1e-14
            allocation_cases+=1
    # An affine coordinate readout has exactly computable posterior mean and
    # variance. This audit does not use experimental query labels.
    p,_=geometry.polytope(np.empty((0,4)),np.empty(0)); _,var=moments(p)
    samples=geometry.sample(p,30000,rng); error=float(abs(np.var(samples[:,0])-var[0]))
    assert error<2e-4
    return {"passed":True,"exact_box_moment_errors":moment_errors,"bruteforce_optimal_integer_allocations":allocation_cases,
        "sampled_coordinate_variance_error":error,"scope":"Exact-moment theorem; floating geometry implementation, not a formal machine certificate."}
