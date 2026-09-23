"""Truncated uniform posterior over discovered feasible activation polytopes.

Exact likelihood model: independent Uniform[-EPS, EPS] observation noise and
Uniform[-.12,.12]^depth parameter prior. Unknown undiscovered posterior mass is
NOT assumed zero in interpretation. Geometry is global, not all-local PC.
"""
from __future__ import annotations
import math,time
import numpy as np
from scipy.optimize import linprog,minimize
from scipy.spatial import ConvexHull,HalfspaceIntersection,QhullError
import streaming_branch_projection as base
from hybrid_discovery_bank import discover as local_discover
from matched_discovery_baselines import discover as bp_discover

PRIOR_BOUND=.12


def polytope(g,rhs):
    """Volume and simplex decomposition after reversible axis scaling."""
    d=g.shape[1]; a=np.r_[g,np.eye(d),-np.eye(d)]
    r=np.r_[rhs,np.full(2*d,PRIOR_BOUND)]
    norm=np.linalg.norm(a,axis=1); zero=norm<1e-14
    if np.any(r[zero]<-1e-12): return None,{"reason":"constant_infeasible"}
    a,r=a[~zero],r[~zero]; norm=norm[~zero]; a=a/norm[:,None]; r=r/norm
    # A bounded LP detects infeasibility; also records coordinate ranges to
    # condition narrow noisily observed directions before calling Qhull.
    lows=[]; highs=[]
    for j in range(d):
        e=np.eye(d)[j]
        ll=linprog(e,A_ub=a,b_ub=r,bounds=[(None,None)]*d,options={"primal_feasibility_tolerance":1e-9})
        hh=linprog(-e,A_ub=a,b_ub=r,bounds=[(None,None)]*d,options={"primal_feasibility_tolerance":1e-9})
        if not ll.success or not hh.success: return None,{"reason":"infeasible_or_lp_error","statuses":[int(ll.status),int(hh.status)]}
        lows.append(ll.x[j]); highs.append(hh.x[j])
    center=(np.array(lows)+highs)/2; scale=(np.array(highs)-lows)/2
    if np.any(scale<=1e-12): return None,{"reason":"numerically_zero_width"}
    ag=a*scale[None,:]; rr=r-a@center; norms=np.linalg.norm(ag,axis=1)
    cc=linprog(np.r_[np.zeros(d),-1.],A_ub=np.c_[ag,norms],b_ub=rr,
        bounds=[(None,None)]*d+[(0,None)],options={"primal_feasibility_tolerance":1e-9})
    if not cc.success or cc.x[-1]<=1e-10: return None,{"reason":"no_strict_numerical_interior"}
    interior=cc.x[:-1]
    try:
        vertices=HalfspaceIntersection(np.c_[ag,-rr],interior).intersections
        hull=ConvexHull(vertices)
    except QhullError:
        return None,{"reason":"qhull_error"}
    facets=vertices[hull.simplices]
    volume=np.abs(np.linalg.det(facets-interior))/math.factorial(d)
    total=float(volume.sum())
    assert np.isclose(total,hull.volume,rtol=1e-6,atol=1e-20),(total,hull.volume)
    if total<=0: return None,{"reason":"zero_volume"}
    info={"center":center,"scale":scale,"interior":interior,"facets":facets,
          "simplex_probs":volume/total,"volume":total*float(np.prod(scale)),"a":a,"rhs":r}
    return info,{"reason":"positive_volume","volume":info["volume"],"vertices":len(vertices),"facets":len(facets)}


def sample(poly,count,rng):
    d=len(poly["center"]); choices=rng.choice(len(poly["facets"]),count,p=poly["simplex_probs"])
    weights=rng.dirichlet(np.ones(d+1),size=count)
    y=np.sum(weights[:,:d,None]*poly["facets"][choices],axis=1)+weights[:,-1,None]*poly["interior"]
    b=poly["center"]+poly["scale"]*y
    assert np.max(b@poly["a"].T-poly["rhs"])<1e-7
    return b


def make_predict(bank):
    def predict(q):
        answer=np.empty(len(q))
        for first in range(0,len(q),256):
            h=np.broadcast_to(q[first:first+256],(len(bank),min(256,len(q)-first)))
            for j in range(bank.shape[1]): h=base.g(h+bank[:,j,None])
            answer[first:first+256]=h.mean(axis=0)
        return answer
    return predict


def fit(x,v,anchor,cfg):
    begin=time.perf_counter()
    if cfg["generator"]=="alm":
        bank,meta=local_discover(x,v,anchor,sweeps=120,restarts=64,dual_rate=cfg.get("dual_rate",.5))
    else:
        bank,meta=bp_discover(x,v,anchor,restarts=64,solver=cfg["generator"],band=True)
    discovery=time.perf_counter()-begin
    polys=[]; trace=[]; point=None; begin=time.perf_counter()
    for start in bank:
        _,_,g,rhs=base.branch_polytope(x,v,start)
        poly,note=polytope(g,rhs); trace.append(note)
        if poly is None: continue
        polys.append(poly)
        if point is None:
            initial=poly["center"]+poly["scale"]*poly["interior"]
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method="SLSQP",
                constraints={"type":"ineq","fun":lambda b:poly["rhs"]-poly["a"]@b,"jac":lambda b:-poly["a"]},
                bounds=[(-PRIOR_BOUND,PRIOR_BOUND)]*len(anchor),options={"maxiter":300,"ftol":1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    geometry=time.perf_counter()-begin
    if point is None: point=np.clip(bank[0],-PRIOR_BOUND,PRIOR_BOUND)
    predict={"point":lambda q:base.forward(q,point)}; saved={"point":point[None]}
    volume=np.array([p["volume"] for p in polys]); begin=time.perf_counter()
    for mode in ["equal","volume"]:
        if not polys: samples=point[None]
        else:
            rng=np.random.default_rng(6173); prob=volume/volume.sum() if mode=="volume" else np.full(len(polys),1/len(polys))
            counts=rng.multinomial(cfg.get("posterior_samples",512),prob)
            samples=np.concatenate([sample(p,int(n),rng) for p,n in zip(polys,counts) if n])
            error=np.max(np.abs(np.stack([base.forward(x,b) for b in samples])-v))
            assert error<=base.EPS+1e-7,error
        predict[mode]=make_predict(samples); saved[mode]=samples
    sampling=time.perf_counter()-begin
    return predict,point,{**meta,"discovery_seconds":discovery,"geometry_seconds":geometry,"sampling_seconds":sampling,
        "positive_volume_regions":len(polys),"discovered_prior_mass":float(volume.sum()/(2*PRIOR_BOUND)**len(anchor)),
        "region_effective_count":float(volume.sum()**2/np.sum(volume**2)) if len(volume) else 0,
        "retained_predictor_bytes":{k:b.nbytes for k,b in saved.items()},"geometry_trace":trace,
        "posterior_claim":"conditional on discovered region union only; missing mass unknown"}


def prior_moments(x,v,depth,features=4096):
    bank=np.random.default_rng(731).uniform(-PRIOR_BOUND,PRIOR_BOUND,(features,depth))
    def values(z):
        h=np.broadcast_to(z,(features,len(z)))
        for j in range(depth): h=base.g(h+bank[:,j,None])
        return h
    phi=values(x); mu=phi.mean(0); centered=phi-mu
    covariance=centered.T@centered/features+np.eye(len(x))*base.EPS**2/3
    alpha=np.linalg.solve(covariance,v-mu); weights=(1+centered@alpha)/features
    def predict(q):
        answer=np.empty(len(q))
        for first in range(0,len(q),128): answer[first:first+128]=weights@values(q[first:first+128])
        return answer
    return predict,{"persistent_state_bytes":bank.nbytes+weights.nbytes,"noise_variance":base.EPS**2/3}


def verify():
    rng=np.random.default_rng(88913); errors=[]
    for d in [2,3,4]:
        # No observed constraints: the entire true prior box must be recovered.
        p,m=polytope(np.empty((0,d)),np.empty(0)); assert p is not None
        error=abs(p["volume"]/(2*PRIOR_BOUND)**d-1); assert error<1e-10; errors.append(error)
        draws=sample(p,10000,rng)
        assert np.max(np.abs(draws.mean(0)))<.003
        assert np.max(np.abs(draws.var(0)-PRIOR_BOUND**2/3))<.00015
    b=rng.uniform(-PRIOR_BOUND,PRIOR_BOUND,4); x=rng.uniform(0,1,8)
    v=base.forward(x,b)+rng.uniform(-base.EPS,base.EPS,len(x))
    _,_,g,rhs=base.branch_polytope(x,v,b); p,m=polytope(g,rhs); assert p is not None,m
    draws=sample(p,500,rng); maxerr=max(np.max(np.abs(base.forward(x,bb)-v)) for bb in draws)
    assert maxerr<=base.EPS+1e-8
    return {"passed":True,"box_relative_volume_errors":errors,"sample_support_max_error":float(maxerr)}
