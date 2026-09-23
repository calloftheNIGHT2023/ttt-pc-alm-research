"""Common true-prior box, larger BP banks, raw region and task-library controls."""
import time
from contextlib import contextmanager
import numpy as np
from scipy.optimize import minimize
import streaming_branch_projection as base
import region_posterior_memory as posterior
from hybrid_discovery_bank import discover as local_discover
from matched_discovery_baselines import discover as bp_discover


@contextmanager
def prior_box():
    # Versioned experiment adapter only. Frozen source is not rewritten; restore
    # even on errors. This runner is single-process / single-threaded at fit level.
    original=base.BOUND
    try:
        base.BOUND=posterior.PRIOR_BOUND
        yield
    finally: base.BOUND=original


def discover(x,v,anchor,cfg):
    with prior_box():
        if cfg["generator"]=="alm":
            return local_discover(x,v,anchor,sweeps=120,restarts=cfg.get("restarts",64),dual_rate=cfg.get("dual_rate",.5))
        if cfg["generator"] in ["trf","lbfgs"]:
            return bp_discover(x,v,anchor,restarts=cfg.get("restarts",64),solver=cfg["generator"],band=True)
        count=cfg["restarts"]
        bank=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(count-1,len(anchor)))])
        errors,moves=base.score(bank,x,v,anchor)
        order=sorted(range(count),key=lambda i:(0,moves[i]) if errors[i]<=base.EPS+base.TOL else (1,errors[i]))
        seen=set(); unique=[]
        for i in order:
            key=base.pattern(x,bank[i]).tobytes()
            if key not in seen: unique.append(i); seen.add(key)
        return bank[unique],{"unique_regions":len(unique),"raw_prior_proposals":count}


def fit(x,v,anchor,cfg):
    begin=time.perf_counter(); bank,meta=discover(x,v,anchor,cfg); discovery=time.perf_counter()-begin
    assert base.BOUND==.15
    begin=time.perf_counter(); polys=[]; trace=[]; point=None
    for start in bank:
        _,_,g,rhs=base.branch_polytope(x,v,start)
        poly,note=posterior.polytope(g,rhs); trace.append(note)
        if poly is None: continue
        polys.append(poly)
        if point is None:
            initial=poly["center"]+poly["scale"]*poly["interior"]
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method="SLSQP",
                constraints={"type":"ineq","fun":lambda b:poly["rhs"]-poly["a"]@b,"jac":lambda b:-poly["a"]},
                bounds=[(-posterior.PRIOR_BOUND,posterior.PRIOR_BOUND)]*len(anchor),options={"maxiter":300,"ftol":1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    geometry=time.perf_counter()-begin; begin=time.perf_counter()
    if point is None: point=bank[0].copy()
    if polys:
        volume=np.array([p["volume"] for p in polys]); probs=volume/volume.sum()
        rng=np.random.default_rng(6173); counts=rng.multinomial(cfg.get("posterior_samples",512),probs)
        samples=np.concatenate([posterior.sample(p,int(n),rng) for p,n in zip(polys,counts) if n])
        err=max(np.max(np.abs(base.forward(x,b)-v)) for b in samples); assert err<=base.EPS+1e-7
    else: samples=point[None]; volume=np.zeros(0)
    sampling=time.perf_counter()-begin
    return posterior.make_predict(samples),point,{**meta,"discovery_seconds":discovery,"geometry_seconds":geometry,"sampling_seconds":sampling,
        "positive_volume_regions":len(polys),"discovered_prior_mass":float(volume.sum()/(2*posterior.PRIOR_BOUND)**len(anchor)),
        "persistent_state_bytes":samples.nbytes,"geometry_trace":trace,"query_read_samples":len(samples),
        "posterior_claim":"conditional on discovered regions only; missing mass unknown"}


def dictionary(x,v,depth,cfg):
    count=cfg["features"]; k=cfg.get("neighbors",16)
    bank=np.random.default_rng(731).uniform(-posterior.PRIOR_BOUND,posterior.PRIOR_BOUND,(count,depth))
    h=np.broadcast_to(x,(count,len(x)))
    for j in range(depth): h=base.g(h+bank[:,j,None])
    distance=np.mean((h-v)**2,axis=1)
    index=np.argsort(distance,kind="stable")[:k]
    retained=bank[index].copy()
    return posterior.make_predict(retained),{"family":"prior_task_dictionary_nearest_neighbors","prior_tasks":count,"neighbors":k,
        "persistent_state_bytes":retained.nbytes,"task_library_parameter_bytes":bank.nbytes,"support_feature_array_bytes":h.nbytes,
        "nearest_support_mse":float(distance[index[0]])}


def verify():
    original=base.BOUND
    try:
        with prior_box():
            assert base.BOUND==posterior.PRIOR_BOUND
            raise RuntimeError("test restore")
    except RuntimeError: pass
    assert base.BOUND==original
    rng=np.random.default_rng(59221); x=rng.uniform(0,1,4); v=base.forward(x,rng.uniform(-.12,.12,4))
    for gen in ["alm","trf","lbfgs","random"]:
        bank,_=discover(x,v,np.zeros(4),{"generator":gen,"restarts":4})
        assert np.max(np.abs(bank))<=posterior.PRIOR_BOUND+1e-12
        assert base.BOUND==original
    return {"passed":True,"common_prior_box_and_restore":True,"geometry":posterior.verify()}
