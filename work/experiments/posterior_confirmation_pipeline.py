"""Frozen posterior pipeline: configurable common discovery domains and ablations."""
from contextlib import contextmanager
import time
import numpy as np
from scipy.optimize import minimize
import streaming_branch_projection as base
import region_posterior_memory as posterior
import region_posterior_controls as controls
from hybrid_discovery_bank import discover as local_discover
from matched_discovery_baselines import discover as bp_discover


@contextmanager
def discovery_box(bound):
    original=base.BOUND
    try:
        base.BOUND=bound
        yield
    finally: base.BOUND=original


def discover(x,v,anchor,cfg):
    original=base.BOUND
    with discovery_box(cfg.get("discovery_bound",.15)):
        if cfg["generator"] in ["alm","pc_gradient"]:
            bank,meta=local_discover(x,v,anchor,sweeps=cfg.get("sweeps",120),restarts=cfg.get("restarts",64),
                dual_rate=cfg.get("dual_rate",.5),gradient_blocks=cfg["generator"]=="pc_gradient")
        elif cfg["generator"] in ["trf","lbfgs"]:
            bank,meta=bp_discover(x,v,anchor,restarts=cfg.get("restarts",64),solver=cfg["generator"],band=True)
        else:
            bank,meta=controls.discover(x,v,anchor,cfg)
    assert base.BOUND==original
    return bank,meta


def fit(x,v,anchor,cfg):
    begin=time.perf_counter(); bank,meta=discover(x,v,anchor,cfg); discovery=time.perf_counter()-begin
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
    if point is None: point=np.clip(bank[0],-posterior.PRIOR_BOUND,posterior.PRIOR_BOUND)
    if polys:
        volume=np.array([p["volume"] for p in polys]); probs=volume/volume.sum()
        rng=np.random.default_rng(6173); counts=rng.multinomial(cfg.get("posterior_samples",512),probs)
        samples=np.concatenate([posterior.sample(p,int(n),rng) for p,n in zip(polys,counts) if n])
        err=max(np.max(np.abs(base.forward(x,b)-v)) for b in samples); assert err<=base.EPS+1e-7
    else: samples=point[None]; volume=np.zeros(0)
    sampling=time.perf_counter()-begin
    return posterior.make_predict(samples),point,{**meta,"discovery_seconds":discovery,"geometry_seconds":geometry,"sampling_seconds":sampling,
        "positive_volume_regions":len(polys),"discovered_prior_mass":float(volume.sum()/(2*posterior.PRIOR_BOUND)**len(anchor)),
        "persistent_state_bytes":samples.nbytes+point.nbytes,"common_observed_context_bytes":x.nbytes+v.nbytes,
        "geometry_trace":trace,"query_read_samples":len(samples),"anchor_output":point.tolist(),
        "posterior_claim":"conditional on discovered regions only; missing mass unknown"}


def verify():
    original=base.BOUND
    rng=np.random.default_rng(81172); x=rng.uniform(0,1,4); v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,4)
    for bound in [.12,.15]:
        for gen in ["alm","trf","lbfgs","pc_gradient"]:
            bank,_=discover(x,v,np.zeros(4),{"generator":gen,"restarts":4,"sweeps":12,"discovery_bound":bound})
            assert np.max(np.abs(bank))<=bound+1e-12
            assert base.BOUND==original
    # Exactly match the original candidate on an audit-only, fixed artificial task.
    c={"generator":"alm","restarts":64,"discovery_bound":.15,"posterior_samples":512}
    pred,b,_=fit(x,v,np.zeros(4),c)
    old,bb,_=posterior.fit(x,v,np.zeros(4),c)
    q=np.linspace(0,1,99)
    assert np.allclose(b,bb,atol=1e-13)
    assert np.allclose(pred(q),old["volume"](q),atol=1e-13)
    return {"passed":True,"original_candidate_equivalence":True,"configurable_common_box":True,"geometry":posterior.verify()}
