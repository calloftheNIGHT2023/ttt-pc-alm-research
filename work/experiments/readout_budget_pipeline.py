"""Same discovered posterior for all readout controls; no query-driven choice."""
import time
import numpy as np
from scipy.optimize import minimize
import streaming_branch_projection as base
import contextual_candidate_bank as candidate
import region_posterior_memory as geometry
import variance_controlled_readout as readout


def construct(x,v,anchor,cfg):
    begin=time.perf_counter(); starts,meta=candidate.proposals(x,v,anchor,cfg["features"],cfg["restarts"])
    if cfg["generator"]=="direct": bank=starts
    elif cfg["generator"]=="alm":
        refined,more=candidate.refine_local(starts,x,v,anchor,cfg["sweeps"],cfg.get("dual_rate",.5)); bank=np.vstack([starts,refined]); meta.update(more)
    else:
        refined,more=candidate.refine_bp(starts,x,v,anchor,cfg["generator"]); bank=np.vstack([starts,refined]); meta.update(more)
    bank=candidate.deduplicate(bank,x,v,anchor); polys=[]; point=None; trace=[]
    for start in bank:
        _,_,g,rhs=base.branch_polytope(x,v,start); poly,note=geometry.polytope(g,rhs); trace.append(note)
        if poly is None: continue
        polys.append(poly)
        if point is None:
            initial=poly["center"]+poly["scale"]*poly["interior"]
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method="SLSQP",
                constraints={"type":"ineq","fun":lambda b:poly["rhs"]-poly["a"]@b,"jac":lambda b:-poly["a"]},
                bounds=[(-.12,.12)]*len(anchor),options={"maxiter":300,"ftol":1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    if point is None: point=np.clip(bank[0],-.12,.12)
    return polys,point,{**meta,"common_discovery_and_geometry_seconds":time.perf_counter()-begin,"geometry_trace":trace,
        "positive_volume_regions":len(polys),"discovered_parameter_volume":sum(p["volume"] for p in polys)}


def verify():
    import contextual_posterior_memory as previous
    rng=np.random.default_rng(65189); x=rng.uniform(0,1,8); v=base.forward(x,rng.uniform(-.12,.12,4))+rng.uniform(-base.EPS,base.EPS,8)
    cfg={"features":4096,"restarts":64,"generator":"alm","sweeps":20}
    polys,point,_=construct(x,v,np.zeros(4),cfg); predict,_=readout.build(polys,point,"iid512")
    reference,bb,_=previous.fit(x,v,np.zeros(4),cfg); q=np.linspace(0,1,99)
    assert np.allclose(point,bb,atol=1e-13) and np.allclose(predict(q),reference(q),atol=1e-13)
    for mode in ["iid512","strat512","adaptive","adaptive_iid","iid64","strat64"]:
        predict,meta=readout.build(polys,point,mode)
        assert np.max(np.abs(predict(x)-v))<=base.EPS+1e-7
    return {"passed":True,"old_iid512_readout_equivalent":True,"all_readout_modes_support_feasible":True,"moments_and_allocation":readout.verify()}
