"""Initial-region preservation with support-conditioned discovery proposals."""
import time
import numpy as np
from scipy.optimize import minimize
import streaming_branch_projection as base
import contextual_candidate_bank as candidate
import region_posterior_memory as geometry


def fit(x,v,anchor,cfg):
    begin=time.perf_counter(); starts,meta=candidate.proposals(x,v,anchor,cfg["features"],cfg["restarts"])
    proposal_time=time.perf_counter()-begin
    seed_patterns={base.pattern(x,b).tobytes() for b in starts}
    begin=time.perf_counter()
    if cfg["generator"]=="direct": bank=starts; refinement={}
    elif cfg["generator"]=="alm":
        refined,refinement=candidate.refine_local(starts,x,v,anchor,sweeps=cfg["sweeps"],dual_rate=cfg.get("dual_rate",.5))
        bank=np.vstack([starts,refined])
    else:
        refined,refinement=candidate.refine_bp(starts,x,v,anchor,cfg["generator"]); bank=np.vstack([starts,refined])
    bank=candidate.deduplicate(bank,x,v,anchor); refinement_time=time.perf_counter()-begin
    final_patterns={base.pattern(x,b).tobytes() for b in bank}
    assert seed_patterns<=final_patterns
    begin=time.perf_counter(); polys=[]; trace=[]; point=None; seed_volume=extra_volume=0.
    for start in bank:
        seed=base.pattern(x,start).tobytes() in seed_patterns
        _,_,g,rhs=base.branch_polytope(x,v,start); poly,note=geometry.polytope(g,rhs)
        trace.append({**note,"initial_seed_region":seed})
        if poly is None: continue
        polys.append(poly)
        if seed: seed_volume+=poly["volume"]
        else: extra_volume+=poly["volume"]
        if point is None:
            initial=poly["center"]+poly["scale"]*poly["interior"]
            qp=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),initial,jac=True,method="SLSQP",
                constraints={"type":"ineq","fun":lambda b:poly["rhs"]-poly["a"]@b,"jac":lambda b:-poly["a"]},
                bounds=[(-.12,.12)]*len(anchor),options={"maxiter":300,"ftol":1e-12})
            point=qp.x if np.max(np.abs(base.forward(x,qp.x)-v))<=base.EPS+base.TOL else initial
    geometry_time=time.perf_counter()-begin; begin=time.perf_counter()
    if point is None: point=np.clip(bank[0],-.12,.12)
    if polys:
        volumes=np.array([p["volume"] for p in polys]); rng=np.random.default_rng(6173)
        counts=rng.multinomial(512,volumes/volumes.sum())
        samples=np.concatenate([geometry.sample(p,int(n),rng) for p,n in zip(polys,counts) if n])
        error=max(np.max(np.abs(base.forward(x,b)-v)) for b in samples); assert error<=base.EPS+1e-7
    else: samples=point[None]
    sampling_time=time.perf_counter()-begin
    return geometry.make_predict(samples),point,{**meta,**refinement,"proposal_seconds":proposal_time,"refinement_seconds":refinement_time,
        "geometry_seconds":geometry_time,"sampling_seconds":sampling_time,"unique_seed_regions":len(seed_patterns),
        "unique_final_regions":len(final_patterns),"initial_region_preservation_verified":True,
        "positive_volume_regions":len(polys),"initial_feasible_parameter_volume":seed_volume,"added_feasible_parameter_volume":extra_volume,
        "persistent_state_bytes":samples.nbytes+point.nbytes,"common_context_bytes":x.nbytes+v.nbytes,
        "geometry_trace":trace,"query_read_samples":len(samples)}
