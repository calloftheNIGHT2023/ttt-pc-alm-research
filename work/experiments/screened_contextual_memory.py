"""Shared screening/readout optimizations for every candidate generator."""
import time
import numpy as np
import streaming_branch_projection as base
import contextual_candidate_bank as candidates
import posterior_confirmation_pipeline as legacy
from run_local_screening import screen_bank,solve_bank
import variance_controlled_readout as readout


def fit(x,v,anchor,cfg):
    start=time.perf_counter()
    if cfg["kind"]=="random":
        bank,pm=legacy.discover(x,v,anchor,cfg)
    else:
        starts,pm=candidates.proposals(x,v,anchor,cfg["features"],cfg["restarts"])
        if cfg["generator"]=="direct": bank=starts
        elif cfg["generator"]=="alm":
            refined,rm=candidates.refine_local(starts,x,v,anchor,cfg["sweeps"],cfg.get("dual_rate",.5)); bank=np.vstack([starts,refined]); pm.update(rm)
        else:
            refined,rm=candidates.refine_bp(starts,x,v,anchor,cfg["generator"]); bank=np.vstack([starts,refined]); pm.update(rm)
        bank=candidates.deduplicate(bank,x,v,anchor)
    discovery_time=time.perf_counter()-start
    mask,proofs,sm,screen_time=screen_bank(x,v,bank,"contract5"); assert not proofs
    polys,point,ids,notes,geometry_time=solve_bank(x,v,anchor,bank,mask)
    predict,rm=readout.build(polys,point,"adaptive_iid",1e-6)
    return predict,point,{**pm,**rm,"discovery_seconds":discovery_time,"screen_seconds":screen_time,"geometry_seconds":geometry_time,
        "candidates":len(bank),"screened":int(mask.sum()),"positive_volume_regions":len(ids),"geometry_trace":notes,
        "persistent_state_bytes":point.nbytes+rm["readout_state_bytes"],"common_observed_context_bytes":x.nbytes+v.nbytes,
        "anchor_output":point.tolist(),"discovered_parameter_volume":sum(p["volume"] for p in polys)}


def verify():
    import readout_budget_pipeline as reference
    rng=np.random.default_rng(45902); x=rng.uniform(0,1,8); v=base.forward(x,rng.uniform(-.12,.12,4)); anchor=np.zeros(4)
    cfg={"kind":"contextual","features":4096,"restarts":64,"generator":"alm","sweeps":20}
    predict,point,meta=fit(x,v,anchor,cfg); polys,b,_=reference.construct(x,v,anchor,cfg); expected,_=readout.build(polys,b,"adaptive_iid",1e-6)
    q=np.linspace(0,1,217); assert np.array_equal(point,b) and np.array_equal(predict(q),expected(q))
    return {"passed":True,"screened_pipeline_matches_unscreened":True}
