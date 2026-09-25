"""Batched BP and previous ALM/serial BP receive identical downstream code."""
import time
import numpy as np
import contextual_candidate_bank as candidates
import finite_bp_budget as serial
import batched_bp_discovery as batched
from run_local_screening import screen_bank,solve_bank
import variance_controlled_readout as readout


def fit(x,v,anchor,cfg):
    if cfg["generator"] not in ["gauss_newton","adam"]:return serial.fit(x,v,anchor,cfg)
    begin=time.perf_counter(); starts,pm=candidates.proposals(x,v,anchor,cfg["features"],cfg["restarts"])
    refined,meta=batched.refine(starts,x,v,anchor,solver=cfg["generator"],steps=cfg["steps"],lr=cfg.get("lr",.003))
    bank=candidates.deduplicate(np.vstack([starts,refined]),x,v,anchor); discovery=time.perf_counter()-begin
    mask,proofs,_,screen_time=screen_bank(x,v,bank,"contract5"); assert not proofs
    polys,point,ids,notes,geometry_time=solve_bank(x,v,anchor,bank,mask); predict,rm=readout.build(polys,point,"adaptive_iid",1e-6)
    return predict,point,{**pm,**meta,**rm,"discovery_seconds":discovery,"screen_seconds":screen_time,"geometry_seconds":geometry_time,
        "candidates":len(bank),"screened":int(mask.sum()),"positive_volume_regions":len(ids),"geometry_trace":notes,
        "persistent_state_bytes":point.nbytes+rm["readout_state_bytes"],"anchor_output":point.tolist()}


def verify():return {"batched":batched.verify(),"serial":serial.verify()}
