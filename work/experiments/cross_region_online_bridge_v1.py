"""358 live parent trajectory and full necessary-language credit screening."""
import time
import numpy as np
import strong_pool_credit_bridge_v1 as bridge
import support_language_chain_v1 as language
from run_cross_region_credit_v1 import pool
import cross_region_credit_v1 as local


def fit(x,v,q,seed,*,channel,propagate=True,steps=128,screen=True,geometry_budget=8):
    begin=time.perf_counter();visited=set();saved_modes=bridge.modes;saved_propose=language.propose
    calls=0;extra={}
    def collect(xx,bank):
        result=saved_modes(xx,bank);visited.update(result);return result
    def propose(xx,vv,credit,pattern,k=8):
        nonlocal calls
        calls+=1;tick=time.perf_counter();regs,poolmeta=pool(xx,vv,pattern,sorted(visited))
        if screen:
            arrays,meta=local.solve(xx,vv,regs,credit,propagate=propagate,steps=steps)
        else:
            arrays=dict(first_step=np.zeros(len(regs),np.int32),proof_credit=np.zeros(regs.shape))
            meta=dict(proofs=[],traces=[],directions=[],total_seconds=0.,local_response_pairs=0,
                      transfer_response_pairs=0,total_response_pairs=0,exact_calls=0)
        ids=np.flatnonzero(arrays['first_step']==0)
        if geometry_budget is not None:ids=ids[:geometry_budget]
        proposals=[dict(mode=regs[i].tobytes().hex(),hamming=int((regs[i]!=pattern).sum()),rank=j) for j,i in enumerate(ids)]
        extra.update(search_regions=regs,search_first_step=arrays['first_step'],search_proof_credit=arrays['proof_credit'])
        return dict(proposals=proposals,pool=poolmeta,screening=meta,
                    meta=dict(total_seconds=time.perf_counter()-tick,forbidden_modes=len(visited),
                              geometry_budget=geometry_budget,screen=screen,propagate=propagate,steps=steps))
    try:
        bridge.modes=collect;language.propose=propose
        arrays,meta=bridge.fit(x,v,q,seed,channel=channel,trace_hash=True)
    finally:
        bridge.modes=saved_modes;language.propose=saved_propose
    assert calls==1 and sorted(visited)==meta['visited_modes']
    arrays.update(extra);meta.update(charged_complete_seconds=time.perf_counter()-begin,
        visited_source='Own current live trajectory, not old files or another solver',
        selector='necessary_language_c20_cross_region_credit',geometry_budget=geometry_budget,
        no_archived_solver_state_input=True)
    return arrays,meta
