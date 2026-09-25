"""Certificate-derived local branch proposals with exact reusable cuts."""
from __future__ import annotations
import numpy as np
import streaming_branch_projection as base
from certified_branch_search import discover
import certified_branch_solver_v3 as regional
from dual_branch_cut import BranchCut


def fit(x,v,anchor,config):
    if np.max(np.abs(base.forward(x,anchor)-v))<=base.EPS:
        return anchor.copy(),{"skipped":True,"regional_steps":0,"regions_attempted":0}
    bank,meta=discover(x,v,anchor,sweeps=config.get("sweeps",120),restarts=config.get("restarts",64))
    maxregions=config.get("max_regions",8); budget=config.get("budget",8000)
    queue=[(base.pattern(x,b),b.copy(),"discovery") for b in bank[:maxregions]]
    seen=set(); cuts=[]; traces=[]; pruned=0; generated=0; remaining=budget
    best=bank[0].copy(); err0,mv0=base.score(best[None],x,v,anchor)
    while queue and remaining>=20 and len(traces)<maxregions:
        regs,start,origin=queue.pop(0); key=regs.tobytes()
        if key in seen: continue
        seen.add(key)
        if any(cut.excludes(regs) for cut in cuts): pruned+=1; continue
        b,info=regional.solve(x,v,anchor,start,steps=remaining,pattern_override=regs)
        remaining-=info["polish_steps"]
        er,mv=base.score(b[None],x,v,anchor)
        if base.better(er,mv,err0,mv0)[0]: best,err0,mv0=b.copy(),er,mv
        traces.append({"origin":origin,**info})
        if info["stop_reason"]!="certified_infeasible": break
        cut=BranchCut(regs,info["infeasibility_certificate"]); cuts.append(cut)
        proposals,cutmeta=cut.proposals(); traces[-1]["cut_proposal_metadata"]=cutmeta
        items=[(p,b.copy(),"dual_cut") for p in proposals if p.tobytes() not in seen]
        generated+=len(items)
        if config.get("priority","first")=="first": queue=items+queue
        else: queue+=items
    return best,{**meta,"regional_steps":budget-remaining,"regions_attempted":len(traces),
        "regions_certified_infeasible":len(cuts),"cut_pruned_without_iteration":pruned,
        "cut_generated_proposals":generated,"regional_trace":traces}
