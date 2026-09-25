"""363 current live trajectory; no component state as candidate input."""
from types import SimpleNamespace
import cross_region_online_bridge_v1 as bridge
import frontier_reallocation_v1 as model


def fit(x,v,q,seed,*,channel,strategy):
    original=bridge.local;extra={}
    def solve(xx,vv,regs,credit,*,propagate,steps):
        assert propagate is False and steps==128
        arrays,meta=model.solve(xx,vv,regs,credit,strategy=strategy,steps=steps,geometry_budget=8)
        extra.update({'reallocation_'+k:a for k,a in arrays.items()})
        return arrays,meta
    try:
        bridge.local=SimpleNamespace(solve=solve)
        arrays,meta=bridge.fit(x,v,q,seed,channel=channel,propagate=False,steps=128,geometry_budget=8)
    finally:bridge.local=original
    arrays.update(extra);meta.update(selector='budget_frontier_reallocation',reallocation_strategy=strategy,
        archived_candidate_state_accessed=False,resource_scope='Same 128M response ceiling, not matched wall time or complete state')
    return arrays,meta
