import json
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import hybrid_discovery_bank as old
import weighted_discovery_bank as weighted

rng=np.random.default_rng(562718); gaps=[]
for ratio in [.25,1.,4.]:
    for _ in range(12):
        a,t=rng.normal(.5,.5,2); b=rng.uniform(-.15,.15); previous=rng.uniform(0,1); tau=.01
        lo=np.maximum(0,np.array([-np.inf,0,.5,1])-b)
        hi=np.minimum(1,np.array([0,.5,1,np.inf])-b)
        slope=base.SLOPES; offset=slope*b+base.INTERCEPTS
        cand=np.clip((a+ratio*slope*(t-offset)+tau*previous)/(1+ratio*slope**2+tau),lo,hi)
        energy=lambda h:(h-a)**2+ratio*(base.g(h+b)-t)**2+tau*(h-previous)**2
        costs=np.where(lo<=hi,energy(cand),np.inf); value=float(costs.min())
        gridvalue=float(energy(np.linspace(0,1,50001)).min())
        assert value-gridvalue<1e-10
        gaps.append(value-gridvalue)
for seed in range(5601000,5601004):
    rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
    b0,_=old.discover(x,v,np.zeros(4)); b1,_=weighted.discover(x,v,np.zeros(4),layer_ratio=1.)
    assert np.array_equal(b0,b1)
out=Path("outputs/ttt-pc-alm-research/results/weighted_hybrid_memory/algebra_audit.json")
out.parent.mkdir(parents=True,exist_ok=True)
result={"passed":True,"scalar_cases":len(gaps),"max_energy_gap_to_grid":max(gaps),"ratio_one_exact_bank_equivalence_cases":4}
out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result))
