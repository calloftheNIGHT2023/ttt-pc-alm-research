"""Check exact equivalence of discovery variants and the no-global-BP boundary."""
import json
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
from run_certified_search import global_bank
from hybrid_discovery_bank import discover
import hybrid_branch_memory as hybrid

rows=[]
for seed in range(5601000,5601004):
    rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4)
    x=rng.uniform(0,1,24); v=base.forward(x,truth); anchor=np.zeros(4)
    for gradient in [False,True]:
        for dual in [0.,.5]:
            sweeps=120 if not gradient else 480
            old,_,_=base.fit_local(x,v,anchor,warm=False,rho=float("inf"),sweeps=sweeps,
                                   restarts=64,dual_rate=dual,gradient_blocks=gradient)
            bank,_=discover(x,v,anchor,sweeps=sweeps,restarts=64,dual_rate=dual,gradient_blocks=gradient)
            diff=float(np.max(np.abs(old-bank[0])))
            assert diff<1e-12,(seed,gradient,dual,diff)
            rows.append({"seed":seed,"gradient_blocks":gradient,"dual_rate":dual,"winner_difference":diff})
    old,_=global_bank(x,v,anchor,{})
    def forbidden(*args,**kwargs): raise AssertionError("Nonlinear global Jacobian invoked inside hybrid")
    original=base.forward_jacobian; base.forward_jacobian=forbidden
    try: new,_=hybrid.fit(x,v,anchor,{})
    finally: base.forward_jacobian=original
    assert np.max(np.abs(old-new))<1e-12
out=Path("outputs/ttt-pc-alm-research/results/hybrid_branch_memory/discovery_audit.json")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"passed":True,"cases":rows,"hybrid_vs_previous_cases":4,"no_global_nonlinear_jacobian_guard":True},indent=2),encoding="utf-8")
print(json.dumps({"passed":True,"discovery_cases":len(rows),"hybrid_equivalence_cases":4}))
