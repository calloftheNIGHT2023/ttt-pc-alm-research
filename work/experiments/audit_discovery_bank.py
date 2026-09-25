from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import certified_branch_search as search

rows=[]
for seed in range(5600000,5600008):
    rng=np.random.default_rng(seed)
    truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
    anchor=np.zeros(4)
    for n in [4,8,16,24]:
        old,_,_=base.fit_local(x[:n],v[:n],anchor,warm=False,sweeps=120,restarts=64,rho=float("inf"))
        bank,meta=search.discover(x[:n],v[:n],anchor)
        error=float(np.max(np.abs(old-bank[0])))
        assert error<1e-12,(seed,n,error)
        rows.append({"seed":seed,"n_context":n,"winner_difference":error,**meta})
        anchor=old
out=Path("outputs/ttt-pc-alm-research/results/certified_branch_search/discovery_audit.json")
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps({"passed":True,"cases":rows},indent=2),encoding="utf-8")
print(json.dumps({"passed":True,"cases":len(rows),"max_difference":max(r["winner_difference"] for r in rows)}))
