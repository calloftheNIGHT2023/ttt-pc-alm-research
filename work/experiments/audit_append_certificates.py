"""Audit zero-extension of all saved exact regional infeasibility proofs."""
import argparse,json
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
from certified_branch_solver import exact_certificate

p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True); p.add_argument("--out",type=Path,required=True); args=p.parse_args()
rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8")); checked=[]
for row in rows:
    rng=np.random.default_rng(row["seed"]); truth=rng.uniform(-.12,.12,4); oldx=rng.uniform(0,1,24)[:row["n_context"]]
    oldv=base.forward(oldx,truth)
    for i,tr in enumerate(row.get("regional_trace",[])):
        cert=tr.get("infeasibility_certificate")
        if cert is None: continue
        regs=np.array(tr["regional_pattern"]); up=np.array(cert["up"]); ua=np.array(cert["ua"])
        # Any extension of the old pattern, even a currently unrealizable one,
        # must be rejected. Random new labels test independence of their values.
        nx=rng.uniform(0,1,7); nv=rng.uniform(0,1,7); nr=rng.integers(0,4,size=(4,7))
        extended=exact_certificate(np.r_[oldx,nx],np.r_[oldv,nv],np.c_[regs,nr],np.c_[up,np.zeros((4,7))],np.c_[ua,np.zeros((4,7))])
        assert extended["positive"]
        assert extended["numerator"]==cert["numerator"] and extended["denominator"]==cert["denominator"]
        checked.append({"seed":row["seed"],"method":row["method"],"n_old":len(oldx),"region":i,"equal_exact_fraction":True})
result={"passed":True,"certificates":len(checked),"new_contexts_per_check":7,"checks":checked}
args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in result.items() if k!="checks"}))
