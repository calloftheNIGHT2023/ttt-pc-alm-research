"""Recheck every saved certificate against exact arithmetic and independent LP."""
from __future__ import annotations
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
from certified_branch_solver import exact_certificate
from audit_dual_branch_cuts import lp_feasible


def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True); args=p.parse_args()
    rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
    checked=[]
    for r in rows:
        rng=np.random.default_rng(r["seed"])
        truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)[:r["n_context"]]; v=base.forward(x,truth)
        for i,t in enumerate(r.get("regional_trace",[])):
            c=t["infeasibility_certificate"]
            if c is None: continue
            regs=np.array(t["regional_pattern"])
            up,ua=np.array(c["up"]),np.array(c["ua"])
            exact=exact_certificate(x,v,regs,up,ua)
            assert exact["numerator"]==c["numerator"] and exact["denominator"]==c["denominator"]
            # Compensate any rounding in the floating epsilon+TOL used by the
            # numerical acceptance check, as opposed to the exact sum of inputs.
            delta=max(F(0),F(float(base.EPS+base.TOL))-F(base.EPS)-F(base.TOL))
            corrected=F(int(c["numerator"]),int(c["denominator"]))-delta*sum((abs(F(float(a))) for a in ua[-1]),F(0))
            assert corrected>0
            assert not lp_feasible(x,v,regs)
            checked.append({"method":r["method"],"seed":r["seed"],"n_context":r["n_context"],
                "region_index":i,"corrected_margin":float(corrected),"dual_proposal":c["dual_proposal"]})
    result={"passed":True,"certificates_rechecked":len(checked),"rows":checked,
        "audit_only":"Independent LP is never used to produce candidate parameters."}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"passed":True,"certificates_rechecked":len(checked)}))


if __name__=="__main__": main()
