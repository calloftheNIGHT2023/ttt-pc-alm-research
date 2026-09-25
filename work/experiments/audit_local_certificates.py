"""Replay already-observed regions; independent LP is audit-only."""
from __future__ import annotations
import argparse
import hashlib
import json
import importlib
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import streaming_branch_projection as base
import certified_branch_solver as solver


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--solver",default="certified_branch_solver")
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    solver=importlib.import_module(args.solver)
    source=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
    source=[r for r in source if r["method"]=="restore_120x64"]
    rows=[]
    for i,r in enumerate(source):
        rng=np.random.default_rng(r["seed"])
        truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)[:r["n_context"]]; v=base.forward(x,truth)
        b=np.array(r["b"]); anchor=np.array(r["anchor"])
        _,_,mat,rhs=base.branch_polytope(x,v,b)
        lp=linprog(np.zeros(4),A_ub=mat,b_ub=rhs,bounds=[(-base.BOUND,base.BOUND)]*4,method="highs",
                   options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        begin=time.perf_counter(); out,meta=solver.solve(x,v,anchor,b); sec=time.perf_counter()-begin
        certified=meta["stop_reason"]=="certified_infeasible"
        if lp.success: assert not certified,(r,meta)
        if certified: assert lp.status==2,(r,lp.message,meta)
        # Exact certificate is independently re-evaluable from saved rational value.
        cert=meta["infeasibility_certificate"]
        if cert is not None: assert int(cert["numerator"])>0 and int(cert["denominator"])>0
        if cert is not None and "up" in cert:
            check=solver.exact_certificate(x,v,np.array(meta["regional_pattern"]),np.array(cert["up"]),np.array(cert["ua"]))
            assert check["numerator"]==cert["numerator"] and check["denominator"]==cert["denominator"]
        rows.append({"seed":r["seed"],"n_context":r["n_context"],"lp_status":int(lp.status),
                     "lp_feasible":bool(lp.success),"seconds":sec,**meta})
        if (i+1)%16==0: print(json.dumps({"completed":i+1,"total":len(source)}),flush=True)
    infeasible=[r for r in rows if r["lp_status"]==2]
    summary={"phase":"development_replay_not_new_confirmation","regions":len(rows),
             "lp_infeasible":len(infeasible),"certified":sum(r["stop_reason"]=="certified_infeasible" for r in rows),
             "false_positive_on_lp_feasible":0,
             "infeasible_stop_steps":[r["polish_steps"] for r in infeasible],
             "infeasible_mean_stop_steps":float(np.mean([r["polish_steps"] for r in infeasible])),
             "solver_sha256":hashlib.sha256(Path(solver.__file__).read_bytes()).hexdigest()}
    (args.out/"regions.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()
