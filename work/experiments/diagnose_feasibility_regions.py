"""Post-hoc support-only diagnosis, NEVER a candidate update or a new test."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, minimize
import streaming_branch_projection as base


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    rows=json.loads((args.results/"episodes.json").read_text(encoding="utf-8"))
    diagnosed=[]
    for row in rows:
        if row["method"]!="restore_120x64" or row["support_feasible"]: continue
        rng=np.random.default_rng(row["seed"])
        truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)
        n=row["n_context"]; x=x[:n]; v=base.forward(x,truth)
        b=np.array(row["b"]); anchor=np.array(row["anchor"])
        _,_,mat,rhs=base.branch_polytope(x,v,b)
        feasible=linprog(np.zeros(4),A_ub=mat,b_ub=rhs,bounds=[(-base.BOUND,base.BOUND)]*4,method="highs",
                         options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        record={"seed":row["seed"],"n_context":n,"original_support_max_error":row["support_max_error"],
                "lp_status":int(feasible.status),"returned_branch_feasible":True if feasible.success else False if feasible.status==2 else None,
                "original_polish_steps":row.get("polish_steps"),"original_gap":row.get("fixed_branch_duality_gap")}
        if feasible.success:
            qp=minimize(lambda z:(.5*np.sum((z-anchor)**2),z-anchor),feasible.x,jac=True,method="SLSQP",
                        bounds=[(-base.BOUND,base.BOUND)]*4,
                        constraints={"type":"ineq","fun":lambda z:rhs-mat@z,"jac":lambda z:-mat},
                        options={"ftol":1e-12,"maxiter":500})
            err=float(np.max(np.abs(base.forward(x,qp.x)-v)))
            record.update({"diagnostic_qp_success":bool(qp.success),"diagnostic_support_max_error":err,
                           "diagnostic_feasible":bool(err<=base.EPS+base.TOL)})
            assert qp.success and err<=base.EPS+base.TOL,record
        diagnosed.append(record)
    summary=[{"n_context":n,"failed_writes":sum(r["n_context"]==n for r in diagnosed),
              "region_feasible":sum(r["n_context"]==n and r["returned_branch_feasible"] is True for r in diagnosed),
              "region_infeasible":sum(r["n_context"]==n and r["returned_branch_feasible"] is False for r in diagnosed)} for n in [4,8,16,24]]
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({"status":"post-hoc diagnosis, not candidate performance",
                                   "uses_query_inputs_or_labels":False,"feeds_repairs_back_into_candidate":False,
                                   "summary":summary,"rows":diagnosed},indent=2),encoding="utf-8")
    print(json.dumps(summary),flush=True)


if __name__=="__main__": main()
