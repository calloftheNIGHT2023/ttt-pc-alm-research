"""Independent split LP is used only to audit branch proposals, never fitting."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import streaming_branch_projection as base
from dual_branch_cut import BranchCut


def lp_feasible(x,v,regs):
    depth,n=regs.shape; size=depth+2*depth*n
    a=np.zeros((2*depth*n,size)); rhs=np.zeros(2*depth*n)
    lo=np.array([-base.BOUND,0,.5,1]); hi=np.array([0,.5,1,1+base.BOUND])
    bounds=[(-base.BOUND,base.BOUND)]*depth
    bounds += [(lo[j],hi[j]) for j in regs.ravel()]
    bounds += [(max(0,v[i]-base.EPS-base.TOL),min(1,v[i]+base.EPS+base.TOL)) if l==depth-1 else (0,1)
               for l in range(depth) for i in range(n)]
    for l in range(depth):
        for i in range(n):
            k=l*n+i; zi=depth+k; hj=depth+depth*n+k
            a[k,zi]=1; a[k,l]=-1
            if l==0: rhs[k]=x[i]
            else: a[k,hj-n]=-1
            a[depth*n+k,hj]=1; a[depth*n+k,zi]=-base.SLOPES[regs[l,i]]
            rhs[depth*n+k]=base.INTERCEPTS[regs[l,i]]
    res=linprog(np.zeros(size),A_eq=a,b_eq=rhs,bounds=bounds,
                options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
    assert res.status in [0,2],res.message
    return res.status==0


def main():
    root=Path("outputs/ttt-pc-alm-research/results/certified_branch_search")
    rows=json.loads((root/"certificate_audit_v2/regions.json").read_text(encoding="utf-8"))
    output=[]
    for r in rows:
        cert=r["infeasibility_certificate"]
        if cert is None: continue
        rng=np.random.default_rng(r["seed"])
        truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24)[:r["n_context"]]; v=base.forward(x,truth)
        regs=np.array(r["regional_pattern"]); cut=BranchCut(regs,cert)
        assert cut.exact_bound(regs)==cut.d0
        # Teacher used in this audit only, never by proposal generation.
        valid=base.pattern(x,truth)
        assert cut.exact_bound(valid)<=0 and not cut.excludes(valid)
        assert not lp_feasible(x,v,regs)
        proposals,meta=cut.proposals()
        item={"seed":r["seed"],"n_context":len(x),**meta,"proposals":[]}
        for p in proposals:
            item["proposals"].append({"changed_coordinates":int(np.sum(p!=regs)),
                "cut_bound":float(cut.exact_bound(p)),"lp_feasible":lp_feasible(x,v,p)})
        output.append(item)
    result={"passed":True,"old_infeasible_regions":len(output),"rows":output}
    (root/"dual_cut_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__": main()
