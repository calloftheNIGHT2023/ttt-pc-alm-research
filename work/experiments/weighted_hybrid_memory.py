"""Local branch discovery followed by global convex regional projection.

This intentionally replaces pure-local refinement; it is not all-local credit.
No nonlinear global loss derivative or original TTT backward is used by fit.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog,minimize
import streaming_branch_projection as base
from weighted_discovery_bank import discover


def fit(x,v,anchor,cfg):
    if np.max(np.abs(base.forward(x,anchor)-v))<=base.EPS:
        return anchor.copy(),{"skipped":True}
    opts={k:cfg[k] for k in ["sweeps","restarts","dual_rate","gradient_blocks","layer_ratio"] if k in cfg}
    bank,meta=discover(x,v,anchor,**opts)
    best=bank[0].copy(); err0,mov0=base.score(best[None],x,v,anchor)
    tried=0; feasible=0
    for start in bank[:cfg.get("max_regions",8)]:
        tried+=1
        _,_,mat,rhs=base.branch_polytope(x,v,start)
        lp=linprog(np.zeros(len(anchor)),A_ub=mat,b_ub=rhs,bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                   options={"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
        if not lp.success: continue
        feasible+=1
        res=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),lp.x,jac=True,method="SLSQP",
             constraints={"type":"ineq","fun":lambda b:rhs-mat@b,"jac":lambda b:-mat},
             bounds=[(-base.BOUND,base.BOUND)]*len(anchor),options={"ftol":1e-12,"maxiter":300})
        for b in [lp.x,res.x]:
            err,mov=base.score(b[None],x,v,anchor)
            if base.better(err,mov,err0,mov0)[0]: best,err0,mov0=b.copy(),err,mov
        if err0[0]<=base.EPS+base.TOL: break
    return best,{**meta,"regions_attempted":tried,"lp_feasible_regions":feasible}
