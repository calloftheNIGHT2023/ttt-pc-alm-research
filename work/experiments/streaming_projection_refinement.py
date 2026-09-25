"""Development refinement: branch discovery followed by local convex projection.

The projection uses adjacent-layer primal-dual operations, not a whole-network
Jacobian. CP/PDHG itself is established prior art, not a novelty claim.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, linprog
import streaming_branch_projection as base


def farkas_certificate(x, v, anchor):
    _,_,g,rhs=base.branch_polytope(x,v,anchor)
    dim=len(anchor)
    g=np.vstack([g,np.eye(dim),-np.eye(dim)])
    rhs=np.r_[rhs,np.full(2*dim,base.BOUND)]
    res=linprog(rhs,A_eq=np.vstack([g.T,np.ones(len(rhs))]),b_eq=np.r_[np.zeros(dim),1.],
                bounds=(0,None),method="highs")
    if not res.success:
        return {"certified":False,"status":int(res.status)}
    # Box-aware residual correction: for any |b| <= B, w^T G b >= -B||G^T w||_1.
    # If w^T rhs is below that lower bound, feasibility is impossible.
    residual=float(np.linalg.norm(g.T@res.x,ord=1))
    rhsdot=float(rhs@res.x)
    margin=-rhsdot-base.BOUND*residual
    return {"certified":bool(margin>1e-9 and res.x.min()>=-1e-12),
            "margin":margin,"rhs_dot":rhsdot,"transpose_residual_l1":residual,
            "minimum_multiplier":float(res.x.min()),"weights":res.x.tolist()}


def branch_refine(x,v,anchor,start,steps=8000):
    """Convex fixed-branch minimum-change projection by diagonal PDHG.

    Step product bound follows weighted Cauchy-Schwarz using absolute row and
    column sums of the adjacent constraint operator. No global derivative.
    """
    depth,n=len(anchor),len(x)
    regs=base.pattern(x,start)
    s=base.SLOPES[regs]; c=base.INTERCEPTS[regs]
    lo=np.maximum(-base.BOUND,np.array([-np.inf,0,.5,1])[regs])
    hi=np.minimum(1+base.BOUND,np.array([0,.5,1,np.inf])[regs])
    hlow=np.zeros((depth,n)); hhigh=np.ones((depth,n))
    hlow[-1]=np.maximum(0,v-base.EPS); hhigh[-1]=np.minimum(1,v+base.EPS)
    b=start.copy(); z=np.empty((depth,n)); h=np.empty_like(z)
    prev=x
    for j in range(depth):
        z[j]=prev+b[j]; h[j]=base.g(z[j]); prev=h[j]
    up=np.zeros_like(h); ua=np.zeros_like(h)
    bb,zb,hb=b.copy(),z.copy(),h.copy()
    eta=.99
    tb=eta/n
    tz=eta/(1+np.abs(s))
    th=np.full((depth,1),eta/2); th[-1]=eta
    sp=np.full((depth,1),eta/3); sp[0]=eta/2
    sa=eta/(1+np.abs(s))
    best=start.copy()
    besterr,bestmove=base.score(best[None],x,v,anchor)
    lastgap=None
    for it in range(steps):
        prev=np.vstack([x,hb[:-1]])
        up+=sp*(zb-prev-bb[:,None])
        ua+=sa*(hb-s*zb-c)
        oldb,oldz,oldh=b,z,h
        b=np.clip((oldb+tb*(up.sum(axis=1)+anchor))/(1+tb),-base.BOUND,base.BOUND)
        z=np.clip(oldz-tz*(up-s*ua),lo,hi)
        hc=ua.copy(); hc[:-1]-=up[1:]
        h=np.clip(oldh-th*hc,hlow,hhigh)
        bb,zb,hb=2*b-oldb,2*z-oldz,2*h-oldh
        if (it+1)%20: continue
        err,move=base.score(b[None],x,v,anchor)
        if base.better(err,move,besterr,bestmove)[0]:
            best,besterr,bestmove=b.copy(),err,move
        # The dual bound is for the accepted epsilon+TOL problem on this branch.
        bc=-up.sum(axis=1); zc=up-s*ua
        bmin=np.clip(anchor-bc,-base.BOUND,base.BOUND)
        dl=float(.5*np.sum((bmin-anchor)**2)+bc@bmin)
        dl+=float(np.sum(zc*np.where(zc>=0,lo,hi)))
        dlo=hlow.copy(); dhi=hhigh.copy()
        dlo[-1]=np.maximum(0,v-base.EPS-base.TOL)
        dhi[-1]=np.minimum(1,v+base.EPS+base.TOL)
        dl+=float(np.sum(hc*np.where(hc>=0,dlo,dhi))-up[0]@x-np.sum(ua*c))
        # A read is feasible for the chosen branch only if its true preactivations
        # satisfy its closed intervals, not merely if free h_L was clamped.
        read=x; branch_ok=True
        for j in range(depth):
            pre=read+best[j]
            branch_ok=branch_ok and bool(np.all(pre>=lo[j]-1e-10) and np.all(pre<=hi[j]+1e-10))
            read=base.g(pre)
        if besterr[0]<=base.EPS+base.TOL and branch_ok:
            lastgap=float(bestmove[0]-dl)
            assert lastgap>=-2e-8,lastgap
            if lastgap<1e-7: break
    return best,{"polish_steps":it+1,"fixed_branch_duality_gap":lastgap,
                 "gap_tolerance":base.EPS+base.TOL,
                 "local_pdhg_only":True}


def fit_restored_slsqp(x,v,anchor,restarts=64):
    """Strong baseline: global BP feasibility restoration then constrained polish."""
    if np.max(np.abs(base.forward(x,anchor)-v))<=base.EPS:
        return anchor.copy(),{"skipped":True,"function_calls":0}
    starts=np.vstack([anchor,np.random.default_rng(912).uniform(-base.BOUND,base.BOUND,(restarts-1,len(anchor)))])
    best=anchor.copy(); besterr,bestmove=base.score(best[None],x,v,anchor)
    calls=0
    def loss(b):
        pred,jac=base.forward_jacobian(x,b)
        error=pred-v
        return .5*np.mean(error**2),jac.T@error/len(x)
    def cons(b):
        er=base.forward(x,b)-v
        return np.r_[base.EPS-er,base.EPS+er]
    def cjac(b):
        jac=base.forward_jacobian(x,b)[1]
        return np.concatenate([-jac,jac])
    for initial in starts:
        res=minimize(loss,initial,jac=True,method="L-BFGS-B",bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                     options={"maxiter":300,"ftol":1e-13,"gtol":1e-9,"maxls":40})
        calls+=res.nfev
        # Save restored solution even when constrained polish fails numerically.
        er,mv=base.score(res.x[None],x,v,anchor)
        if base.better(er,mv,besterr,bestmove)[0]: best,besterr,bestmove=res.x.copy(),er,mv
        if er[0]>base.EPS+base.TOL: continue
        ref=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),res.x,jac=True,method="SLSQP",
                     bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                     constraints={"type":"ineq","fun":cons,"jac":cjac},options={"maxiter":300,"ftol":1e-11})
        calls+=ref.nfev
        er,mv=base.score(ref.x[None],x,v,anchor)
        if base.better(er,mv,besterr,bestmove)[0]: best,besterr,bestmove=ref.x.copy(),er,mv
    return best,{"function_calls":calls,"restoration_restarts":restarts}


def verify():
    rng=np.random.default_rng(286)
    # Local fixed-branch projection against a global convex QP, no teacher seeding
    # in task evaluation; the feasible test start here is purely a solver unit test.
    discrepancies=[]; gaps=[]
    for _ in range(5):
        x=rng.uniform(0,1,12); true=rng.uniform(-.12,.12,4)
        v=base.forward(x,true); anchor=true+rng.uniform(-.02,.02,4)
        anchor=np.clip(anchor,-base.BOUND,base.BOUND)
        out,meta=branch_refine(x,v,anchor,true,steps=30000)
        _,_,mat,rhs=base.branch_polytope(x,v,true)
        ref=minimize(lambda b:(.5*np.sum((b-anchor)**2),b-anchor),true,jac=True,method="SLSQP",
                     bounds=[(-base.BOUND,base.BOUND)]*4,
                     constraints={"type":"ineq","fun":lambda b:rhs-mat@b,"jac":lambda b:-mat},
                     options={"maxiter":1000,"ftol":1e-13})
        assert ref.success
        diff=float(np.linalg.norm(out-ref.x)); discrepancies.append(diff); gaps.append(meta["fixed_branch_duality_gap"])
        assert np.max(np.abs(base.forward(x,out)-v))<=base.EPS+base.TOL
        assert diff<.002,(diff,meta)
    return {"passed":True,"local_convex_vs_qp_cases":5,"max_parameter_l2_difference":max(discrepancies),
            "local_duality_gaps":gaps}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    checks=verify(); print(json.dumps(checks),flush=True)
    (args.out/"verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    if args.verify_only: return
    protocol={"phase":"development_after_first_results","seed0":5600000,"count":args.count,
              "local_search":{"sweeps":600,"restarts":16,"warm":False},"polish_max_steps":8000,
              "strong_baseline":"64 L-BFGS restorations then SLSQP for feasible starts",
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "base_sha256":hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]
    for method in ["local_branch_refined","restored_slsqp64"]:
        for seed in range(5600000,5600000+args.count):
            rng=np.random.default_rng(seed)
            truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
            q=rng.uniform(0,1,2048); target=base.forward(q,truth)
            b=np.zeros(4); nold=0
            for n in [4,8,16,24]:
                anchor=b.copy(); diagnostic=base.branch_feasibility(x[:n],v[:n],anchor)
                if diagnostic["current_branch_feasible"] is False:
                    diagnostic["farkas"]=farkas_certificate(x[:n],v[:n],anchor)
                before=time.perf_counter()
                if method=="local_branch_refined":
                    b,_,meta=base.fit_local(x[:n],v[:n],anchor,warm=False)
                    raw_b=b.copy()
                    b,extra=branch_refine(x[:n],v[:n],anchor,b)
                    meta.update(extra)
                    meta["before_polish_max_error"]=float(np.max(np.abs(base.forward(x[:n],raw_b)-v[:n])))
                else:
                    b,meta=fit_restored_slsqp(x[:n],v[:n],anchor)
                seconds=time.perf_counter()-before
                pred=base.forward(x[:n],b); err=float(np.max(np.abs(pred-v[:n])))
                rows.append({"method":method,"seed":seed,"n_context":n,"seconds":seconds,
                             "query_mse":float(np.mean((base.forward(q,b)-target)**2)),
                             "support_max_error":err,"support_feasible":bool(err<=base.EPS+base.TOL),
                             "old_support_max_error":float(np.max(np.abs(pred[:nold]-v[:nold]))) if nold else None,
                             "b":b.tolist(),"anchor":anchor.tolist(),"move_squared":float(np.sum((b-anchor)**2)),
                             **diagnostic,**meta})
                nold=n
        for n in [4,8,16,24]:
            selected=[r for r in rows if r["method"]==method and r["n_context"]==n]
            print(json.dumps({"method":method,"n":n,"query_mse":float(np.mean([r["query_mse"] for r in selected])),
                              "feasible":sum(r["support_feasible"] for r in selected),
                              "median_seconds":float(np.median([r["seconds"] for r in selected]))}),flush=True)
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")


if __name__=="__main__": main()
