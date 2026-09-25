"""Matched-start strong BP controls with explicit finite iteration budgets."""
import time
import numpy as np
from scipy.optimize import minimize,least_squares
import streaming_branch_projection as base
import contextual_candidate_bank as candidates
from matched_discovery_baselines import loss_gradient
from run_local_screening import screen_bank,solve_bank
import variance_controlled_readout as readout
import screened_contextual_memory as candidate_pipeline


def refine(starts,x,v,anchor,solver,budget):
    best=starts.copy(); errors,moves=base.score(best,x,v,anchor); calls=0; iterations=0; status=[]
    for i,start in enumerate(starts):
        cache_b=None; cache=None
        def evaluate(b):
            nonlocal calls,cache_b,cache
            if cache_b is not None and np.array_equal(cache_b,b):return cache
            calls+=1
            if solver=="trf":
                pred,jac=base.forward_jacobian(x,b); raw=pred-v
                residual=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0)
                cache=(residual,jac*(np.abs(raw)>base.EPS)[:,None])
            else:
                loss,grad,raw=loss_gradient(b,x,v,True); cache=(loss,grad)
            err=float(np.max(np.abs(raw))); move=.5*float(np.sum((b-anchor)**2))
            if base.better(np.array([err]),np.array([move]),errors[i:i+1],moves[i:i+1])[0]:
                best[i]=b; errors[i]=err; moves[i]=move
            cache_b=b.copy(); return cache
        if solver=="trf":
            res=least_squares(lambda z:evaluate(z)[0],start,jac=lambda z:evaluate(z)[1],bounds=(-base.BOUND,base.BOUND),
                method="trf",max_nfev=budget,ftol=1e-12,xtol=1e-12,gtol=1e-10)
            iterations+=res.nfev
        else:
            res=minimize(evaluate,start,jac=True,method="L-BFGS-B",bounds=[(-base.BOUND,base.BOUND)]*len(anchor),
                options={"maxiter":budget,"ftol":1e-13,"gtol":1e-9,"maxls":40})
            iterations+=res.nit
        evaluate(res.x); status.append(int(res.status))
    return best,{"refinement_solver":solver,"per_start_budget":budget,"function_derivative_evaluations":calls,
        "sum_solver_iterations_or_nfev":iterations,"solver_statuses":status}


def fit(x,v,anchor,cfg):
    if cfg["generator"]=="alm":return candidate_pipeline.fit(x,v,anchor,cfg)
    begin=time.perf_counter(); starts,pm=candidates.proposals(x,v,anchor,cfg["features"],cfg["restarts"])
    refined,meta=refine(starts,x,v,anchor,cfg["generator"],cfg["budget"])
    bank=candidates.deduplicate(np.vstack([starts,refined]),x,v,anchor); discovery=time.perf_counter()-begin
    mask,proofs,_,screen_time=screen_bank(x,v,bank,"contract5"); assert not proofs
    polys,point,ids,notes,geometry_time=solve_bank(x,v,anchor,bank,mask); predict,rm=readout.build(polys,point,"adaptive_iid",1e-6)
    return predict,point,{**pm,**meta,**rm,"discovery_seconds":discovery,"screen_seconds":screen_time,"geometry_seconds":geometry_time,
        "candidates":len(bank),"screened":int(mask.sum()),"positive_volume_regions":len(ids),"geometry_trace":notes,
        "persistent_state_bytes":point.nbytes+rm["readout_state_bytes"],"anchor_output":point.tolist()}


def verify():
    rng=np.random.default_rng(23925); x=rng.uniform(0,1,8); v=base.forward(x,rng.uniform(-.12,.12,4)); anchor=np.zeros(4)
    starts,_=candidates.proposals(x,v,anchor,256,8)
    for solver in ["trf","lbfgs"]:
        got,_=refine(starts,x,v,anchor,solver,300); expected,_=candidates.refine_bp(starts,x,v,anchor,solver)
        assert np.array_equal(got,expected)
    return {"passed":True,"budget300_refiners_identical_to_original":True}
