"""Vectorized Adam / damped Gauss-Newton controls, not candidate mechanisms.

Whole-network chain derivatives are intentionally allowed for BP baselines.
The same support-only best-point rule is applied to all evaluated trial points.
"""
import numpy as np
import streaming_branch_projection as base
from matched_discovery_baselines import loss_gradient


def evaluate(bank,x,v,with_jacobian=True):
    r,d=bank.shape; n=len(x); h=np.broadcast_to(x,(r,n))
    jac=np.zeros((r,n,d)) if with_jacobian else None
    for j in range(d):
        z=h+bank[:,j,None]
        if with_jacobian:
            slope=base.derivative(z); jac*=slope[:,:,None]; jac[:,:,j]+=slope
        h=base.g(z)
    raw=h-v; residual=np.sign(raw)*np.maximum(np.abs(raw)-base.EPS,0)
    if with_jacobian:jac*=np.abs(raw[:,:,None])>base.EPS
    return .5*np.mean(residual**2,axis=1),residual,jac,raw


def refine(starts,x,v,anchor,*,solver="gauss_newton",steps=20,lr=.003):
    b=starts.copy(); r,d=b.shape; n=len(x)
    best=b.copy(); errors,moves=base.score(best,x,v,anchor)
    def retain(trials,raw):
        err=np.max(np.abs(raw),axis=1); move=.5*np.sum((trials-anchor)**2,axis=1)
        update=base.better(err,move,errors,moves)
        best[update]=trials[update]; errors[update]=err[update]; moves[update]=move[update]
    m=np.zeros_like(b); second=np.zeros_like(b); damping=np.full(r,.01)
    evaluation_rows=0; derivative_rows=0; accepted=0
    for it in range(1,steps+1):
        value,residual,jac,raw=evaluate(b,x,v); retain(b,raw)
        evaluation_rows+=r; derivative_rows+=r
        grad=np.einsum("rni,rn->ri",jac,residual,optimize=False)/n
        if solver=="adam":
            m=.9*m+.1*grad; second=.999*second+.001*grad**2
            b=np.clip(b-lr*(m/(1-.9**it))/(np.sqrt(second/(1-.999**it))+1e-8),-base.BOUND,base.BOUND)
        elif solver=="gauss_newton":
            hessian=np.einsum("rni,rnj->rij",jac,jac,optimize=False)/n
            scale=np.maximum(np.diagonal(hessian,axis1=1,axis2=2),1e-4)
            system=hessian+damping[:,None,None]*np.eye(d)[None]*scale[:,None,:]
            delta=np.linalg.solve(system,-grad[:,:,None])[:,:,0]
            # Bounded joint proposal and a fixed support-only backtracking grid.
            delta*=np.minimum(1,.05/np.maximum(np.max(np.abs(delta),axis=1),1e-30))[:,None]
            nextb=b.copy(); nextvalue=value.copy(); changed=np.zeros(r,dtype=bool)
            for alpha in [1.,.5,.25,.125,.0625]:
                trial=np.clip(b+alpha*delta,-base.BOUND,base.BOUND)
                val,_,_,raw=evaluate(trial,x,v,False); evaluation_rows+=r; retain(trial,raw)
                better=val<nextvalue
                nextb[better]=trial[better]; nextvalue[better]=val[better]; changed|=better
            b=nextb; damping=np.clip(damping*np.where(changed,.3,10.),1e-8,1e8); accepted+=int(changed.sum())
        else:raise ValueError(solver)
    _,_,_,raw=evaluate(b,x,v,False); retain(b,raw); evaluation_rows+=r
    return best,{"batched_bp_solver":solver,"batched_steps":steps,"learning_rate":lr if solver=="adam" else None,
        "per_restart_forward_evaluations_total":evaluation_rows,"per_restart_global_jacobian_evaluations_total":derivative_rows,
        "accepted_joint_steps":accepted,"major_arrays_bytes_subtotal":sum(t.nbytes for t in [b,best,m,second,jac,residual]),
        "state_scope":"array subtotal only; excludes temporary solve/line-search and Python/native workspace"}


def verify():
    rng=np.random.default_rng(591145); maxjac=0.; maxgradient=0.; maxfinite=0.; rowcases=0
    for d,n in [(2,5),(4,8),(4,24)]:
        x=rng.uniform(0,1,n); v=rng.uniform(0,1,n); bank=rng.uniform(-.12,.12,(7,d))
        value,residual,jac,raw=evaluate(bank,x,v)
        grad=np.einsum("rni,rn->ri",jac,residual,optimize=False)/n
        for i,b in enumerate(bank):
            pred,expected=base.forward_jacobian(x,b); expected*=np.abs(pred-v)[:,None]>base.EPS
            reference=loss_gradient(b,x,v,True)
            maxjac=max(maxjac,float(np.max(np.abs(jac[i]-expected))))
            maxgradient=max(maxgradient,float(np.max(np.abs(grad[i]-reference[1]))))
            numeric=np.array([(evaluate((b+np.eye(d)[j]*1e-7)[None],x,v,False)[0][0]-evaluate((b-np.eye(d)[j]*1e-7)[None],x,v,False)[0][0])/2e-7 for j in range(d)])
            maxfinite=max(maxfinite,float(np.max(np.abs(numeric-grad[i])))); rowcases+=1
        for solver in ["adam","gauss_newton"]:
            joint,_=refine(bank,x,v,np.zeros(d),solver=solver,steps=7)
            individual=np.vstack([refine(b[None],x,v,np.zeros(d),solver=solver,steps=7)[0] for b in bank])
            assert np.allclose(joint,individual,atol=1e-12,rtol=1e-12)
    assert maxjac<1e-12 and maxgradient<1e-12 and maxfinite<1e-6
    return {"passed":True,"whole_chain_jacobian_reference_max_error":maxjac,"reverse_bp_gradient_max_error":maxgradient,
        "finite_difference_max_error":maxfinite,"derivative_cases":rowcases,"joint_vs_independent_restart_equivalence":True}


if __name__=="__main__":
    import json
    print(json.dumps(verify()))
