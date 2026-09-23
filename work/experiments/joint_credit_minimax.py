"""Full existing-credit convex hull over preactivation-consistent boxes.

HiGHS only proposes a combination. A positive result is accepted after a
Fraction fixed-credit certificate AND an exact convex-mixture rounding bound.
"""
import time
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog
import optimized_branch_dual as original


def system(x,v,reg,bank):
    k,d,n=bank.shape;m=d*n;dim=d+2*m
    zl,zh,hl,hh=[a[0] for a in original.screen.boxes(v,reg[None])]
    eq=np.zeros((m,dim+1));rhs=np.zeros(m)
    for j in range(d):
        for i in range(n):
            row=j*n+i;eq[row,j]=-1;eq[row,d+row]=1
            if j:eq[row,d+m+(j-1)*n+i]=-1
            else:rhs[row]=x[i]
    aa=bank.reshape(k,m);s=original.base.SLOPES[reg].ravel();c=original.base.INTERCEPTS[reg].ravel()
    ub=np.column_stack([np.zeros((k,d)),-aa*s,aa,-np.ones(k)])
    target=aa@c
    bounds=[(-original.screen.B,original.screen.B)]*d+list(zip(zl.ravel(),zh.ravel()))+list(zip(hl.ravel(),hh.ravel()))+[(None,None)]
    obj=np.zeros(dim+1);obj[-1]=1
    return obj,ub,target,eq,rhs,bounds


def certify_mixture(x,v,reg,bank,weights,a):
    proof=original.exact_optimum(x,v,reg,a)
    if 'empty_layer' in proof:return dict(accepted=False,fixed_credit=proof,reason='Empty relaxed domain is a separate certificate')
    ids=np.flatnonzero(weights>0);ww=[F(float(weights[i])) for i in ids];total=sum(ww,F(0));assert total>0
    ww=[w/total for w in ww]
    exact_a=[sum((w*F(float(bank[i].ravel()[j])) for i,w in zip(ids,ww)),F(0)) for j in range(a.size)]
    err=sum((abs(target-F(float(actual))) for target,actual in zip(exact_a,a.ravel())),F(0))
    # Both h and the branch activation s*z+c lie in [0,1] on Y, so |r|<=1.
    value=F(int(proof['numerator']),int(proof['denominator']));lower=value-err
    return dict(accepted=lower>0,fixed_credit=proof,mixture_rounding_l1=dict(numerator=str(err.numerator),denominator=str(err.denominator),value=float(err)),
        exact_convex_lower=dict(numerator=str(lower.numerator),denominator=str(lower.denominator),value=float(lower)),
        support_ids=ids.tolist(),float_weights=weights[ids].tolist(),weight_rule='Positive stored float weights interpreted as rationals, then normalized exactly',
        claim='D of an exact convex mixture is at least D(stored float credit) minus its exact L1 rounding error')


def solve(x,v,reg,bank):
    begin=time.perf_counter();args=system(x,v,reg,bank);obj,ub,target,eq,rhs,bounds=args
    build=time.perf_counter()-begin;start=time.perf_counter()
    lp=linprog(obj,A_ub=ub,b_ub=target,A_eq=eq,b_eq=rhs,bounds=bounds,method='highs',
        options=dict(dual_feasibility_tolerance=1e-9,primal_feasibility_tolerance=1e-9))
    lp_time=time.perf_counter()-start
    meta=dict(success=bool(lp.success),status=int(lp.status),message=lp.message,iterations=int(lp.nit),
        build_seconds=build,lp_seconds=lp_time,numeric_arrays_bytes=sum(a.nbytes for a in args[:5]),directions=len(bank))
    if not lp.success:return None,dict(meta,seconds=time.perf_counter()-begin)
    weights=np.maximum(0.,-lp.ineqlin.marginals);assert abs(weights.sum()-1)<1e-7
    weights/=weights.sum();a=np.einsum('k,kdn->dn',weights,bank)
    start=time.perf_counter();proof=certify_mixture(x,v,reg,bank,weights,a);cert_time=time.perf_counter()-start
    residual=(lp.x[reg.shape[0]+reg.size:-1]-original.base.SLOPES[reg].ravel()*lp.x[reg.shape[0]:reg.shape[0]+reg.size]-original.base.INTERCEPTS[reg].ravel())
    data=dict(credit=a,weights=weights,primal=lp.x)
    meta.update(numerical_value=float(lp.fun),primal_max_credit=float(np.max(bank.reshape(len(bank),-1)@residual)),
        proposed_credit_value=proof['fixed_credit']['value'],duality_gap=float(lp.fun-proof['fixed_credit']['value']),
        positive=proof['accepted'],proof=proof,certification_seconds=cert_time,seconds=time.perf_counter()-begin)
    return data,meta
