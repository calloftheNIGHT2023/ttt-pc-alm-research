"""Conditional quadratic proposals for profiled activity/shared-bias blocks."""
from __future__ import annotations
import numpy as np
import streaming_branch_projection as base
from joint_activity_bias_memory import profile,joint_block as grid_block


def conditional_proposals(a,target,oldh,oldb,seeds,trust=.01):
    hs,_=profile(a,target,oldh,oldb,seeds,trust)
    bias=seeds[:,:,None]
    regs=np.searchsorted(base.KNOTS,hs+bias,side="right")
    ss=base.SLOPES[regs]; cc=base.INTERCEPTS[regs]
    aa=a[:,None,:]; tt=target[:,None,:]; oh=oldh[:,None,:]
    free=(aa+ss*(tt-ss*bias-cc)+trust*oh)/(1+ss**2+trust)
    lowraw=np.array([-np.inf,0,.5,1])[regs]-bias
    highraw=np.array([0,.5,1,np.inf])[regs]-bias
    lo=np.maximum(0,lowraw); hi=np.minimum(1,highraw)
    dh=-ss**2/(1+ss**2+trust)
    dh=np.where(free<lo,np.where(lowraw>0,-1.,0.),dh)
    dh=np.where(free>hi,np.where(highraw<1,-1.,0.),dh)
    intercept=hs-dh*bias
    aa2=np.mean((1+trust)*dh**2+ss**2*(1+dh)**2,axis=2)+trust
    bb=np.mean(dh*(aa-intercept)+ss*(1+dh)*(tt-ss*intercept-cc)+trust*dh*(oh-intercept),axis=2)+trust*oldb[:,None]
    return np.clip(bb/aa2,-base.BOUND,base.BOUND)


def energy(a,target,oldh,oldb,h,b,trust=.01):
    return np.mean((h-a)**2+(base.g(h+b[:,None])-target)**2+trust*(h-oldh)**2,axis=1)+trust*(b-oldb)**2


def coordinate_block(a,target,oldh,oldb,passes=8,trust=.01):
    b=oldb.copy()
    for _ in range(passes):
        hs,_=profile(a,target,oldh,oldb,b[:,None],trust)
        b=base.bias_solve(hs[:,0,:],target,oldb,0.,float("inf"),trust)
    hs,_=profile(a,target,oldh,oldb,b[:,None],trust); h=hs[:,0,:]
    oldvalue=energy(a,target,oldh,oldb,oldh,oldb,trust)
    newvalue=energy(a,target,oldh,oldb,h,b,trust)
    assert np.all(newvalue<=oldvalue+1e-10)
    return h,b,{"max_increase_over_old":float(np.max(newvalue-oldvalue)),"max_increase_over_separate":0.}


def analytic_block(a,target,oldh,oldb,trust=.01):
    gh,gb,_=grid_block(a,target,oldh,oldb,trust=trust)
    r=len(oldb)
    seeds=np.column_stack([np.broadcast_to(np.linspace(-base.BOUND,base.BOUND,17),(r,17)),oldb,gb])
    proposed=conditional_proposals(a,target,oldh,oldb,seeds,trust)
    candidates=np.column_stack([proposed,gb,oldb])
    hs,values=profile(a,target,oldh,oldb,candidates,trust)
    ii=np.argmin(values,axis=1); rr=np.arange(r)
    h,b=hs[rr,ii],candidates[rr,ii]
    oldvalue=energy(a,target,oldh,oldb,oldh,oldb,trust)
    gridvalue=energy(a,target,oldh,oldb,gh,gb,trust)
    val=values[rr,ii]
    assert np.all(val<=gridvalue+1e-10)
    return h,b,{"max_increase_over_old":float(np.max(val-oldvalue)),
               "max_increase_over_separate":float(np.max(val-gridvalue))}
