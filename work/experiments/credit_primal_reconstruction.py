"""Support-only parameter proposals from the analytic dual's primal witness.

The layerwise relaxed minimizer is NOT a fitted network. Its true forward path
is evaluated and its actual branch must be independently materialized later.
"""
import numpy as np
import optimized_branch_dual as dual
base=dual.base;screen=dual.screen


def reconstruct(x,v,regs,a):
    count,d,n=regs.shape;zl,zh,hl,hh=screen.boxes(v,regs);sa=base.SLOPES[regs]*a
    b=np.zeros((count,d));h=np.empty((count,d,n));z=np.empty_like(h);valid=np.ones(count,bool)
    h[:,-1]=np.where(a[:,-1]>0,hl[:,-1],np.where(a[:,-1]<0,hh[:,-1],(hl[:,-1]+hh[:,-1])/2))
    for j in range(d):
        pl=np.broadcast_to(x,(count,n)) if j==0 else hl[:,j-1]
        ph=np.broadcast_to(x,pl.shape) if j==0 else hh[:,j-1]
        ap=np.zeros_like(pl) if j==0 else a[:,j-1];c=ap-sa[:,j]
        low=np.maximum(-screen.B,(zl[:,j]-ph).max(1));high=np.minimum(screen.B,(zh[:,j]-pl).min(1));valid&=low<=high
        kink=np.where(c>=0,zl[:,j]-pl,zh[:,j]-ph);options=np.clip(np.column_stack([low,high,kink]),low[:,None],high[:,None])
        lo=np.maximum(pl[:,None],zl[:,j,None]-options[:,:,None]);hi=np.minimum(ph[:,None],zh[:,j,None]-options[:,:,None])
        values=(c[:,None]*np.where(c[:,None]>=0,lo,hi)-sa[:,j,None]*options[:,:,None]).sum(2)
        best=values.min(1);ties=values==best[:,None]
        left=np.where(ties,options,np.inf).min(1);right=np.where(ties,options,-np.inf).max(1)
        b[:,j]=(left+right)/2
        lo=np.maximum(pl,zl[:,j]-b[:,j,None]);hi=np.minimum(ph,zh[:,j]-b[:,j,None])
        hp=np.where(c>0,lo,np.where(c<0,hi,(lo+hi)/2))
        if j:h[:,j-1]=hp
        z[:,j]=hp+b[:,j,None]
    residual=h-(base.SLOPES[regs]*z+base.INTERCEPTS[regs]);objective=(a*residual).sum((1,2))
    upper=(np.max(np.abs(residual),axis=2)*2.**np.arange(d-1,-1,-1)[None]).sum(1)
    forward=np.stack([base.forward(x,bb) for bb in b]);forward_gap=np.max(np.abs(forward-h[:,-1]),axis=1)
    assert np.all(forward_gap[valid]<=upper[valid]+1e-10)
    return dict(b=b,h=h,z=z,residual=residual,objective=objective,valid=valid,forward_gap=forward_gap,propagated_residual_bound=upper,
        observed_support_max_error=np.max(np.abs(forward-v),axis=1))


def verify():
    rng=np.random.default_rng(461183);cases=0;errors=[]
    for d,n in [(1,2),(2,4),(4,4),(4,8)]:
        x=rng.uniform(0,1,n);v=base.forward(x,rng.uniform(-.12,.12,d));bank=rng.uniform(-.12,.12,(32,d));regs=np.array([base.pattern(x,b) for b in bank],np.uint8)
        for a in [rng.normal(size=regs.shape),np.zeros(regs.shape)]:
            out=reconstruct(x,v,regs,a);value=dual.float_optimum(x,v,regs,a);valid=out['valid'];assert np.array_equal(valid,np.isfinite(value))
            assert np.max(np.abs(out['b'][valid]))<=screen.B+1e-14
            diff=float(np.max(np.abs(out['objective'][valid]-value[valid]),initial=0));errors.append(diff);assert diff<1e-10
            zl,zh,hl,hh=screen.boxes(v,regs)
            assert np.all(out['z'][valid]>=zl[valid]-1e-12) and np.all(out['z'][valid]<=zh[valid]+1e-12)
            assert np.all(out['h'][valid]>=hl[valid]-1e-12) and np.all(out['h'][valid]<=hh[valid]+1e-12)
            true=np.stack([base.forward(x,b) for b in out['b']]);assert np.all(np.max(np.abs(true[valid]-out['h'][valid,-1]),axis=1)<=out['propagated_residual_bound'][valid]+1e-10)
            cases+=len(regs)
    return dict(passed=True,cases=cases,max_objective_vs_optimized_dual_error=max(errors),full_forward_error_bound_checked=True,
        zero_credit_is_closed_form_box_midpoint_control=True,scope='primal relaxation witness is only a proposal, never assumed to solve observed task')


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
