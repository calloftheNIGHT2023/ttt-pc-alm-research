"""Exactly eliminate preactivation dual p for fixed supplied local credit a.

Layerwise scalar convex PL minimization computes max_p D(p,a) by LP strong
duality. Float values propose; exact binary-rational evaluation certifies.
This is not a closed-form solution of the nonlinear adaptation problem.
"""
from fractions import Fraction as F
import time
import numpy as np
import local_region_screen as screen
base=screen.base


def float_optimum(x,v,regs,a):
    zl,zh,hl,hh=screen.boxes(v,regs); sa=base.SLOPES[regs]*a
    total=np.minimum(a[:,-1]*hl[:,-1],a[:,-1]*hh[:,-1]).sum(1)-(a*base.INTERCEPTS[regs]).sum((1,2))
    invalid=np.zeros(len(regs),bool)
    for j in range(regs.shape[1]):
        pl=np.broadcast_to(x,(len(regs),len(x))) if j==0 else hl[:,j-1]
        ph=np.broadcast_to(x,pl.shape) if j==0 else hh[:,j-1]
        ap=np.zeros_like(pl) if j==0 else a[:,j-1]
        c=ap-sa[:,j]
        low=np.maximum(-screen.B,(zl[:,j]-ph).max(1));high=np.minimum(screen.B,(zh[:,j]-pl).min(1));invalid|=low>high
        kink=np.where(c>=0,zl[:,j]-pl,zh[:,j]-ph)
        b=np.clip(np.column_stack([low,high,kink]),low[:,None],high[:,None])
        lo=np.maximum(pl[:,None],zl[:,j,None]-b[:,:,None]);hi=np.minimum(ph[:,None],zh[:,j,None]-b[:,:,None])
        values=(c[:,None]*np.where(c[:,None]>=0,lo,hi)-sa[:,j,None]*b[:,:,None]).sum(2)
        total+=values.min(1)
    total[invalid]=np.inf
    return total


def exact_optimum(x,v,reg,a):
    """All float boxes interpreted exactly as binary rationals; no rounding claim."""
    zl,zh,hl,hh=screen.boxes(v,reg[None]);zl=zl[0];zh=zh[0];hl=hl[0];hh=hh[0]
    d,n=reg.shape; box=F(screen.B); aa=[[F(float(t)) for t in row] for row in a]
    ss=[[int(base.SLOPES[reg[j,i]])*aa[j][i] for i in range(n)] for j in range(d)]
    total=sum(min(aa[-1][i]*F(float(hl[-1,i])),aa[-1][i]*F(float(hh[-1,i]))) for i in range(n))
    total-=sum(aa[j][i]*int(base.INTERCEPTS[reg[j,i]]) for j in range(d) for i in range(n))
    minima=[]
    for j in range(d):
        pl=[F(float(x[i] if j==0 else hl[j-1,i])) for i in range(n)]
        ph=[F(float(x[i] if j==0 else hh[j-1,i])) for i in range(n)]
        zlo=[F(float(t)) for t in zl[j]];zhi=[F(float(t)) for t in zh[j]]
        c=[(F(0) if j==0 else aa[j-1][i])-ss[j][i] for i in range(n)]
        lo=max([-box]+[zlo[i]-ph[i] for i in range(n)]);hi=min([box]+[zhi[i]-pl[i] for i in range(n)])
        if lo>hi:
            gap=lo-hi
            return dict(positive=True,empty_layer=j,numerator=str(gap.numerator),denominator=str(gap.denominator),value=float(gap),claim='empty shared-bias interval, not dual objective')
        choices=[lo,hi]+[max(lo,min(hi,zlo[i]-pl[i] if c[i]>=0 else zhi[i]-ph[i])) for i in range(n)]
        values=[]
        for b in choices:
            h=[max(pl[i],zlo[i]-b) if c[i]>=0 else min(ph[i],zhi[i]-b) for i in range(n)]
            values.append(sum(c[i]*h[i]-ss[j][i]*b for i in range(n)))
        minimum=min(values);minima.append(float(minimum));total+=minimum
    return dict(positive=total>0,numerator=str(total.numerator),denominator=str(total.denominator),value=float(total),layer_minima=minima,
        claim='minimum exact linear residual objective over preactivation-consistent boxes, equal to optimized p dual by strong LP duality')


def screen_bank(x,v,regs,bank):
    begin=time.perf_counter(); proofs=[]; rejected=np.zeros(len(regs),bool)
    if not len(regs) or not len(bank):return rejected,proofs,dict(seconds=time.perf_counter()-begin,directions=len(bank),numeric_pairs=0,main_array_bytes_subtotal=0,exact_checks=0)
    k=len(bank);rr=np.repeat(regs,k,axis=0);aa=np.tile(bank,(len(regs),1,1));values=float_optimum(x,v,rr,aa).reshape(len(regs),k)
    selected=values.argmax(1);rough=values[np.arange(len(regs)),selected];scale=1+np.abs(bank[selected]).sum((1,2));exact_checks=0
    for index in np.flatnonzero(rough>1e-10*scale):
        a=bank[selected[index]];exact=exact_optimum(x,v,regs[index],a);exact_checks+=1
        if not exact['positive']:continue
        rejected[index]=True;proofs.append(dict(index=int(index),pattern=regs[index].tobytes().hex(),a=a.tolist(),direction=int(selected[index]),
            rough_optimized_value=float(rough[index]),exact=exact))
    return rejected,proofs,dict(seconds=time.perf_counter()-begin,directions=len(bank),numeric_pairs=len(rr),main_array_bytes_subtotal=rr.nbytes+aa.nbytes+values.nbytes,
        direction_bytes=bank.nbytes,exact_checks=exact_checks,scope='exact rational checking charged; temporary arrays and Python object allocation additional')


def verify():
    from scipy.optimize import linprog
    import branch_normal_credit as normal
    rng=np.random.default_rng(939122);checks=0;errors=[];oldgaps=[]
    for d,n in [(1,2),(2,3),(4,4)]:
        x=rng.uniform(0,1,n);v=base.forward(x,rng.uniform(-.12,.12,d));bs=rng.uniform(-.12,.12,(8,d));regs=np.array([base.pattern(x,b) for b in bs],np.uint8);a=rng.normal(size=regs.shape)
        values=float_optimum(x,v,regs,a);pp,bounds,_=normal.refine(x,v,regs,a,4)
        finite=np.isfinite(values);assert np.all(values[finite]>=bounds[4][finite]-1e-10)
        oldgaps.extend((values[finite]-bounds[4][finite]).tolist())
        zl,zh,hl,hh=screen.boxes(v,regs)
        for r in range(len(regs)):
            # Independent primal LP in b,z,h; only preactivation equalities.
            eq=np.zeros((d*n,d+2*d*n));rhs=np.zeros(d*n)
            for j in range(d):
                for i in range(n):
                    row=j*n+i;eq[row,j]=-1;eq[row,d+row]=1
                    if j:eq[row,d+d*n+(j-1)*n+i]=-1
                    else:rhs[row]=x[i]
            obj=np.r_[np.zeros(d),(-base.SLOPES[regs[r]]*a[r]).ravel(),a[r].ravel()]
            bb=[(-screen.B,screen.B)]*d+list(zip(zl[r].ravel(),zh[r].ravel()))+list(zip(hl[r].ravel(),hh[r].ravel()))
            lp=linprog(obj,A_eq=eq,b_eq=rhs,bounds=bb)
            exact=exact_optimum(x,v,regs[r],a[r])
            if not np.isfinite(values[r]):assert lp.status==2 and 'empty_layer' in exact
            else:
                assert lp.success;target=lp.fun-float((a[r]*base.INTERCEPTS[regs[r]]).sum())
                error=max(abs(values[r]-target),abs(values[r]-exact['value']));errors.append(error);assert error<1e-9
            checks+=1
    return dict(passed=True,independent_primal_lp_cases=checks,max_float_fraction_lp_error=max(errors),
        dominates_fixed_a_coordinate_bound=True,max_improvement_over_four_sweeps=max(oldgaps))


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
