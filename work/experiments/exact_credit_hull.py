"""Two-sided, exact certificates for a FIXED local-credit convex hull.

Floating LPs propose. Positive dual certificates use the existing exact
validator. A nonpositive result needs an exact common primal witness, not
an LP value/tolerance. No claim about all possible optimizer trajectories.
"""
from fractions import Fraction as F
from math import lcm
import time
import numpy as np
from scipy.linalg import qr
from scipy.optimize import linprog
import joint_credit_minimax as joint


def pack(q):return [str(q.numerator),str(q.denominator)]
def unpack(q):return F(int(q[0]),int(q[1]))


class RationalBank:
    """Binary floats share a power-of-two denominator: fast exact dot products."""
    def __init__(self,bank):
        self.bank=bank
        ratios=[[float(t).as_integer_ratio() for t in row] for row in bank.reshape(len(bank),-1)]
        self.exponent=max(d.bit_length()-1 for row in ratios for _,d in row)
        self.denominator=1<<self.exponent
        self.rows=[tuple(n<<(self.exponent-(d.bit_length()-1)) for n,d in row) for row in ratios]

    def values(self,residual):
        den=lcm(*(v.denominator for v in residual));nums=[v.numerator*(den//v.denominator) for v in residual]
        vals=[sum(a*b for a,b in zip(row,nums)) for row in self.rows]
        return vals,den*self.denominator


def affine_domain(x,v,reg):
    """Eliminate z; exact RHS values are retained separately from LP proposals."""
    d,n=reg.shape;m=d*n;q=d+m
    zl,zh,hl,hh=[a[0] for a in joint.original.screen.boxes(v,reg[None])]
    rows=[];rhs=[]
    for j,(lo,hi) in enumerate([(-.12,.12)]*d+list(zip(hl.ravel(),hh.ravel()))):
        a=np.zeros(q,dtype=np.int64);a[j]=1;rows.extend([a,-a]);rhs.extend([F(float(hi)),-F(float(lo))])
    rr=np.zeros((m,q),dtype=np.int64);cc=[]
    for j in range(d):
        for i in range(n):
            t=j*n+i;a=np.zeros(q,dtype=np.int64);a[j]=1
            off=F(float(x[i])) if j==0 else F(0)
            if j:a[d+(j-1)*n+i]=1
            rows.extend([a,-a]);rhs.extend([F(float(zh[j,i]))-off,off-F(float(zl[j,i]))])
            ss=int(joint.original.base.SLOPES[reg[j,i]]);c=int(joint.original.base.INTERCEPTS[reg[j,i]])
            rr[t]=-ss*a;rr[t,d+t]+=1;cc.append(-ss*off-c)
    return np.array(rows),rhs,rr,cc


def check_witness(x,v,reg,bank,y):
    """Direct exact evaluation, not a tolerance test of the construction matrix."""
    d,n=reg.shape
    if len(y)!=d+d*n:return dict(accepted=False,reason='dimension')
    zl,zh,hl,hh=[a[0] for a in joint.original.screen.boxes(v,reg[None])]
    b=y[:d];h=[y[d+j*n:d+(j+1)*n] for j in range(d)];residual=[]
    for t in b:
        if not -F(.12)<=t<=F(.12):return dict(accepted=False,reason='bias box')
    for j in range(d):
        for i in range(n):
            z=b[j]+(F(float(x[i])) if j==0 else h[j-1][i])
            if not F(float(zl[j,i]))<=z<=F(float(zh[j,i])):return dict(accepted=False,reason='preactivation box')
            if not F(float(hl[j,i]))<=h[j][i]<=F(float(hh[j,i])):return dict(accepted=False,reason='activity box')
            residual.append(h[j][i]-int(joint.original.base.SLOPES[reg[j,i]])*z-int(joint.original.base.INTERCEPTS[reg[j,i]]))
    vals,den=bank.values(residual);maximum=max(vals);mx=F(maximum,den)
    return dict(accepted=maximum<=0,reason='exact common witness' if maximum<=0 else 'positive credit',
        max_credit=pack(mx),max_credit_float=float(mx),directions_checked=len(vals),residual=[pack(r) for r in residual])


def exact_linear_solve(rows,rhs):
    """Small square rational Gaussian elimination, with exact singularity checks."""
    n=len(rhs);a=[list(map(F,row))+[F(b)] for row,b in zip(rows,rhs)]
    for col in range(n):
        pivot=next((j for j in range(col,n) if a[j][col]),None)
        if pivot is None:return None
        a[col],a[pivot]=a[pivot],a[col];scale=a[col][col]
        for j in range(col,n+1):a[col][j]/=scale
        for row in range(col+1,n):
            scale=a[row][col]
            if scale:
                for j in range(col,n+1):a[row][j]-=scale*a[col][j]
    result=[F(0)]*n
    for row in range(n-1,-1,-1):result[row]=a[row][-1]-sum(a[row][j]*result[j] for j in range(row+1,n))
    return result


def propose_witness(x,v,reg,bank):
    start=time.perf_counter();yr,yb,rr,cc=affine_domain(x,v,reg);aa=bank.bank.reshape(len(bank.rows),-1)
    credit=aa@rr;target=-aa@np.array([float(t) for t in cc]);mat=np.r_[yr,credit];rhs=np.r_[[float(t) for t in yb],target]
    lp=linprog(np.zeros(mat.shape[1]),A_ub=mat,b_ub=rhs,bounds=[(None,None)]*mat.shape[1],
        options=dict(primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    meta=dict(lp_success=bool(lp.success),lp_status=int(lp.status),lp_message=lp.message,attempts=[])
    if not lp.success:return None,dict(meta,accepted=False,seconds=time.perf_counter()-start)
    def attempt(y,label):
        proof=check_witness(x,v,reg,bank,y);meta['attempts'].append(dict(kind=label,accepted=proof['accepted'],reason=proof['reason']))
        if proof['accepted']:return dict(y=[pack(t) for t in y],proof=proof,construction=label)
        return None
    exact=[F(float(t)) for t in lp.x]
    for label,y in [('binary',exact)]+[(f'limit_{den}',[t.limit_denominator(den) for t in exact]) for den in [10**6,10**9]]:
        result=attempt(y,label)
        if result is not None:return result,dict(meta,accepted=True,seconds=time.perf_counter()-start)
    cache={};ny=len(yr);dim=mat.shape[1]
    def exact_row(index):
        if index<ny:return list(map(int,yr[index])),yb[index]
        if index not in cache:
            a=bank.rows[index-ny]
            row=[F(sum(a[t]*int(rr[t,j]) for t in range(len(a))),bank.denominator) for j in range(dim)]
            b=-sum((F(val,bank.denominator)*t for val,t in zip(a,cc)),F(0));cache[index]=(row,b)
        return cache[index]
    slack=rhs-mat@lp.x;norm=np.linalg.norm(mat,axis=1);seen=set()
    for tol in [1e-9,1e-7,1e-5]:
        ids=np.flatnonzero((abs(slack)<=tol*(1+abs(rhs)))&(norm>0))
        if len(ids)<dim:meta['attempts'].append(dict(kind=f'basis_{tol}',accepted=False,reason='too few active rows'));continue
        _,_,perm=qr((mat[ids]/norm[ids,None]).T,pivoting=True,mode='economic');chosen=tuple(int(t) for t in ids[perm[:dim]])
        if chosen in seen:continue
        seen.add(chosen);eq=[exact_row(i) for i in chosen];y=exact_linear_solve([a for a,b in eq],[b for a,b in eq])
        if y is None:meta['attempts'].append(dict(kind=f'basis_{tol}',accepted=False,reason='exact singular'));continue
        result=attempt(y,f'basis_{tol}')
        if result is not None:
            result['active_rows']=list(chosen)
            return result,dict(meta,accepted=True,seconds=time.perf_counter()-start)
    return None,dict(meta,accepted=False,seconds=time.perf_counter()-start)


def solve(x,v,reg,bank):
    start=time.perf_counter();data,dual=joint.solve(x,v,reg,bank.bank)
    if data is not None and dual['positive']:
        return dict(status='positive',dual=dual,weights=data['weights'].tolist(),credit=data['credit'].tolist(),seconds=time.perf_counter()-start)
    witness,meta=propose_witness(x,v,reg,bank)
    return dict(status='nonpositive' if witness is not None else 'unknown',dual=dual,witness=witness,primal=meta,seconds=time.perf_counter()-start)
