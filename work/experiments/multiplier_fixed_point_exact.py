"""Rational recovery and exhaustive scalar-block fixed-point verification.

All input floats and constants are interpreted as their exact binary rationals.
Screening/active-set tolerances propose equations only; acceptance is exact.
The prox term remains in global comparisons (its derivative is zero at a fixed
point). Rank deficiency is exposed and free variables retain the saved value.
"""
from fractions import Fraction as F
import numpy as np

B=F(.12);EPS=F(.001);TRUST=F(.01)
K=[F(0),F(1,2),F(1)];S=[0,2,-2,0];C=[0,0,2,0]
def pack(q):return [str(q.numerator),str(q.denominator)]
def unpack(q):return F(int(q[0]),int(q[1]))
def g(z):return max(F(0),1-abs(2*z-1))
def branch(z):return sum(z>=k for k in K)
def clip(z,lo,hi):return min(max(z,lo),hi)
def bounds(v):
    # Match the original code's rounded computation of hard activity bounds.
    return [F(float(max(0,float(t)-.001))) for t in v],[F(float(min(1,float(t)+.001))) for t in v]

def rref_solve(a,b,proposal):
    m=len(a);n=len(proposal);a=[list(map(F,row))+[F(t)] for row,t in zip(a,b)];piv=[];r=0
    for col in range(n):
        p=next((i for i in range(r,m) if a[i][col]),None)
        if p is None:continue
        a[p],a[r]=a[r],a[p];scale=a[r][col];a[r]=[t/scale for t in a[r]]
        for i in range(m):
            if i!=r and a[i][col]:
                scale=a[i][col];a[i]=[x-scale*y for x,y in zip(a[i],a[r])]
        piv.append(col);r+=1
        if r==m:break
    if any(not any(row[:n]) and row[n] for row in a):return None,dict(reason='inconsistent equations',rank=r)
    free=[j for j in range(n) if j not in piv];y=list(proposal)
    for i,j in enumerate(piv):y[j]=a[i][n]-sum(a[i][t]*y[t] for t in free)
    return y,dict(rank=r,free_variables=free,free_values=[pack(y[t]) for t in free],
        parameterization=[dict(pivot=j,constant=pack(a[i][n]),free_coefficients=[pack(-a[i][t]) for t in free]) for i,j in enumerate(piv)])

def recover(x,v,b,h):
    d,n=h.shape;q=d+d*n;xx=list(map(lambda t:F(float(t)),x));lo,hi=bounds(v)
    initial=[F(float(t)) for t in np.r_[b,h.ravel()]]
    slopes=np.empty((d,n),int);rr=[];cc=[];zrows=[];zc=[]
    for j in range(d):
        for i in range(n):
            z=[F(0)]*q;z[j]=1;off=xx[i] if j==0 else F(0)
            if j:z[d+(j-1)*n+i]=1
            reg=branch(sum(t*y for t,y in zip(z,initial))+off);s=S[reg];slopes[j,i]=s
            row=[-s*t for t in z];row[d+j*n+i]+=1
            rr.append(row);cc.append(-s*off-C[reg]);zrows.append(z);zc.append(off)
    equations=[];rhs=[];labels=[]
    def add(row,value,label):equations.append(row);rhs.append(value);labels.append(label)
    def boundary(index,value,label):
        row=[F(0)]*q;row[index]=1;add(row,value,label)
    def near(a,b):return abs(float(a-b))<=1e-10
    for j in range(d):
        if near(initial[j],-B):boundary(j,-B,f'b{j}:lower')
        elif near(initial[j],B):boundary(j,B,f'b{j}:upper')
        else:
            hit=next(((j*n+i,k) for i in range(n) for k in K if near(sum(t*y for t,y in zip(zrows[j*n+i],initial))+zc[j*n+i],k)),None)
            if hit:
                t,k=hit;add(zrows[t],k-zc[t],f'b{j}:kink:{t}:{k}')
            else:
                row=[sum(int(slopes[j,i])*rr[j*n+i][c] for i in range(n)) for c in range(q)]
                value=-sum(int(slopes[j,i])*cc[j*n+i] for i in range(n));add(row,value,f'b{j}:stationary')
    for j in range(d):
        for i in range(n):
            index=d+j*n+i;lower=lo[i] if j==d-1 else F(0);upper=hi[i] if j==d-1 else F(1)
            if near(initial[index],lower):boundary(index,lower,f'h{j},{i}:lower')
            elif near(initial[index],upper):boundary(index,upper,f'h{j},{i}:upper')
            else:
                hit=next((k for k in K if j<d-1 and near(initial[index]+initial[j+1],k)),None)
                if hit is not None:
                    row=[F(0)]*q;row[index]=row[j+1]=1;add(row,hit,f'h{j},{i}:kink')
                elif j==d-1:add(rr[j*n+i],-cc[j*n+i],f'h{j},{i}:stationary')
                else:
                    s=int(slopes[j+1,i]);row=[a-s*b for a,b in zip(rr[j*n+i],rr[(j+1)*n+i])]
                    add(row,-cc[j*n+i]+s*cc[(j+1)*n+i],f'h{j},{i}:stationary')
    y,meta=rref_solve(equations,rhs,initial);meta.update(labels=labels,equations=len(equations),variables=q)
    if y is None:return None,meta
    assert all(sum(t*z for t,z in zip(row,y))==value for row,value in zip(equations,rhs))
    meta['distance_from_saved']=float(max(abs(a-b) for a,b in zip(y,initial)))
    return (y[:d],[y[d+j*n:d+(j+1)*n] for j in range(d)]),meta

def activity_candidates(x,v,b,h,j,i,u=None):
    d=len(b);u=u if u is not None else [[F(0)]*len(x) for _ in b]
    before=h[j][i];prev=x[i] if j==0 else h[j-1][i];a=g(prev+b[j])-u[j][i]
    if j==d-1:
        lo,hi=bounds(v);z=clip((a+TRUST*before)/(1+TRUST),lo[i],hi[i])
        return [(z,(z-a)**2+TRUST*(z-before)**2)]
    target=h[j+1][i]+u[j+1][i];nb=b[j+1];ans=[]
    for k,s in enumerate(S):
        lower=F(0) if k==0 else max(F(0),K[k-1]-nb)
        upper=F(1) if k==3 else min(F(1),K[k]-nb)
        if lower>upper:continue
        off=s*nb+C[k];z=clip((a+s*(target-off)+TRUST*before)/(1+s*s+TRUST),lower,upper)
        ans.append((z,(z-a)**2+(g(z+nb)-target)**2+TRUST*(z-before)**2))
    return ans

def bias_candidates(x,b,h,j,u=None):
    n=len(x);u=u if u is not None else [[F(0)]*n for _ in b]
    previous=x if j==0 else h[j-1];target=[t+z for t,z in zip(h[j],u[j])]
    breaks=sorted({-B,B}|{k-p for p in previous for k in K if -B<k-p<B});ans=[]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        mid=(lo+hi)/2;ss=[S[branch(p+mid)] for p in previous];cc=[s*p+C[branch(p+mid)] for s,p in zip(ss,previous)]
        aa=F(sum(s*s for s in ss),n);bb=sum((s*(t-c) for s,t,c in zip(ss,target,cc)),F(0))/n
        z=clip((bb+TRUST*b[j])/(aa+TRUST),lo,hi)
        energy=sum(((g(p+z)-t)**2 for p,t in zip(previous,target)),F(0))/n+TRUST*(z-b[j])**2
        ans.append((z,energy))
    return ans

def verify(x,v,b,h):
    x=[F(float(t)) for t in x];lo,hi=bounds(v);d=len(b);n=len(x)
    if any(not -B<=t<=B for t in b):return dict(accepted=False,reason='bias outside box')
    if any(not (lo[i] if j==d-1 else 0)<=h[j][i]<=(hi[i] if j==d-1 else 1) for j in range(d) for i in range(n)):
        return dict(accepted=False,reason='activity outside box')
    checks=[];residual=[]
    for j in range(d):
        for i in range(n):
            prev=x[i] if j==0 else h[j-1][i];r=h[j][i]-g(prev+b[j]);residual.append(r)
            current=r*r+( (g(h[j][i]+b[j+1])-h[j+1][i])**2 if j<d-1 else 0)
            candidates=activity_candidates(x,v,b,h,j,i);z,energy=min(candidates,key=lambda t:t[1])
            checks.append(dict(block=f'h{j},{i}',candidates=len(candidates),gap=pack(energy-current),selected_same=z==h[j][i],
                other_minimizers=sum(e==energy and t!=h[j][i] for t,e in candidates)))
    for j in range(d):
        current=sum((residual[j*n+i]**2 for i in range(n)),F(0))/n
        candidates=bias_candidates(x,b,h,j);z,energy=min(candidates,key=lambda t:t[1])
        checks.append(dict(block=f'b{j}',candidates=len(candidates),gap=pack(energy-current),selected_same=z==b[j],
            other_minimizers=sum(e==energy and t!=b[j] for t,e in candidates)))
    accepted=all(unpack(c['gap'])==0 and c['selected_same'] and c['other_minimizers']==0 for c in checks)
    return dict(accepted=accepted,reason='exact unique block minima' if accepted else 'block comparison failed',blocks=checks,
        nonzero_residuals=sum(t!=0 for t in residual),residual=[pack(t) for t in residual],residual_max=pack(max(abs(t) for t in residual)))

def encode(state):
    b,h=state;return dict(b=[pack(t) for t in b],h=[[pack(t) for t in row] for row in h])
def decode(state):return [unpack(t) for t in state['b']],[[unpack(t) for t in row] for row in state['h']]
