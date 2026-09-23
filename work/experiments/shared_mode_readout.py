"""Same geometry/readout for every discovered pool; exact positive-interior check."""
from fractions import Fraction as F
import time
import numpy as np
import region_posterior_memory as geometry
import multiplier_fixed_point_exact as exact

def strict_interior(x,v,pattern,point):
    d,n=pattern.shape;b=[F(float(t)) for t in point];rows=[];rhs=[];coeff=[[F(0)]*d for _ in x];offset=[F(float(t)) for t in x]
    for j in range(d):
        for i in range(n):
            z=coeff[i].copy();z[j]+=1;c=offset[i];k=int(pattern[j,i]);lo=None if k==0 else exact.K[k-1];hi=None if k==3 else exact.K[k]
            if lo is not None:rows.append([-t for t in z]);rhs.append(c-lo)
            if hi is not None:rows.append(z.copy());rhs.append(hi-c)
            coeff[i]=[exact.S[k]*t for t in z];offset[i]=exact.S[k]*c+exact.C[k]
    for i in range(n):
        rows.extend([coeff[i],[-t for t in coeff[i]]]);rhs.extend([F(float(v[i]))+exact.EPS-offset[i],offset[i]-F(float(v[i]))+exact.EPS])
    for j in range(d):
        row=[F(0)]*d;row[j]=1;rows.extend([row,[-t for t in row]]);rhs.extend([exact.B,exact.B])
    slack=[r-sum(a*t for a,t in zip(row,b)) for row,r in zip(rows,rhs)]
    accepted=all(s>0 if any(row) else s>=0 for row,s in zip(rows,slack))
    return dict(accepted=accepted,point=[exact.pack(t) for t in b],slacks=[exact.pack(t) for t in slack],
        nonconstant_constraints=sum(any(row) for row in rows),minimum_nonconstant_slack=exact.pack(min(s for row,s in zip(rows,slack) if any(row))))

def build(x,v,b):
    start=time.perf_counter();_,_,matrix,rhs=geometry.base.branch_polytope(x,v,b);poly,note=geometry.polytope(matrix,rhs);proof=None
    if poly is not None:
        point=poly['center']+poly['scale']*poly['interior'];proof=strict_interior(x,v,geometry.base.pattern(x,b),point)
        if not proof['accepted']:poly=None;note=dict(note,reason='unknown: exact interior failed')
    return poly,dict(note=note,proof=proof,seconds=time.perf_counter()-start)

def draw(polys,keys,count,rng):
    weights=np.array([polys[k]['volume'] for k in keys]);weights/=weights.sum();allocation=rng.multinomial(count,weights)
    points=np.concatenate([geometry.sample(polys[k],int(n),rng) for k,n in zip(keys,allocation) if n])
    return points,allocation
