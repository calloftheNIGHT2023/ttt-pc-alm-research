"""Exact outer cones of activity adjoints, not a realizable optimizer trajectory."""
from fractions import Fraction as F
from itertools import product
import time
import numpy as np

B=F(.12)
EPS=F(.001)
ETA=F(1,1<<40)
S=(0,2,-2,0)
C=(0,0,2,0)
INTERVALS=((None,F(0)),(F(0),F(1,2)),(F(1,2),F(1)),(F(1),None))
METHODS=('ideal_forward','outward_forward','all_modes_both_signs')

def pack(q):return [str(q.numerator),str(q.denominator)]

def reachable(x,pattern,eta=F(0)):
    """Exact reachable interval in the single-observation closed-branch model."""
    lo=hi=F(float(x));trace=[]
    for branch in pattern:
        zl=lo-B-eta;zh=hi+B+eta;left,right=INTERVALS[branch]
        if left is not None:zl=max(zl,left)
        if right is not None:zh=min(zh,right)
        if zl>zh:return None,trace
        ends=[S[branch]*zl+C[branch],S[branch]*zh+C[branch]]
        lo=max(F(0),min(ends)-eta);hi=min(F(1),max(ends)+eta)
        assert lo<=hi
        trace.append((zl,zh,lo,hi))
    return (lo,hi),trace

def signs(interval,v,eta=F(0)):
    if interval is None:return ()
    lo,hi=interval;v=F(float(v));answer=[]
    if lo<v-EPS+eta:answer.append(1)
    if hi>v+EPS-eta:answer.append(-1)
    return tuple(answer)

def chain(pattern):
    c=[1]*len(pattern)
    for j in reversed(range(len(pattern)-1)):c[j]=S[pattern[j+1]]*c[j+1]
    return tuple(c)

def build(x,v,depth=4):
    assert len(x)==len(v) and all(0<=t<=1 for t in x)
    patterns=list(product(range(4),repeat=depth));banks={};metadata={};details={}
    for method in METHODS:
        start=time.perf_counter();eta=ETA if method=='outward_forward' else F(0);rows={};records=[];reachable_count=signed_count=0
        for i,(xx,vv) in enumerate(zip(x,v)):
            for pattern in patterns:
                if method=='all_modes_both_signs':output=None;sgns=(1,-1)
                else:
                    output,_=reachable(xx,pattern,eta);sgns=signs(output,vv,eta)
                    if output is not None:reachable_count+=1
                    records.append(dict(observation=i,pattern=list(pattern),output=None if output is None else [pack(q) for q in output],signs=list(sgns)))
                signed_count+=len(sgns)
                for sign in sgns:
                    for c in [chain(pattern),tuple([0]*(depth-1)+[1])]:
                        row=np.zeros((depth,len(x)));row[:,i]=np.array(c)*sign
                        row/=np.max(np.abs(row));row[row==0]=0. # Canonical positive zero.
                        rows[row.tobytes()]=row
        bank=np.array([rows[k] for k in sorted(rows)]);assert len(bank)>0
        banks[method]=bank;details[method]=records
        metadata[method]=dict(directions=len(bank),bank_bytes=bank.nbytes,reachable_single_observation_patterns=reachable_count if records else None,
            signed_patterns=signed_count,construction_seconds=time.perf_counter()-start,eta=pack(eta),scope='Single-observation cone relaxation; not a reachable joint credit or online method')
    for a,b in zip(METHODS,METHODS[1:]):assert {r.tobytes() for r in banks[a]}<={r.tobytes() for r in banks[b]},(a,b)
    return banks,metadata,details
