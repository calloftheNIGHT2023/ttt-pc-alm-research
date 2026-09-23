"""Single-observation feasible starts, without BP or posterior-weight claims."""
from fractions import Fraction as F
from itertools import product
import hashlib
import time
import numpy as np
import forward_credit_envelope as envelope
import streaming_branch_projection as base

def pack(q):return [str(q.numerator),str(q.denominator)]
def unpack(q):return F(int(q[0]),int(q[1]))

def inverse(x,pattern,trace,target):
    current=target;bias=[None]*len(pattern)
    for j in reversed(range(len(pattern))):
        k=pattern[j];s=envelope.S[k];c=envelope.C[k];zl,zh,hl,hh=trace[j];assert hl<=current<=hh
        if s:z=(current-c)/s
        else:assert current==c;z=(zl+zh)/2
        assert zl<=z<=zh
        pl,ph=(F(float(x)),F(float(x))) if j==0 else trace[j-1][2:]
        low=max(pl,z-envelope.B);high=min(ph,z+envelope.B);assert low<=high
        previous=(low+high)/2;bias[j]=z-previous;current=previous
    assert current==F(float(x))
    return tuple(bias)

def generate(x,v,depth=4):
    begin=time.perf_counter();exact=[];lookup={};sources=[];empty=0
    for i,(xx,vv) in enumerate(zip(x,v)):
        for pattern in product(range(4),repeat=depth):
            output,trace=envelope.reachable(xx,pattern)
            if output is None:empty+=1;continue
            low=max(F(0),output[0],F(float(vv))-envelope.EPS);high=min(F(1),output[1],F(float(vv))+envelope.EPS)
            if low>high:empty+=1;continue
            target=(low+high)/2;bias=inverse(xx,pattern,trace,target)
            if bias not in lookup:lookup[bias]=len(exact);exact.append(bias)
            sources.append(dict(observation=i,pattern=list(pattern),point=lookup[bias],target=pack(target),band_intersection=[pack(low),pack(high)]))
    raw=np.array([[float(b) for b in p] for p in exact],dtype=np.float64).reshape(-1,depth);points=np.clip(raw,-.12,.12)
    # Validation is part of charged generation. Do not remove rounded endpoints.
    h=np.broadcast_to(x,(len(points),len(x))).copy()
    for j in range(depth):h=base.g(h+points[:,j,None])
    excess=[]
    for rec in sources:
        value=h[rec['point'],rec['observation']];error=abs(value-v[rec['observation']]);rec['actual_output']=float(value);rec['actual_error']=float(error)
        rec['actual_band_excess']=float(max(0.,error-base.EPS));excess.append(rec['actual_band_excess'])
    meta=dict(single_observation_patterns=len(x)*4**depth,empty_intersections=empty,sources=len(sources),exact_unique_points=len(exact),
        float_unique_points=len({p.tobytes() for p in points}),point_clipped_elements=int(np.count_nonzero(raw!=points)),
        max_float_band_excess=max(excess,default=0.),numeric_array_bytes=raw.nbytes+points.nbytes+h.nbytes,parameter_bytes=points.nbytes,
        generation_seconds=time.perf_counter()-begin,points_sha256=hashlib.sha256(points.tobytes()).hexdigest(),
        scope='One representative per nonempty closed single-observation branch; not joint feasibility or conditional-prior samples')
    records=dict(exact_points=[[pack(q) for q in p] for p in exact],sources=sources)
    return points,records,meta
