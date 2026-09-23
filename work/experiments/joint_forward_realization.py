"""Construct shared-parameter forward probes from single-observation paths.

Only a baseline diagnostic: these BP credits never initialize the ALM learner.
No candidate-region labels, proof weights or query answers enter generation.
"""
from fractions import Fraction as F
from itertools import product
import time
import numpy as np
import forward_credit_envelope as envelope
from credit_history_capture import normalize
import streaming_branch_projection as base

METHODS=('constructed_joint','old_plus_constructed_joint')

def pack(q):return [str(q.numerator),str(q.denominator)]
def unpack(q):return F(int(q[0]),int(q[1]))

def reconstruct(x,v,pattern,sign):
    output,trace=envelope.reachable(x,pattern);assert sign in envelope.signs(output,v)
    lo,hi=output;vf=F(float(v))
    if sign==1:hi=min(hi,vf-envelope.EPS)
    else:lo=max(lo,vf+envelope.EPS)
    target=(lo+hi)/2;current=target;bias=[None]*len(pattern)
    for j in reversed(range(len(pattern))):
        k=pattern[j];s=envelope.S[k];c=envelope.C[k];zl,zh,hl,hh=trace[j];assert hl<=current<=hh
        if s:z=(current-c)/s
        else:assert current==c;z=(zl+zh)/2
        assert zl<=z<=zh
        pl,ph=(F(float(x)),F(float(x))) if j==0 else trace[j-1][2:]
        low=max(pl,z-envelope.B);high=min(ph,z+envelope.B);assert low<=high
        previous=(low+high)/2;bias[j]=z-previous;current=previous
    assert current==F(float(x));h=F(float(x))
    for b,k in zip(bias,pattern):
        assert -envelope.B<=b<=envelope.B;z=h+b;left,right=envelope.INTERVALS[k]
        assert (left is None or z>=left) and (right is None or z<=right)
        h=max(F(0),1-abs(2*z-1));assert h==envelope.S[k]*z+envelope.C[k]
    assert h==target and (h<vf-envelope.EPS if sign==1 else h>vf+envelope.EPS)
    return tuple(bias),target

def evaluate(points,x,v):
    """Same floating arithmetic as original current-activity credit evaluation."""
    r,d=points.shape;n=len(x);h=np.broadcast_to(x,(r,n));zs=[];hs=[];ss=[]
    for j in range(d):
        z=h+points[:,j,None];s=base.derivative(z);h=base.g(z);zs.append(z);ss.append(s);hs.append(h)
    raw_error=h-v;loss_residual=np.sign(raw_error)*np.maximum(np.abs(raw_error)-base.EPS,0)
    alpha=np.empty((r,d,n));alpha[:,-1]=-loss_residual
    for j in range(d-2,-1,-1):alpha[:,j]=ss[j+1]*alpha[:,j+1]
    residual=np.zeros_like(alpha);residual[:,-1]=-loss_residual
    return dict(preactivations=np.stack(zs,axis=1),activities=np.stack(hs,axis=1),slopes=np.stack(ss,axis=1),
        outputs=h.copy(),raw_error=raw_error,loss_residual=loss_residual,current_bp=alpha,output_credit=residual)

def generate(x,v,depth=4):
    start=time.perf_counter();points=[];lookup={};sources=[]
    for i,(xx,vv) in enumerate(zip(x,v)):
        for pattern in product(range(4),repeat=depth):
            output,_=envelope.reachable(xx,pattern)
            for sign in envelope.signs(output,vv):
                bias,target=reconstruct(xx,vv,pattern,sign)
                if bias not in lookup:lookup[bias]=len(points);points.append(bias)
                sources.append(dict(observation=i,pattern=list(pattern),sign=sign,point=lookup[bias],target=pack(target)))
    assert points;construction=time.perf_counter()-start
    started=time.perf_counter();raw_points=np.array([[float(b) for b in point] for point in points]);floats=np.clip(raw_points,-.12,.12);arrays=dict(points_before_clip=raw_points,points=floats,**evaluate(floats,x,v))
    forward_seconds=time.perf_counter()-started
    for rec in sources:
        point=rec['point'];i=rec['observation'];k=rec['pattern']
        rec['actual_sign_preserved']=int(np.sign(arrays['current_bp'][point,-1,i]))==rec['sign']
        rec['actual_slopes_preserved']=np.array_equal(arrays['slopes'][point,:,i],np.array(envelope.S)[k])
        rec['actual_closed_pattern_preserved']=all((envelope.INTERVALS[b][0] is None or arrays['preactivations'][point,j,i]>=float(envelope.INTERVALS[b][0])) and
            (envelope.INTERVALS[b][1] is None or arrays['preactivations'][point,j,i]<=float(envelope.INTERVALS[b][1])) for j,b in enumerate(k))
    started=time.perf_counter();raw=np.concatenate([arrays['current_bp'],arrays['output_credit']]);bank,selected,mapping,norm=normalize(raw)
    labels=np.array(['current_bp']*len(points)+['output_residual']*len(points));point_ids=np.tile(np.arange(len(points),dtype=np.int32),2)
    arrays.update(raw=raw,bank=bank,selected=selected,mapping=mapping,norm=norm,raw_labels=labels,labels=labels[selected],raw_point_ids=point_ids,point_ids=point_ids[selected])
    normalization=time.perf_counter()-started
    records=dict(exact_points=[[pack(q) for q in point] for point in points],sources=sources)
    meta=dict(sources=len(sources),exact_unique_points=len(points),float_unique_points=len({p.tobytes() for p in floats}),actual_evaluations=len(points),
        raw_directions=len(raw),directions=len(bank),raw_credit_bytes=raw.nbytes,bank_bytes=bank.nbytes,diagnostic_array_bytes=sum(t.nbytes for t in arrays.values()),
        sign_preserved=sum(r['actual_sign_preserved'] for r in sources),slopes_preserved=sum(r['actual_slopes_preserved'] for r in sources),closed_pattern_preserved=sum(r['actual_closed_pattern_preserved'] for r in sources),
        point_clipped_elements=int(np.count_nonzero(raw_points!=floats)),construction_seconds=construction,forward_seconds=forward_seconds,normalization_seconds=normalization,
        total_generation_seconds=time.perf_counter()-start,scope='All single-observation inverse probes evaluated on all observations with one shared bias; baseline diagnostic only')
    return arrays,records,meta
