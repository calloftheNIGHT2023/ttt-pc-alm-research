"""Exact 1D spline compilation for the toy family's weighted function bank.

This is an implementation control, NOT a new learning method. It prevents an
unoptimized repeated feature evaluation from masquerading as a required cost.
"""
import numpy as np
import streaming_branch_projection as base


def segments(b):
    knots=np.array([0.,1.]); slopes=np.array([1.]); offsets=np.array([0.])
    for bias in b:
        additions=[]
        for k in [0.,.5,1.]:
            crossing=np.divide(k-offsets-bias,slopes,out=np.full_like(slopes,np.nan),where=slopes!=0)
            valid=(crossing>knots[:-1])&(crossing<knots[1:])
            additions.extend(crossing[valid].tolist())
        new=np.unique(np.r_[knots,additions]); mid=(new[:-1]+new[1:])/2
        old=np.searchsorted(knots,mid,side="right")-1
        z=slopes[old]*mid+offsets[old]+bias
        region=np.searchsorted(base.KNOTS,z,side="right")
        offsets=base.SLOPES[region]*(offsets[old]+bias)+base.INTERCEPTS[region]
        slopes=base.SLOPES[region]*slopes[old]; knots=new
    return knots,slopes,offsets


def prepare(bank):
    knots=[]; jumps=[]; owners=[]; initial=[]; values=[]
    for i,b in enumerate(bank):
        k,s,c=segments(b)
        knots.extend(k[1:-1]); jumps.extend(np.diff(s)); owners.extend([i]*(len(s)-1))
        initial.append(s[0]); values.append(c[0])
    order=np.argsort(knots,kind="stable")
    return {"knots":np.asarray(knots)[order],"jumps":np.asarray(jumps)[order],"owners":np.asarray(owners,dtype=np.int64)[order],
        "initial_slopes":np.asarray(initial),"initial_values":np.asarray(values)}


def compile_weights(cache,weights):
    # Long-double prefix summation reduces cancellation in signed ridge weights.
    k=cache["knots"]; changes=cache["jumps"].astype(np.longdouble)*weights[cache["owners"]].astype(np.longdouble)
    unique,starts=np.unique(k,return_index=True)
    summed=np.add.reduceat(changes,starts) if len(starts) else np.empty(0,dtype=np.longdouble)
    first_slope=np.dot(weights.astype(np.longdouble),cache["initial_slopes"].astype(np.longdouble))
    slopes=np.r_[first_slope,first_slope+np.cumsum(summed)]
    knots=np.r_[0.,unique].astype(np.longdouble)
    first_value=np.dot(weights.astype(np.longdouble),cache["initial_values"].astype(np.longdouble))
    values=np.r_[first_value,first_value+np.cumsum(slopes[:-1]*np.diff(knots))]
    def predict(q):
        q=np.asarray(q,dtype=np.longdouble); assert np.all((q>=0)&(q<=1))
        index=np.searchsorted(knots,q,side="right")-1
        return np.asarray(values[index]+slopes[index]*(q-knots[index]),dtype=float)
    return predict,{"compiled_breakpoints":len(knots),"persistent_predictor_bytes":knots.nbytes+slopes.nbytes+values.nbytes}


def verify():
    rng=np.random.default_rng(6831); errors=[]
    for count in [1,16,128]:
        bank=rng.uniform(-.12,.12,(count,4)); w=rng.normal(size=count); w/=np.abs(w).sum()
        cache=prepare(bank); predict,_=compile_weights(cache,w)
        q=np.r_[np.linspace(0,1,3001),rng.uniform(0,1,200)]
        expected=w@np.stack([base.forward(q,b) for b in bank]); err=float(np.max(np.abs(expected-predict(q))))
        assert err<1e-11,err; errors.append(err)
    return {"passed":True,"max_absolute_errors":errors,"claim":"identical predictor up to floating-point arithmetic, no fitted query labels"}


if __name__=="__main__":
    import json
    print(json.dumps(verify()))
