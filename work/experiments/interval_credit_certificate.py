"""Outward-rounded enclosure of the exact binary-rational layer bound.

Standalone next primitive. Not used by the frozen round194 benchmark.
No unchecked float value can authorize a rejection. Each NumPy float64
arithmetic operation is expanded by nextafter; inputs are exact floats.
"""
import numpy as np
import optimized_branch_dual as original


def point(a):
    a=np.asarray(a,dtype=np.float64)
    return a,a


def down(a):return np.nextafter(a,-np.inf)
def up(a):return np.nextafter(a,np.inf)
def add(a,b):return down(a[0]+b[0]),up(a[1]+b[1])
def sub(a,b):return down(a[0]-b[1]),up(a[1]-b[0])
def minimum(a,b):return np.minimum(a[0],b[0]),np.minimum(a[1],b[1])
def maximum(a,b):return np.maximum(a[0],b[0]),np.maximum(a[1],b[1])


def mul(a,b):
    values=np.stack(np.broadcast_arrays(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]))
    return down(values.min(0)),up(values.max(0))


def sum_last(a):
    total=point(np.zeros(a[0].shape[:-1]))
    for i in range(a[0].shape[-1]):total=add(total,(a[0][...,i],a[1][...,i]))
    return total


def reduce_min(a,axis):return a[0].min(axis),a[1].min(axis)
def reduce_max(a,axis):return a[0].max(axis),a[1].max(axis)
def expand(a,axis):return np.expand_dims(a[0],axis),np.expand_dims(a[1],axis)


def enclose(x,v,regs,a):
    """One credit per region. Returns objective intervals or empty-box gaps.

    If an empty shared-bias interval is proved, the gap is a separate
    infeasibility witness, not a value of the dual objective.
    """
    assert regs.shape==a.shape and np.all(np.isfinite(a))
    rows,d,n=regs.shape
    zl,zh,hl,hh=original.screen.boxes(v,regs)
    sa=mul(point(original.base.SLOPES[regs]),point(a))
    total=sum_last(minimum(mul(point(a[:,-1]),point(hl[:,-1])),mul(point(a[:,-1]),point(hh[:,-1]))))
    constant=mul(point(a),point(original.base.INTERCEPTS[regs]))
    total=sub(total,sum_last((constant[0].reshape(rows,-1),constant[1].reshape(rows,-1))))
    empty=np.zeros(rows,bool);gap_lower=np.zeros(rows);gap_upper=np.zeros(rows);uncertain_empty=np.zeros(rows,bool)
    for j in range(d):
        pl=np.broadcast_to(x,(rows,n)) if j==0 else hl[:,j-1]
        ph=np.broadcast_to(x,(rows,n)) if j==0 else hh[:,j-1]
        previous=np.zeros((rows,n)) if j==0 else a[:,j-1]
        saj=(sa[0][:,j],sa[1][:,j]);c=sub(point(previous),saj)
        lower_kinks=sub(point(zl[:,j]),point(pl));upper_kinks=sub(point(zh[:,j]),point(ph))
        low=maximum(point(-original.screen.B),reduce_max(sub(point(zl[:,j]),point(ph)),1))
        high=minimum(point(original.screen.B),reduce_min(sub(point(zh[:,j]),point(pl)),1))
        gap=sub(low,high);proved=gap[0]>0
        new=proved&~empty;gap_lower[new]=gap[0][new];gap_upper[new]=gap[1][new];empty|=proved
        uncertain_empty|=(low[1]>high[0])&~proved
        # Both sets of possible kinks are included, so uncertain coefficient
        # signs cannot silently discard an exact minimizer.
        candidates=(np.concatenate([low[0][:,None],high[0][:,None],lower_kinks[0],upper_kinks[0]],1),
            np.concatenate([low[1][:,None],high[1][:,None],lower_kinks[1],upper_kinks[1]],1))
        b=minimum(maximum(candidates,expand(low,1)),expand(high,1));b=expand(b,2)
        lo=maximum(point(pl[:,None]),sub(point(zl[:,j,None]),b))
        hi=minimum(point(ph[:,None]),sub(point(zh[:,j,None]),b))
        term=sub(minimum(mul(expand(c,1),lo),mul(expand(c,1),hi)),mul(expand(saj,1),b))
        total=add(total,reduce_min(sum_last(term),1))
    lower,upper=total
    # Ambiguous feasibility must fall back to an exact solver. Empty gaps
    # already proved positive remain valid regardless of other layers.
    lower=np.where(uncertain_empty&~empty,-np.inf,lower)
    upper=np.where(uncertain_empty&~empty,np.inf,upper)
    lower=np.where(empty,gap_lower,lower);upper=np.where(empty,gap_upper,upper)
    return dict(lower=lower,upper=upper,empty_shared_bias=empty,positive=lower>0)


def verify():
    from fractions import Fraction
    rng=np.random.default_rng(482301);checks=0;objective=0;empty=0
    for d,n in [(1,2),(2,3),(4,4),(6,5)]:
        x=rng.uniform(0,1,n);v=rng.uniform(0,1,n)
        biases=rng.uniform(-.12,.12,(20,d))
        structured=np.array([original.base.pattern(x,b) for b in biases],np.uint8)
        regs=np.r_[structured,rng.integers(0,4,(20,d,n),dtype=np.uint8)];a=rng.normal(size=regs.shape)
        result=enclose(x,v,regs,a)
        for i in range(len(regs)):
            exact=original.exact_optimum(x,v,regs[i],a[i])
            if result['empty_shared_bias'][i]:assert 'empty_layer' in exact;empty+=1
            elif 'empty_layer' not in exact:
                value=Fraction(int(exact['numerator']),int(exact['denominator']))
                lo=result['lower'][i];hi=result['upper'][i]
                assert (not np.isfinite(lo) or Fraction(float(lo))<=value) and (not np.isfinite(hi) or value<=Fraction(float(hi)))
                objective+=1
            if result['positive'][i]:assert exact['positive']
            checks+=1
    return dict(passed=True,exact_cases=checks,finite_objective_enclosures=objective,empty_box_witnesses=empty,
        scope='generic interval primitive; not yet deployed, not benchmark timing')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
