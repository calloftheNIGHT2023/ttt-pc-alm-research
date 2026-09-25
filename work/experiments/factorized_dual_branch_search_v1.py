"""320 exact K-best fixed-credit repair of whole-network branch patterns.

No geometry, global parameter gradient, query target, or optimization library.
The selector ranks necessary lower bounds, not posterior mass or risk.
"""
from fractions import Fraction as F
from itertools import product
import time

B=F(.12)
EPS=F(.001)
S=(0,2,-2,0)
C=(0,0,2,0)
ZLO=(-B,F(0),F(1,2),F(1))
ZHI=(F(0),F(1,2),F(1),1+B)


def row_value(x, previous_credit, credit, row):
    """Exact scalar shared-bias minimization; None denotes an empty box."""
    n=len(row)
    lo=list(map(F,x)) if x is not None else [F(0)]*n
    hi=lo if x is not None else [F(1)]*n
    sa=[S[r]*a for r,a in zip(row,credit)]
    coef=[p-a for p,a in zip(previous_credit,sa)]
    zl=[ZLO[r] for r in row];zh=[ZHI[r] for r in row]
    left=max([-B]+[z-h for z,h in zip(zl,hi)])
    right=min([B]+[z-h for z,h in zip(zh,lo)])
    if left>right:return None
    knots={left,right}
    for z0,z1,h0,h1,c in zip(zl,zh,lo,hi,coef):
        knots.add(max(left,min(right,z0-h0 if c>=0 else z1-h1)))
    values=[]
    for bias in knots:
        h=[max(h0,z0-bias) if c>=0 else min(h1,z1-bias)
           for h0,h1,z0,z1,c in zip(lo,hi,zl,zh,coef)]
        values.append(sum((c*hh-s*bias for c,hh,s in zip(coef,h,sa)),F(0)))
    return min(values)-sum((a*C[r] for a,r in zip(credit,row)),F(0))


def tables(x,v,credit):
    aa=[[F(float(q)) for q in row] for row in credit]
    d,n=len(aa),len(x)
    assert all(len(row)==n for row in aa) and len(v)==n
    assert all(0<=F(float(q))<=1 for q in x)
    output_lo=[max(F(0),F(float(q))-EPS) for q in v]
    output_hi=[min(F(1),F(float(q))+EPS) for q in v]
    assert all(lo<=hi for lo,hi in zip(output_lo,output_hi))
    terminal=sum((min(a*lo,a*hi) for a,lo,hi in zip(aa[-1],output_lo,output_hi)),F(0))
    rows=list(product(range(4),repeat=n));values=[]
    for j in range(d):
        previous=[F(0)]*n if j==0 else aa[j-1]
        values.append({r:row_value(x if j==0 else None,previous,aa[j],r) for r in rows})
    return values,terminal


def kbest_shells(layer_values,original,k=8):
    """Exact per-Hamming-shell K-best sum (cost, flattened pattern)."""
    assert k>=1 and len(layer_values)==len(original)
    dp={0:[(F(0),())]};combinations=0;row_counts=[]
    for values,old in zip(layer_values,original):
        grouped={}
        for row,value in values.items():
            if value is None:continue
            distance=sum(a!=int(b) for a,b in zip(row,old))
            grouped.setdefault(distance,[]).append((value,row))
        for distance,candidates in grouped.items():grouped[distance]=sorted(candidates)[:k]
        row_counts.append(sum(map(len,grouped.values())))
        candidates={}
        for old_distance,prefixes in dp.items():
            for delta,rows in grouped.items():
                pool=candidates.setdefault(old_distance+delta,[])
                for before,path in prefixes:
                    for value,row in rows:
                        pool.append((before+value,path+row));combinations+=1
        dp={m:sorted(pool)[:k] for m,pool in candidates.items()}
    return dp,dict(dp_combinations=combinations,retained_layer_rows=row_counts)


def propose(x,v,credit,original,k=8):
    begin=time.perf_counter();values,terminal=tables(x,v,credit)
    table_seconds=time.perf_counter()-begin
    start=time.perf_counter();shells,meta=kbest_shells(values,original,k)
    current_terms=[values[j][tuple(map(int,row))] for j,row in enumerate(original)]
    assert all(t is not None for t in current_terms), 'A true forward pattern must have nonempty shared-bias rows'
    current=terminal+sum(current_terms,F(0))
    admissible=[m for m,pool in sorted(shells.items()) if m>=1 and pool[0][0]+terminal<=0]
    minimum=admissible[0] if admissible else None
    proposals=[]
    if minimum is not None:
        for m in range(minimum,min(minimum+2,len(original)*len(x))+1):
            for rank,(value,path) in enumerate(shells.get(m,[])):
                if value+terminal<=0:
                    proposals.append(dict(mode=bytes(path).hex(),hamming=m,rank=rank,lower=str(value+terminal)))
    meta.update(table_seconds=table_seconds,selection_seconds=time.perf_counter()-start,
                total_seconds=time.perf_counter()-begin,layer_rows_evaluated=sum(map(len,values)),
                empty_layer_rows=sum(v is None for row in values for v in row.values()),
                rational_values_stored=sum(v is not None for row in values for v in row.values()),
                state_scope='Counts of exact table values and DP combinations; not Python/native peak bytes')
    return dict(current_lower=str(current),current_certified_infeasible=current>0,
                minimum_nonexcluded_hamming=minimum,
                necessary_hamming_lower_bound=minimum if current>0 else None,
                shell_minima={str(m):str(pool[0][0]+terminal) for m,pool in sorted(shells.items())},
                proposals=proposals,meta=meta)
