"""321 exact K-best branch-image-aware chain; no BP or geometry solver."""
from fractions import Fraction as F
from itertools import product
import heapq
import time
from factorized_dual_branch_search_v1 import B,EPS,S,C,ZLO,ZHI


def mask(row):return sum((1<<i) for i,r in enumerate(row) if r in [1,2])


def layer_value(low,high,previous,credit,row):
    zl=[ZLO[r] for r in row];zh=[ZHI[r] for r in row]
    left=max([-B]+[z-h for z,h in zip(zl,high)])
    right=min([B]+[z-l for z,l in zip(zh,low)])
    if left>right:return None
    sa=[S[r]*a for r,a in zip(row,credit)];coef=[p-a for p,a in zip(previous,sa)]
    knots={left,right}
    for c,z0,z1,l,h in zip(coef,zl,zh,low,high):
        knots.add(max(left,min(right,z0-l if c>=0 else z1-h)))
    values=[]
    for b in knots:
        hh=[max(l,z-b) if c>=0 else min(h,w-b) for c,l,h,z,w in zip(coef,low,high,zl,zh)]
        values.append(sum((c*h-s*b-a*C[r] for c,h,s,a,r in zip(coef,hh,sa,credit,row)),F(0)))
    return min(values)


def terminal(v,credit,output_mask):
    lo=[max(F(0),F(float(y))-EPS) for y in v]
    hi=[min(F(1) if output_mask&(1<<i) else F(0),F(float(y))+EPS) for i,y in enumerate(v)]
    if any(l>h for l,h in zip(lo,hi)):return None
    return sum((min(a*l,a*h) for a,l,h in zip(credit,lo,hi)),F(0))


def fixed_value(x,v,credit,pattern):
    n=len(x);aa=[[F(float(a)) for a in row] for row in credit];total=F(0)
    for j,row in enumerate(pattern):
        low=list(map(F,x)) if j==0 else [F(0)]*n
        high=low if j==0 else [F(int(bool(mask(pattern[j-1])&(1<<i)))) for i in range(n)]
        previous=[F(0)]*n if j==0 else aa[j-1]
        value=layer_value(low,high,previous,aa[j],row)
        if value is None:return None
        total+=value
    last=terminal(v,aa[-1],mask(pattern[-1]))
    return None if last is None else total+last


def pair_best(prefixes,rows,k):
    heap=[];out=[]
    for i,(value,path) in enumerate(prefixes):
        heap.append((value+rows[0][0],path+rows[0][1],i,0))
    heapq.heapify(heap)
    while heap and len(out)<k:
        value,path,i,j=heapq.heappop(heap);out.append((value,path))
        if j+1<len(rows):
            heapq.heappush(heap,(prefixes[i][0]+rows[j+1][0],prefixes[i][1]+rows[j+1][1],i,j+1))
    return out


def shells(x,v,credit,original,k=8):
    n=len(x);d=len(original);aa=[[F(float(a)) for a in row] for row in credit]
    allrows=list(product(range(4),repeat=n));dp={(0,-1):[(F(0),())]}
    count=0;empty=0;retained=0;peak_entries=0;pair_outputs=0;table_seconds=0.;selection_seconds=0.
    for j in range(d):
        start=time.perf_counter();transitions={}
        for incoming in sorted({m for _,m in dp}):
            lo=list(map(F,x)) if j==0 else [F(0)]*n
            hi=lo if j==0 else [F(int(bool(incoming&(1<<i)))) for i in range(n)]
            previous=[F(0)]*n if j==0 else aa[j-1];grouped={}
            for row in allrows:
                outgoing=mask(row);last=terminal(v,aa[-1],outgoing) if j==d-1 else F(0)
                if last is None:empty+=1;continue
                value=layer_value(lo,hi,previous,aa[j],row);count+=1
                if value is None:empty+=1;continue
                delta=sum(a!=int(b) for a,b in zip(row,original[j]))
                grouped.setdefault((delta,outgoing),[]).append((value+last,row))
            transitions[incoming]={key:sorted(pool)[:k] for key,pool in grouped.items()}
            retained+=sum(len(pool) for pool in transitions[incoming].values())
        table_seconds+=time.perf_counter()-start;start=time.perf_counter();nextdp={}
        for (distance,incoming),prefixes in dp.items():
            for (delta,outgoing),rows in transitions[incoming].items():
                candidates=pair_best(prefixes,rows,k);pair_outputs+=len(candidates)
                key=(distance+delta,outgoing);pool=nextdp.setdefault(key,[]);pool.extend(candidates)
                # Capping after each incoming group is exact and bounds storage.
                nextdp[key]=sorted(pool)[:k]
        dp=nextdp;peak_entries=max(peak_entries,sum(map(len,dp.values())))
        selection_seconds+=time.perf_counter()-start
    result={}
    for (m,outgoing),pool in dp.items():result.setdefault(m,[]).extend(pool)
    result={m:sorted(pool)[:k] for m,pool in result.items()}
    return result,dict(layer_rows_evaluated=count,empty_transitions=empty,retained_transition_rows=retained,
                       pair_outputs=pair_outputs,max_retained_dp_entries=peak_entries,
                       table_seconds=table_seconds,selection_seconds=selection_seconds,
                       state_scope='Exact value/DP counts, not Python/native peak memory')


def propose(x,v,credit,original,k=8):
    begin=time.perf_counter();dp,meta=shells(x,v,credit,original,k);current=fixed_value(x,v,credit,original)
    admissible=sorted(m for m,pool in dp.items() if m>=1 and pool[0][0]<=0);minimum=admissible[0] if admissible else None
    proposals=[]
    if minimum is not None:
        for m in range(minimum,min(minimum+2,len(x)*len(original))+1):
            for rank,(value,path) in enumerate(dp.get(m,[])):
                if value<=0:proposals.append(dict(mode=bytes(path).hex(),hamming=m,rank=rank,lower=str(value)))
    meta['total_seconds']=time.perf_counter()-begin
    return dict(current_lower=None if current is None else str(current),current_structurally_infeasible=current is None,
                current_certified_infeasible=current is None or current>0,minimum_nonexcluded_hamming=minimum,
                necessary_hamming_lower_bound=minimum if current is None or current>0 else None,
                shell_minima={str(m):str(pool[0][0]) for m,pool in sorted(dp.items())},proposals=proposals,meta=meta)
