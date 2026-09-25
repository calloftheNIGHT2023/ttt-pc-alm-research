"""320 exhaustive K-best checks, independent primal LP, constructive witness."""
import argparse
from collections import defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog
import factorized_dual_branch_search_v1 as search
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def forward(x,b):
    h=np.array(x,dtype=float);rows=[]
    for q in b:
        z=h+q;rows.append(((z>=0).astype(int)+(z>=.5)+(z>=1)).tolist())
        h=np.maximum(0,1-np.abs(2*z-1))
    return rows,h


def primal_lp(x,v,credit,pattern):
    d,n=np.shape(pattern);s=np.array(search.S)[pattern];c=np.array(search.C)[pattern]
    objective=np.r_[np.zeros(d),(-s*credit).ravel(),credit.ravel()]
    eq=np.zeros((d*n,d+2*d*n));rhs=np.zeros(d*n)
    for j in range(d):
        for i in range(n):
            t=j*n+i;eq[t,j]=-1;eq[t,d+t]=1
            if j:eq[t,d+d*n+(j-1)*n+i]=-1
            else:rhs[t]=x[i]
    boxes=[(-.12,.12)]*d
    boxes.extend((float(search.ZLO[r]),float(search.ZHI[r])) for r in np.array(pattern).ravel())
    boxes.extend((max(0,v[i]-.001),min(1,v[i]+.001)) if j==d-1 else (0,1)
                 for j in range(d) for i in range(n))
    result=linprog(objective,A_eq=eq,b_eq=rhs,bounds=boxes)
    return result,float(np.sum(credit*c))


def run(out):
    begin=time.perf_counter();rng=np.random.default_rng(320071);counts=defaultdict(int);gap=0.
    for d,n in [(1,2),(2,2),(3,2)]:
        for zero in [False,True]:
            x=rng.uniform(.05,.95,n);v=rng.uniform(.05,.95,n)
            a=np.zeros((d,n)) if zero else rng.normal(size=(d,n));original,_=forward(x,rng.uniform(-.12,.12,d))
            values,terminal=search.tables(x,v,a);dp,_=search.kbest_shells(values,original,k=8)
            brute=defaultdict(list)
            for path in product(range(4),repeat=d*n):
                row=[path[j*n:(j+1)*n] for j in range(d)];terms=[values[j][r] for j,r in enumerate(row)]
                if any(t is None for t in terms):continue
                dist=sum(a!=b for a,b in zip(path,np.array(original).ravel()))
                brute[dist].append((sum(terms,F(0)),path));counts['exhaustive_patterns']+=1
            assert dp=={m:sorted(pool)[:8] for m,pool in brute.items()}
            proposal=search.propose(x,v,a,original)
            for p in proposal['proposals']:
                assert F(p['lower'])<=0
                path=tuple(bytes.fromhex(p['mode']));m=p['hamming']
                assert (F(p['lower'])-terminal,path)==dp[m][p['rank']];counts['proposal_checks']+=1
            counts['exhaustive_shells']+=len(dp);counts['exhaustive_cases']+=1
    for d,n in [(1,2),(2,3),(4,4)]:
        x=rng.uniform(0,1,n);b=rng.uniform(-.12,.12,d);truth,v=forward(x,b)
        for q in range(8):
            a=rng.normal(size=(d,n));values,terminal=search.tables(x,v,a)
            pattern=truth if q==0 else rng.integers(0,4,(d,n)).tolist()
            terms=[values[j][tuple(row)] for j,row in enumerate(pattern)]
            lp,offset=primal_lp(x,v,a,pattern)
            if any(t is None for t in terms):assert lp.status==2;counts['empty_box_lp_checks']+=1
            else:
                exact=terminal+sum(terms,F(0));assert lp.success
                error=abs(float(exact)-(lp.fun-offset));gap=max(gap,error);assert error<1e-10
                if q==0:assert exact<=0;counts['feasible_not_excluded']+=1
                counts['finite_lp_checks']+=1
    # Shared-bias wrong branch: x+.04 straddles .5 and fits both observations.
    x=np.array([.45,.47]);v=np.array([.98,.98]);a=np.array([[1.,-1.]])
    witness=search.propose(x,v,a,[[1,1]])
    assert witness['current_certified_infeasible'] and witness['necessary_hamming_lower_bound']==1
    assert any(p['mode']==bytes([1,2]).hex() for p in witness['proposals'])
    result=dict(passed=True,counts=dict(counts),max_exact_lp_error=gap,witness=witness,
                source_sha256={n:sha(Path(__file__).parent/n) for n in [Path(__file__).name,'factorized_dual_branch_search_v1.py']},
                seconds=time.perf_counter()-begin,real_development_data_accessed=False,query_targets_accessed=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(args.out)
