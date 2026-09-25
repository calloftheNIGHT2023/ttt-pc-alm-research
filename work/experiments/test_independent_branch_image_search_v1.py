"""Additional independent exact enumeration and lifted-primal LP checks."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog
import branch_image_chain_v1 as candidate
from independent_branch_image_search_v1 import IndependentSearch
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def forward(x,b):
    out=x.copy();pattern=[]
    for bias in b:
        z=out+bias;pattern.append(np.searchsorted([0.,.5,1.],z,side='right'))
        out=np.maximum(0,1-np.abs(2*z-1))
    return np.array(pattern),out


def lp_value(x,v,credit,pattern):
    d,n=pattern.shape;s=np.array([0,2,-2,0])[pattern];c=np.array([0,0,2,0])[pattern]
    aeq=np.zeros((d*n,d+2*d*n));rhs=np.zeros(d*n)
    for j in range(d):
        for i in range(n):
            t=j*n+i;aeq[t,j]=-1;aeq[t,d+t]=1
            if j:aeq[t,d+d*n+(j-1)*n+i]=-1
            else:rhs[t]=x[i]
    zlo=[-.12,0,.5,1];zhi=[0,.5,1,1.12];bounds=[(-.12,.12)]*d
    bounds.extend((zlo[r],zhi[r]) for r in pattern.ravel())
    for j,row in enumerate(pattern):
        for i,label in enumerate(row):
            low=max(0,v[i]-.001) if j==d-1 else 0.
            high=min(float(label in [1,2]),v[i]+.001) if j==d-1 else float(label in [1,2])
            if low>high:return None
            bounds.append((low,high))
    result=linprog(np.r_[np.zeros(d),(-s*credit).ravel(),credit.ravel()],A_eq=aeq,b_eq=rhs,bounds=bounds)
    if result.status==2:return None
    assert result.success;return result.fun-float(np.sum(c*credit))


def run(out):
    start=time.perf_counter();rng=np.random.default_rng(3219901);counts=Counter();largest=0.
    for d,n in [(2,2),(3,2),(2,3)]:
        for zero in [False,True]:
            x=rng.uniform(0,1,n);pattern,v=forward(x,rng.uniform(-.12,.12,d));a=np.zeros((d,n)) if zero else rng.normal(size=(d,n))
            checker=IndependentSearch(x,v,a,pattern);dp,_=candidate.shells(x,v,a,pattern);brute=defaultdict(list)
            for path in product(range(4),repeat=d*n):
                rows=[path[j*n:(j+1)*n] for j in range(d)];value=checker.fixed(rows);counts['enumerated_patterns']+=1
                expected=candidate.fixed_value(x,v,a,rows);assert value==expected
                if value is not None:brute[sum(q!=r for q,r in zip(path,pattern.ravel()))].append((value,path))
            expected={m:sorted(pool)[:8] for m,pool in brute.items()};assert expected==dp
            for m,pool in expected.items():assert checker.kbest(m)==pool;counts['best_first_shells']+=1
            checker.clear();counts['exhaustive_cases']+=1
    for d,n in [(2,3),(4,4)]:
        for _ in range(12):
            x=rng.uniform(0,1,n);p,v=forward(x,rng.uniform(-.12,.12,d));a=rng.normal(size=(d,n))
            original=p.copy();checker=IndependentSearch(x,v,a,p)
            variants=[original,rng.integers(0,4,(d,n))]
            for changes in [1,2]:
                pool=checker.kbest(changes,k=1)
                if pool:variants.append(np.array(pool[0][1]).reshape(d,n))
            for pattern in variants:
                value=checker.fixed(pattern);lp=lp_value(x,v,a,pattern)
                assert (value is None)==(lp is None)
                if value is not None:
                    gap=abs(float(value)-lp);largest=max(largest,gap);assert gap<1e-9;counts['finite_primal_lp']+=1
                else:counts['empty_primal_lp']+=1
            checker.clear()
    result=dict(passed=True,counts=dict(counts),max_fraction_lp_gap=largest,seconds=time.perf_counter()-start,
                source_sha256={n:sha(Path(__file__).parent/n) for n in [Path(__file__).name,'independent_branch_image_search_v1.py','branch_image_chain_v1.py']},
                real_development_data_accessed=False,query_targets_accessed=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(args.out)
