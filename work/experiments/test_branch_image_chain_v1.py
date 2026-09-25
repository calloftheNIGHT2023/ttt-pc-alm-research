"""321 full enumeration, old-bound domination and feasible-point tests."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import time
import numpy as np
import branch_image_chain_v1 as chain
import factorized_dual_branch_search_v1 as old
from test_factorized_dual_branch_search_v1 import forward
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def run(out):
    begin=time.perf_counter();counts=Counter();rng=np.random.default_rng(321071)
    for d,n in [(1,2),(2,2),(3,1)]:
        for zero in [False,True]:
            x=rng.uniform(0,1,n);v=rng.uniform(.05,.95,n);a=np.zeros((d,n)) if zero else rng.normal(size=(d,n))
            original,_=forward(x,rng.uniform(-.12,.12,d));dp,_=chain.shells(x,v,a,original)
            layer,term=old.tables(x,v,a);brute=defaultdict(list)
            for path in product(range(4),repeat=d*n):
                pattern=[path[j*n:(j+1)*n] for j in range(d)];value=chain.fixed_value(x,v,a,pattern)
                if value is None:counts['excluded_patterns']+=1;continue
                terms=[layer[j][row] for j,row in enumerate(pattern)];assert all(q is not None for q in terms)
                assert value>=term+sum(terms,F(0));counts['bound_domination']+=1
                distance=sum(q!=r for q,r in zip(path,np.array(original).ravel()))
                brute[distance].append((value,path))
            assert dp=={m:sorted(pool)[:8] for m,pool in brute.items()};counts['exhaustive_cases']+=1
    for _ in range(24):
        x=rng.uniform(0,1,4);b=rng.uniform(-.12,.12,4);pattern,v=forward(x,b);a=rng.normal(size=(4,4))
        value=chain.fixed_value(x,v,a,pattern);assert value is not None and value<=0;counts['feasible_not_excluded']+=1
    # Strict structural strengthening at zero credit.
    x=np.array([.1]);v=np.array([.5]);a=np.zeros((2,1));pattern=[[0],[2]]
    table,term=old.tables(x,v,a);assert term+sum((table[j][tuple(r)] for j,r in enumerate(pattern)),F(0))==0
    assert chain.fixed_value(x,v,a,pattern) is None;counts['strict_counterexample']+=1
    # All-active pattern has no additional internal image restrictions.
    x=np.array([.45]);v=np.array([.04]);a=np.array([[.3],[-.2]]);pattern=[[1],[2]]
    table,term=old.tables(x,v,a);assert chain.fixed_value(x,v,a,pattern)==term+sum((table[j][tuple(r)] for j,r in enumerate(pattern)),F(0))
    counts['equality_case']+=1
    result=dict(passed=True,counts=dict(counts),seconds=time.perf_counter()-begin,
                source_sha256={n:sha(Path(__file__).parent/n) for n in [Path(__file__).name,'branch_image_chain_v1.py','factorized_dual_branch_search_v1.py']},
                real_development_data_accessed=False,query_targets_accessed=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(args.out)
