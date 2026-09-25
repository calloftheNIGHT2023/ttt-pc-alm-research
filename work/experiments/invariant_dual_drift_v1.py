"""317 algebraic invariant-line proposal; no guard or fast-forward claim."""
from fractions import Fraction as F
from affine_root_certificate_v1 import system,solve
from effective_affine_map_v1 import evaluate


def propose_line(rows,point,dual_start):
    n=len(rows);assert 0<=dual_start<n
    a,c=system(rows);dual=list(range(dual_start,n));k=len(dual)
    top=[row+[F(i==j) for j in dual] for i,row in enumerate(a)]
    bottom=[[F(0)]*n+[row[j] for j in dual] for row in a]
    result=solve(top+bottom,c+[F(0)]*n,list(point)+[F(0)]*k)
    result['dual_start']=dual_start
    if result['consistent']:
        q=result['root'][:n];d=[F(0)]*dual_start+result['root'][n:]
        assert [z-v for z,v in zip(evaluate(rows,q),q)]==d
        for i,row in enumerate(rows):
            assert sum((coef*d[j] for j,coef in row.terms.items() if j>=0),F(0))==d[i]
        result.update(q=q,d=d,nonzero_drift=any(d))
    return result
