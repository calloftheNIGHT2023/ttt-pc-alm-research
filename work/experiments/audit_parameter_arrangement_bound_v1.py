"""369 exact symbolic enumeration audit; supporting checks, not the general proof."""
from fractions import Fraction as F
from itertools import product
from math import gcd,comb
from functools import reduce
from pathlib import Path
import time
from budget_reinvestment_suite_v1 import save,sha


def canonical(coeff,rhs):
    den=rhs.denominator;row=[int(c)*den for c in coeff]+[rhs.numerator]
    divisor=reduce(gcd,map(abs,row));row=[v//divisor for v in row]
    if next(v for v in row[:-1] if v)!=abs(next(v for v in row[:-1] if v)):
        row=[-v for v in row]
    return tuple(row)


def merged(x,d):
    forms={(tuple([0]*d),x)};planes=set();counts=[]
    for layer in range(d):
        nextforms={(tuple([0]*d),F(0))}
        for coeff,c in forms:
            z=list(coeff);z[layer]+=1
            assert z[layer]==1 and not any(z[layer+1:])
            for edge in [F(0),F(1,2),F(1)]:planes.add(canonical(z,edge-c))
            for s,o in [(2,0),(-2,2)]:nextforms.add((tuple(s*t for t in z),s*c+o))
        counts.append(len(forms));forms=nextforms
    return planes,counts,forms


def exhaustive(x,d):
    planes=set();last=set()
    for path in product(range(4),repeat=d):
        coeff=[0]*d;c=x
        for layer,branch in enumerate(path):
            coeff[layer]+=1
            for edge in [F(0),F(1,2),F(1)]:planes.add(canonical(coeff,edge-c))
            s=[0,2,-2,0][branch];o=[0,0,2,0][branch]
            coeff=[s*t for t in coeff];c=s*c+o
        last.add((tuple(coeff),c))
    return planes,last


def main():
    begin=time.perf_counter();root=Path(__file__).resolve().parents[2]
    out=root/'results/parameter_arrangement_bound/audit_v1';out.mkdir(parents=True,exist_ok=False)
    rows=[];checks=0
    for d in range(1,6):
        allplanes=set()
        for x in [F(1,10),F(3,10),F(7,10),F(9,10)]:
            planes,counts,forms=merged(x,d);reference,last=exhaustive(x,d)
            assert planes==reference and forms==last
            assert all(c<=2**(j+1)-1 for j,c in enumerate(counts))
            assert len(planes)<=3*(2**(d+1)-d-2)
            allplanes|=planes;checks+=len(planes)+len(forms)
            rows.append(dict(d=d,x=str(x),preactivation_form_counts=counts,unique_hyperplanes=len(planes),output_forms=len(forms)))
        assert len(allplanes)<=12*(2**(d+1)-d-2)
    for d in range(1,9):
        for h in range(1,50):
            assert sum(comb(h,j) for j in range(d+1) if j<=h)==sum(comb(h-1,j) for j in range(d+1) if j<=h-1)+sum(comb(h-1,j) for j in range(d) if j<=h-1)
            checks+=1
    save(out/'rows.json',rows)
    result=dict(passed=True,symbolic_cases=len(rows),symbolic_and_recurrence_checks=checks,seconds=time.perf_counter()-begin,
        query_targets_accessed=False,source_sha256=sha(Path(__file__)),
        appendix_sha256=sha(root/'outputs/ttt-pc-alm-research/369_parameter_arrangement_bound_appendix_v1.md'),
        scope='Exact finite symbolic checks supporting the written general proof; no runtime, sampling or PC superiority theorem',
        outputs_sha256={'rows.json':sha(out/'rows.json')})
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':main()
