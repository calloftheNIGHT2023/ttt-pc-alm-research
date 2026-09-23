"""Exact midpoint-bias restricted primal and fixed-tree interval audit."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import directional_primal_upper_gate as gate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_upper(x,v,reg,a):
    zl,zh,hl,hh=gate.original.screen.boxes(v,reg[None]);zl,zh,hl,hh=[a[0] for a in [zl,zh,hl,hh]]
    d,n=reg.shape;aa=[[F(float(t)) for t in row] for row in a]
    total=sum(min(aa[-1][i]*F(float(hl[-1,i])),aa[-1][i]*F(float(hh[-1,i]))) for i in range(n))
    total-=sum(aa[j][i]*int(gate.original.base.INTERCEPTS[reg[j,i]]) for j in range(d) for i in range(n))
    if np.any(hl>hh):return None
    for j in range(d):
        pl=[F(float(t)) for t in (x if j==0 else hl[j-1])];ph=[F(float(t)) for t in (x if j==0 else hh[j-1])]
        low=max([-F(.12)]+[F(float(zl[j,i]))-ph[i] for i in range(n)])
        high=min([F(.12)]+[F(float(zh[j,i]))-pl[i] for i in range(n)])
        if low>high:return None
        b=(low+high)/2
        for i in range(n):
            lo=max(pl[i],F(float(zl[j,i]))-b);hi=min(ph[i],F(float(zh[j,i]))-b)
            assert lo<=hi
            sa=int(gate.original.base.SLOPES[reg[j,i]])*aa[j][i]
            c=(F(0) if j==0 else aa[j-1][i])-sa
            total+=min(c*lo,c*hi)-sa*b
    return total


def verify():
    rng=np.random.default_rng(483223);checks=dict(regions=0,directions=0,finite_upper=0,empty_domains=0,strict_gate_rows=0,pair_sum_intervals=0)
    for width in [1,2,3,4,7,16,17,31,96]:
        a=rng.normal(size=(9,width));lo,hi=gate.pair_sum(gate.iv.point(a))
        for row,l,h in zip(a,lo,hi):assert F(float(l))<=sum(F(float(x)) for x in row)<=F(float(h));checks['pair_sum_intervals']+=1
    for d,n in [(1,2),(2,3),(4,4),(4,8),(4,24),(6,5)]:
        x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);bs=rng.uniform(-.12,.12,(16,d))
        regs=np.r_[np.array([gate.original.base.pattern(x,b) for b in bs],np.uint8),rng.integers(0,4,(16,d,n),dtype=np.uint8)]
        bank=rng.normal(size=(9,d,n));lower,upper,valid=gate.strict_upper(x,v,regs,bank);skip,selected,_=gate.gate(x,v,regs,bank)
        for r,reg in enumerate(regs):
            checks['regions']+=1;checks['strict_gate_rows']+=int(skip[r])
            for k,a in enumerate(bank):
                target=exact_upper(x,v,reg,a);checks['directions']+=1
                if target is None:assert not valid[r] and not skip[r];checks['empty_domains']+=1;continue
                if valid[r]:assert F(float(lower[r,k]))<=target<=F(float(upper[r,k]));checks['finite_upper']+=1
                optimum=gate.original.exact_optimum(x,v,reg,a);assert 'empty_layer' not in optimum
                value=F(int(optimum['numerator']),int(optimum['denominator']));assert value<=target
                if skip[r]:assert target<=0 and not optimum['positive']
    return dict(passed=True,checks=checks,scope='Directional activity optimum at exact midpoint bias upper-bounds full certificate optimum')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/primal_upper_gate/component/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    result=verify()
    for path in [Path(__file__),Path(gate.__file__)]:hashes[path.name]=sha(path)
    result.update(source_sha256=hashes,parent_protocol_sha256=sha(parent))
    out=root/'results/directional_primal_gate/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=result['checks'],sources=len(hashes))),flush=True)


if __name__=='__main__':main()
