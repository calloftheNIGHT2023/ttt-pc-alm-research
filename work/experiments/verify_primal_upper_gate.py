"""Fraction audit of relaxed-primal feasibility, enclosures and safe skips."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import primal_upper_gate as gate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_witnesses(x,v,reg):
    zl,zh,hl,hh=gate.original.screen.boxes(v,reg[None]);zl,zh,hl,hh=[a[0] for a in [zl,zh,hl,hh]]
    d,n=reg.shape;answer=[]
    for choice in range(3):
        b=[];z=[];h=[[F(0)]*n for _ in range(d)]
        h[-1]=[(F(float(a))+F(float(c)))/2 for a,c in zip(hl[-1],hh[-1])]
        if any(F(float(a))>F(float(c)) for a,c in zip(hl[-1],hh[-1])):return None
        for j in range(d):
            pl=[F(float(t)) for t in (x if j==0 else hl[j-1])];ph=[F(float(t)) for t in (x if j==0 else hh[j-1])]
            lower=max([-F(.12)]+[F(float(a))-c for a,c in zip(zl[j],ph)])
            upper=min([F(.12)]+[F(float(a))-c for a,c in zip(zh[j],pl)])
            if lower>upper:return None
            bj=[lower,(lower+upper)/2,upper][choice];b.append(bj)
            left=[max(a,F(float(t))-bj) for a,t in zip(pl,zl[j])];right=[min(a,F(float(t))-bj) for a,t in zip(ph,zh[j])]
            assert all(a<=c for a,c in zip(left,right))
            prev=pl if j==0 else [(a+c)/2 for a,c in zip(left,right)]
            if j:h[j-1]=prev
            z.append([a+bj for a in prev])
        for j in range(d):
            assert -F(.12)<=b[j]<=F(.12)
            for i in range(n):
                assert F(float(zl[j,i]))<=z[j][i]<=F(float(zh[j,i]))
                assert F(float(hl[j,i]))<=h[j][i]<=F(float(hh[j,i]))
                assert z[j][i]==(F(float(x[i])) if j==0 else h[j-1][i])+b[j]
        residual=[[h[j][i]-int(gate.original.base.SLOPES[reg[j,i]])*z[j][i]-int(gate.original.base.INTERCEPTS[reg[j,i]]) for i in range(n)] for j in range(d)]
        answer.append(dict(b=b,z=z,h=h,residual=residual))
    return answer


def verify():
    rng=np.random.default_rng(483201);checks=dict(regions=0,feasible_relaxations=0,empty_relaxations=0,witnesses=0,
        scalar_interval_enclosures=0,exact_direction_bounds=0,strict_skip_rows=0,skipped_directions_checked=0)
    for d,n in [(1,2),(2,3),(4,4),(4,8),(4,24),(6,5)]:
        x=rng.uniform(0,1,n);v=rng.uniform(0,1,n)
        bs=rng.uniform(-.12,.12,(12,d));regs=np.r_[np.array([gate.original.base.pattern(x,b) for b in bs],np.uint8),rng.integers(0,4,(12,d,n),dtype=np.uint8)]
        bank=rng.normal(size=(7,d,n));ww=gate.witnesses(x,v,regs);skip,upper,_=gate.gate(x,v,regs,bank)
        for r,reg in enumerate(regs):
            exact=exact_witnesses(x,v,reg);checks['regions']+=1
            if exact is None:
                assert not ww['valid'][r] and not skip[r];checks['empty_relaxations']+=1;continue
            checks['feasible_relaxations']+=1
            for w,record in enumerate(exact):
                checks['witnesses']+=1
                for name in ['b','h','z','residual']:
                    arr=np.asarray(record[name],dtype=object);lo=ww[name][0][w,r];hi=ww[name][1][w,r]
                    for value,left,right in zip(arr.flat,lo.flat,hi.flat):
                        assert F(float(left))<=value<=F(float(right)),(name,d,n,r)
                        checks['scalar_interval_enclosures']+=1
            for k,a in enumerate(bank):
                objectives=[sum(F(float(a[j,i]))*w['residual'][j][i] for j in range(d) for i in range(n)) for w in exact]
                target=min(objectives);opt=gate.original.exact_optimum(x,v,reg,a)
                assert 'empty_layer' not in opt
                optimum=F(int(opt['numerator']),int(opt['denominator']));assert optimum<=target;checks['exact_direction_bounds']+=1
                if np.isfinite(upper[r,k]):assert target<=F(float(upper[r,k]))
                if skip[r]:assert target<=0 and not opt['positive'];checks['skipped_directions_checked']+=1
            checks['strict_skip_rows']+=int(skip[r])
        # Construct a non-vacuous negative-credit case from each feasible witness.
        for r,reg in enumerate(regs):
            exact=exact_witnesses(x,v,reg)
            if exact is None or not ww['valid'][r]:continue
            res=np.array(exact[1]['residual'],float)
            if np.linalg.norm(res)<1e-9:continue
            positive_bank=np.stack([-res,-2*res,-.5*res]);ss,uu,_=gate.gate(x,v,reg[None],positive_bank)
            assert ss[0],(d,n,r)
            for a in positive_bank:assert not gate.original.exact_optimum(x,v,reg,a)['positive']
            checks['strict_skip_rows']+=1;checks['skipped_directions_checked']+=3;break
    return dict(passed=True,checks=checks,scope='Generic rational feasibility and safe nonpositive credit upper bounds; no speed claim')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/context_block_scaling/reachability/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    result=verify()
    for path in [Path(__file__),Path(gate.__file__)]:hashes[path.name]=sha(path)
    out=root/'results/primal_upper_gate/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    result.update(source_sha256=hashes,parent_protocol_sha256=sha(parent))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=result['checks'],sources=len(hashes))),flush=True)


if __name__=='__main__':main()
