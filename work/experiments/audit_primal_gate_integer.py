"""Independent common-denominator integer audit of all frozen gate rows.

The integer computation contains no floating arithmetic after boxes and
stored credits are decoded as exact binary rationals. It evaluates the
midpoint-bias restricted primal, not the implemented error-bound formula.
"""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import verify_directional_primal_gate as reference


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def exponent(a):
    return max(float(t).as_integer_ratio()[1].bit_length()-1 for t in np.asarray(a).ravel())


def integers(a,e):
    def convert(t):
        num,den=float(t).as_integer_ratio();return num << (e-den.bit_length()+1)
    return np.array([convert(t) for t in np.asarray(a).ravel()],object).reshape(np.shape(a))


def exact_values(x,v,reg,bank_int,bank_exponent):
    zl,zh,hl,hh=[a[0] for a in reference.gate.original.screen.boxes(v,reg[None])]
    if np.any(hl>hh):return None,None
    e=max(exponent(t) for t in [x,zl,zh,hl,hh,np.array([.12])])+1
    xx,zzl,zzh,hhl,hhh=[integers(t,e) for t in [x,zl,zh,hl,hh]]
    bound=integers(np.array(.12),e).item();scale=1 << e
    slopes=reference.gate.original.base.SLOPES[reg].astype(object)
    intercepts=reference.gate.original.base.INTERCEPTS[reg].astype(object)
    # int(), not float-valued objects: keep the computation entirely exact.
    slopes=np.vectorize(int,otypes=[object])(slopes)
    intercepts=np.vectorize(int,otypes=[object])(intercepts)
    a=bank_int;k,d,n=a.shape
    total=np.minimum(a[:,-1]*hhl[-1],a[:,-1]*hhh[-1]).sum(1)
    total-=(a*intercepts[None]).sum((1,2))*scale
    for j in range(d):
        pl=xx if j==0 else hhl[j-1];ph=xx if j==0 else hhh[j-1]
        low=max(-bound,max(zzl[j]-ph));high=min(bound,min(zzh[j]-pl))
        if low>high:return None,None
        assert (low+high)%2==0
        b=(low+high)//2;lo=np.maximum(pl,zzl[j]-b);hi=np.minimum(ph,zzh[j]-b)
        assert np.all(lo<=hi)
        sa=a[:,j]*slopes[j];c=(0 if j==0 else a[:,j-1])-sa
        total+=(np.minimum(c*lo,c*hi)-sa*b).sum(1)
    assert all(isinstance(t,int) for t in total)
    return total,1 << (e+bank_exponent)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/light_h2_credit/full_bank_ceiling';out=root/'results/bounded_error_primal_gate/component'
    protocol=json.loads((out/'protocol.json').read_text());hashes=dict(protocol['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    assert json.loads((out/'summary.json').read_text())['passed']
    audits=json.loads((out/'audits.json').read_text());checks=dict(banks=0,regions=0,exact_directions=0,finite_enclosures=0,
        skipped_regions=0,skipped_directions=0,empty_domains=0,fraction_crosschecks=0)
    details=[]
    for rec in audits:
        path=inp/rec['source_data_file'];assert sha(path)==rec['source_data_sha256']
        gp=out/rec['gate_file'];assert sha(gp)==rec['gate_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        with np.load(gp) as z:skip=z['skip'];upper=z['upper']
        ae=exponent(bank);ai=integers(bank,ae);start=dict(checks)
        assert skip.shape==(len(regs),) and upper.shape==(len(regs),len(bank))
        for r,reg in enumerate(regs):
            vals,den=exact_values(x,v,reg,ai,ae);checks['regions']+=1
            if vals is None:
                assert not skip[r] and np.all(np.isposinf(upper[r]));checks['empty_domains']+=1;continue
            checks['exact_directions']+=len(vals)
            for k,(value,bound) in enumerate(zip(vals,upper[r])):
                if np.isfinite(bound):
                    num,bd=float(bound).as_integer_ratio();assert value*bd<=num*den,(rec['seed'],rec['method'],r,k)
                    checks['finite_enclosures']+=1
                else:assert np.isposinf(bound)
                if skip[r]:assert value<=0;checks['skipped_directions']+=1
            checks['skipped_regions']+=int(skip[r])
            if r in [0,len(regs)//2,len(regs)-1]:
                for k in sorted({0,len(bank)//2,len(bank)-1}):
                    assert F(vals[k],den)==reference.exact_upper(x,v,reg,bank[k]);checks['fraction_crosschecks']+=1
        assert int(skip.sum())==rec['skipped']
        checks['banks']+=1;details.append(dict(seed=rec['seed'],method=rec['method'],checks={k:checks[k]-start[k] for k in checks}))
        print(json.dumps(dict(banks=checks['banks'],exact_directions=checks['exact_directions'],skipped_directions=checks['skipped_directions'])),flush=True)
    assert checks['skipped_regions']==2437 and checks['regions']==3998
    result=dict(passed=True,checks=checks,details=details,source_sha256=sha(Path(__file__)),parent_source_count=len(hashes),
        protocol_sha256=sha(out/'protocol.json'),audits_sha256=sha(out/'audits.json'),summary_sha256=sha(out/'summary.json'),
        scope='Every real region and direction; exact binary-rational midpoint upper and every authorized skip. Not a general floating error proof or online timing.')
    target=out/'integer_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,checks=checks)),flush=True)


if __name__=='__main__':main()
