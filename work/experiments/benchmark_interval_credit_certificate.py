"""Exact inclusion audit and isolated certificate-component timings.

All 96 frozen full-bank pools; every selected direction is checked, including
nonpositive cases. Timed verification uses the unchanged rough positive gate.
"""
import argparse
import hashlib
import json
import time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import interval_credit_certificate as interval
import factorized_credit_bank as factor


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check_interval(enclosure,i,exact):
    if enclosure['empty_shared_bias'][i]:assert 'empty_layer' in exact
    elif 'empty_layer' not in exact:
        value=F(int(exact['numerator']),int(exact['denominator']))
        lo=enclosure['lower'][i];hi=enclosure['upper'][i]
        assert (not np.isfinite(lo) or F(float(lo))<=value) and (not np.isfinite(hi) or value<=F(float(hi)))
    if enclosure['positive'][i]:assert exact['positive']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/light_h2_credit/full_bank_ceiling';out=root/'results/interval_credit_certificate/development'
    parent=root/'results/online_factorized_h2/development/protocol.json'
    p=json.loads(parent.read_text());hashes=dict(p['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(interval.__file__)]:hashes[path.name]=sha(path)
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(parent),verification=interval.verify(),
        variants=['rational','interval_fallback'],repetitions=3,order_seed=482309,
        scope='Selected-direction certificate component on 96 frozen development pools; not whole screening or online timing')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=json.loads((inp/'records.json').read_text());rows=[];audits=[];order=np.random.default_rng(protocol['order_seed'])
    for record in records:
        path=inp/record['data_file'];assert sha(path)==record['data_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        values=factor.Bank(x,v,bank).values(regs);selected=values.argmax(1);a=bank[selected]
        rough=values[np.arange(len(regs)),selected];gate=rough>1e-10*(1+abs(a).sum((1,2)))
        enclosure=interval.enclose(x,v,regs,a);exact=[interval.original.exact_optimum(x,v,r,aa) for r,aa in zip(regs,a)]
        for i,e in enumerate(exact):check_interval(enclosure,i,e)
        eligible=regs[gate];aa=a[gate];expected=np.array([e['positive'] for i,e in enumerate(exact) if gate[i]],bool)
        keys={r.tobytes().hex() for r,yes in zip(eligible,expected) if yes}
        assert keys=={e['pattern'] for e in record['proofs']}
        proposals=len(eligible);jobs=[(variant,rep) for variant in protocol['variants'] for rep in range(3)]
        direct=int(np.sum(enclosure['positive']&gate));fallback=proposals-direct
        for index in order.permutation(len(jobs)):
            variant,rep=jobs[index];start=time.perf_counter()
            if variant=='rational':mask=np.array([interval.original.exact_optimum(x,v,r,a0)['positive'] for r,a0 in zip(eligible,aa)],bool)
            elif proposals:
                result=interval.enclose(x,v,eligible,aa);mask=result['positive'].copy()
                for i in np.flatnonzero(~mask):mask[i]=interval.original.exact_optimum(x,v,eligible[i],aa[i])['positive']
            else:mask=np.zeros(0,bool)
            elapsed=time.perf_counter()-start;assert np.array_equal(mask,expected)
            rows.append(dict(seed=record['seed'],method=record['method'],variant=variant,repetition=rep,
                seconds=elapsed,proposals=proposals,positive=int(mask.sum()),interval_direct=direct,fallback=fallback))
        audits.append(dict(seed=record['seed'],method=record['method'],data_sha256=record['data_sha256'],
            selected_regions_exact_checked=len(regs),proposals=proposals,positive=len(keys),interval_direct=direct,fallback=fallback))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(banks=len(audits),total=96,method=record['method'],seed=record['seed'],proposals=proposals,
            interval_direct=direct,fallback=fallback)),flush=True)
    summaries=[]
    for name in dict.fromkeys(r['method'] for r in records):
        means={variant:float(np.mean([r['seconds'] for r in rows if r['method']==name and r['variant']==variant])) for variant in protocol['variants']}
        summaries.append(dict(method=name,mean_seconds=means,rational_over_interval=means['rational']/means['interval_fallback']))
    result=dict(passed=True,banks=len(audits),rows=len(rows),selected_regions_exact_checked=sum(r['selected_regions_exact_checked'] for r in audits),
        proposals=sum(r['proposals'] for r in audits),positive=sum(r['positive'] for r in audits),
        interval_direct=sum(r['interval_direct'] for r in audits),fallback=sum(r['fallback'] for r in audits),summaries=summaries,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),scope=protocol['scope'])
    for name,value in [('summary.json',result),('audits.json',audits)]: (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
