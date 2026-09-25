"""Independent exact/LP audit of every deployed interval or fallback proof."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import online_interval_h2 as memory
from audit_light_h2_certificates_v2 import relaxed_lp


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/online_interval_h2/development';old=root/'results/online_factorized_h2/development'
    out=root/'results/online_interval_h2/certificate_audit';p=json.loads((inp/'protocol.json').read_text())
    assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    lookup={(r['seed'],r['method']):r for r in json.loads((inp/'rows.json').read_text()) if r['repetition']==0 and r['n']==4}
    oldrows={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text()) if r['repetition']==0 and r['n']==4}
    checks=dict(details=0,full_banks_bitwise=0,proof_sequence_equal_to_rational=0,exact_certificates=0,
        independent_full_region_lp=0,independent_relaxed_lp=0,interval_enclosures=0,interval_direct=0,rational_fallback=0)
    maximum=0.;records=[]
    for seed in p['seeds']:
        for cfg in [c for c in p['configs'] if c.get('factorized_full')]:
            name=cfg['name'];refname=cfg.get('rational_comparator',name);row=lookup[seed,name];refrow=oldrows[seed,refname]
            assert sha(inp/row['detail_file'])==row['detail_sha256'] and sha(old/refrow['detail_file'])==refrow['detail_sha256']
            detail=json.loads((inp/row['detail_file']).read_text());reference=json.loads((old/refrow['detail_file']).read_text());checks['details']+=1
            bank=np.array(detail['credit_bank']);assert np.array_equal(bank,np.array(reference['credit_bank']));checks['full_banks_bitwise']+=1
            key=lambda pp:(pp['pattern'],pp['direction'],pp['wave'])
            assert list(map(key,detail['credit_proofs']))==list(map(key,reference['credit_proofs']));checks['proof_sequence_equal_to_rational']+=1
            with np.load(inp/row['state_file']) as z:x=z['x'];v=z['v']
            for proof,ref in zip(detail['credit_proofs'],reference['credit_proofs']):
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,len(x));a=bank[proof['direction']]
                exact=memory.interval.original.exact_optimum(x,v,reg,a);assert exact==ref['exact'] and exact['positive'];checks['exact_certificates']+=1
                if proof.get('kind')=='interval':
                    interval=memory.interval.enclose(x,v,reg[None],a[None])
                    assert interval['positive'][0] and interval['lower'][0]==proof['lower'] and interval['upper'][0]==proof['upper']
                    assert bool(interval['empty_shared_bias'][0])==proof['empty_shared_bias']
                    if proof['empty_shared_bias']:assert 'empty_layer' in exact
                    else:
                        value=F(int(exact['numerator']),int(exact['denominator']))
                        assert F(proof['lower'])<=value<=F(proof['upper']) and proof['lower']>0
                    checks['interval_enclosures']+=1;checks['interval_direct']+=1
                else:
                    assert proof['exact']==exact
                    if proof.get('kind')=='rational_fallback':checks['rational_fallback']+=1
                _,_,matrix,rhs=memory.neighbor.pattern_matrix(x,v,reg)
                lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4,options={'primal_feasibility_tolerance':1e-9})
                assert lp.status==2;checks['independent_full_region_lp']+=1
                lp,value=relaxed_lp(x,v,reg,a)
                if 'empty_layer' in exact:assert lp.status==2
                else:
                    assert lp.success;error=abs(exact['value']-value);assert error<1e-8;maximum=max(maximum,error)
                checks['independent_relaxed_lp']+=1
            if cfg.get('interval_verifier'):
                direct=sum(c['credit']['interval_direct'] for c in detail['screen_calls'])
                fractions=sum(c['credit']['exact_checks'] for c in detail['screen_calls'])
                assert direct==sum(pp.get('kind')=='interval' for pp in detail['credit_proofs'])
                assert fractions==sum(c['credit']['rational_checks'] for c in detail['screen_calls'])
            else:direct=0;fractions=sum(c['credit']['exact_checks'] for c in detail['screen_calls'])
            records.append(dict(seed=seed,method=name,proofs=len(detail['credit_proofs']),interval_direct=direct,fraction_calls=fractions,
                credit_seconds=sum(c['credit']['seconds'] for c in detail['screen_calls']),
                float_seconds=sum(c['credit']['float_seconds'] for c in detail['screen_calls']),
                interval_seconds=sum(c['credit'].get('interval_seconds',0.) for c in detail['screen_calls']),
                rational_seconds=sum(c['credit']['exact_seconds'] for c in detail['screen_calls']),
                detail_sha256=sha(inp/row['detail_file'])))
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    result=dict(passed=True,checks=checks,maximum_relaxed_lp_error=maximum,source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(inp/'protocol.json'),scope='Every deployed proof rechecked with Fraction and independent LP outside algorithm timing')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary.json',result),('records.json',records)]: (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
