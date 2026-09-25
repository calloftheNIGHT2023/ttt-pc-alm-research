"""Exact online certificate, independent LP and identical snapshot-bank audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import online_factorized_h2 as memory
from audit_light_h2_certificates_v2 import relaxed_lp


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/online_factorized_h2/development';out=root/'results/online_factorized_h2/certificate_audit'
    ceiling=root/'results/light_h2_credit/full_bank_ceiling';p=json.loads((inp/'protocol.json').read_text())
    assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    frozen={(r['seed'],r['method']):r for r in json.loads((ceiling/'records.json').read_text())}
    rows={(r['seed'],r['method']):r for r in json.loads((inp/'rows.json').read_text()) if r['repetition']==0 and r['n']==4}
    checks=dict(details=0,full_banks_bitwise=0,exact_certificates=0,independent_full_region_lp=0,
        independent_relaxed_lp=0,c20_ceiling_sets_equal=0,c5_certificate_supersets=0);maximum=0.;records=[]
    for seed in p['seeds']:
        for cfg in [c for c in p['configs'] if c.get('factorized_full')]:
            name=cfg['name'];path=inp/f'detail_{seed}_{name}.json';detail=json.loads(path.read_text())
            assert sha(path)==rows[seed,name]['detail_sha256'];checks['details']+=1
            with np.load(inp/rows[seed,name]['state_file']) as z:x=z['x'];v=z['v']
            bank=np.array(detail['credit_bank']);key=(seed,cfg['learner']+'_'+cfg['credit_mode']);old=frozen[key]
            assert sha(ceiling/old['data_file'])==old['data_sha256']
            with np.load(ceiling/old['data_file']) as z:assert np.array_equal(bank,z['bank']);checks['full_banks_bitwise']+=1
            actual={proof['pattern']:dict(pattern=proof['pattern'],direction=proof['direction'],exact=proof['exact']) for proof in detail['credit_proofs']}
            reference={proof['pattern']:proof for proof in old['proofs']}
            if cfg['contract_rounds']==20:assert actual==reference;checks['c20_ceiling_sets_equal']+=1
            else:
                assert reference.keys()<=actual.keys() and all(actual[k]==value for k,value in reference.items())
                checks['c5_certificate_supersets']+=1
            assert not actual.keys()&set(detail['positive_mode_keys'])
            for proof in detail['credit_proofs']:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,len(x));a=bank[proof['direction']]
                exact=memory.factor.original.exact_optimum(x,v,reg,a);assert exact==proof['exact'] and exact['positive'];checks['exact_certificates']+=1
                _,_,matrix,rhs=memory.neighbor.pattern_matrix(x,v,reg)
                lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4,options={'primal_feasibility_tolerance':1e-9})
                assert lp.status==2;checks['independent_full_region_lp']+=1
                lp,value=relaxed_lp(x,v,reg,a)
                if 'empty_layer' in exact:assert lp.status==2
                else:
                    assert lp.success;error=abs(exact['value']-value);assert error<1e-8;maximum=max(maximum,error)
                checks['independent_relaxed_lp']+=1
            baseline=rows[seed,cfg['learner']+f'_c{cfg["contract_rounds"]}']
            records.append(dict(seed=seed,method=name,proofs=len(detail['credit_proofs']),unique_proofs=len(actual),
                geometry_calls_avoided=baseline['geometry_calls']-rows[seed,name]['geometry_calls'],
                bank_rows=len(bank),factorized=detail['factorized'],detail_sha256=sha(path)))
        print(json.dumps(dict(seed=seed,**checks)),flush=True)
    result=dict(passed=True,checks=checks,maximum_relaxed_lp_error=maximum,source_sha256=sha(Path(__file__)),
        protocol_sha256=sha(inp/'protocol.json'),scope='Actual current-wave proof replay; no future candidate access at deployment')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('summary.json',result),('records.json',records)]: (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
