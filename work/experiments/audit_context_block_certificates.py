"""Recheck every retained cold-block credit proof in exact arithmetic and LP."""
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
    inp=root/'results/context_block_scaling/development';p=json.loads((inp/'protocol.json').read_text())
    assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());checks=dict(details=0,exact_certificates=0,interval_enclosures=0,full_lp=0,relaxed_lp=0)
    maximum=0.;records=[]
    for row in rows:
        if 'detail_file' not in row:continue
        path=inp/row['detail_file'];assert sha(path)==row['detail_sha256'];detail=json.loads(path.read_text());checks['details']+=1
        proofs=detail.get('credit_proofs',[])
        if not proofs:continue
        bank=np.asarray(detail['credit_bank'])
        with np.load(inp/row['state_file']) as z:x=z['x'];v=z['v']
        for proof in proofs:
            reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,len(x));direction=bank[proof['direction']]
            exact=memory.interval.original.exact_optimum(x,v,reg,direction);assert exact['positive'];checks['exact_certificates']+=1
            if proof.get('kind')=='interval':
                enclosure=memory.interval.enclose(x,v,reg[None],direction[None])
                assert enclosure['positive'][0] and enclosure['lower'][0]==proof['lower'] and enclosure['upper'][0]==proof['upper']
                if proof['empty_shared_bias']:assert 'empty_layer' in exact
                else:assert F(proof['lower'])<=F(int(exact['numerator']),int(exact['denominator']))<=F(proof['upper']) and proof['lower']>0
                checks['interval_enclosures']+=1
            else:assert proof['exact']==exact
            _,_,matrix,rhs=memory.neighbor.pattern_matrix(x,v,reg)
            lp=linprog(np.zeros(4),A_ub=matrix,b_ub=rhs,bounds=[(-.12,.12)]*4,
                options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9})
            assert lp.status==2;checks['full_lp']+=1
            lp,value=relaxed_lp(x,v,reg,direction)
            if 'empty_layer' in exact:assert lp.status==2
            else:
                assert lp.success;error=abs(exact['value']-value);assert error<1e-8;maximum=max(maximum,error)
            checks['relaxed_lp']+=1
        records.append(dict(seed=row['seed'],method=row['method'],n=row['n'],proofs=len(proofs),detail_sha256=row['detail_sha256']))
        print(json.dumps(dict(seed=row['seed'],n=row['n'],method=row['method'],**checks)),flush=True)
    result=dict(passed=True,checks=checks,maximum_relaxed_lp_error=maximum,records=records,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='All successful first-repeat retained proofs; failed original attempts expose costs but not discarded proof objects')
    out=inp.parent/'certificate_audit';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=checks)),flush=True)


if __name__=='__main__':main()
