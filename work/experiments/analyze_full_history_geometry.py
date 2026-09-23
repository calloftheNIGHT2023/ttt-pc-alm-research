"""Exact geometry of every retained full-history separation; no selection."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from exact_credit_hull import RationalBank,unpack,pack
from audit_joint_credit_minimax import exact_mixture,rational_optimum
from run_credit_wall_budget import load_inputs

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/baseline_history';prior=root/'results/exact_credit_hull/development';inputs,hashes=load_inputs(root)
    assert read(base/'audit/summary.json')['passed'];cases=read(base/'analysis/strict_cases.json');rows=[]
    for case in cases:
        if case['full_status']!='nonpositive':continue
        seed=case['seed'];index=case['index'];x,v,regs,banks=inputs[seed];reg=regs[index];key=reg.tobytes().hex();assert key==case['pattern']
        old=next(r for r in read(prior/f'bank_{seed}_{case["comparator"]}.json') if r['pattern']==key)['result']
        residual=[unpack(t) for t in old['witness']['proof']['residual']]
        cap=next(r for r in read(base/'development/captures.json') if r['seed']==seed and r['method']==case['full_method'])
        assert sha(base/'development'/cap['arrays_file'])==cap['arrays_sha256']
        with np.load(base/'development'/cap['arrays_file']) as z:full=z['bank']
        vals,den=RationalBank(full).values(residual);assert max(vals)<=0
        ar=read(prior/f'bank_{seed}_alm_native.json')[index]['result'];w=np.array(ar['weights']);credit=exact_mixture(banks['alm_native'],w)
        lower=rational_optimum(x,v,reg,credit);products=[sum((credit[j][i]*residual[j*len(x)+i] for i in range(len(x))),F(0)) for j in range(4)]
        assert sum(products)>0 and lower>0
        src=root/hashes[seed]['banks']['alm_native']['file']
        with np.load(src) as z:steps=z['steps'];labels=z['labels']
        av,_=RationalBank(banks['alm_native']).values(residual);ids=np.flatnonzero(np.array([t>0 for t in av]));assert len(ids)
        rows.append(dict(seed=seed,index=index,pattern=key,full_method=case['full_method'],directions=len(full),negative=sum(t<0 for t in vals),zero=sum(t==0 for t in vals),
            upper=pack(F(max(vals),den)),residual=[pack(t) for t in residual],alm_exact_lower=pack(lower),alm_exact_lower_float=float(lower),
            alm_at_witness=pack(sum(products)),alm_layer_products=[pack(t) for t in products],alm_layer_products_float=[float(t) for t in products],
            first_stored_alm_direction_violating_witness=int(steps[ids].min()),positive_at_witness_kinds=sorted(set(str(t) for t in labels[ids])),
            returned_alm_certificate_steps=sorted(set(int(t) for t in steps[w>0])),
            warning='First stored witness violation is neither first generated useful credit nor earliest global positive certificate'))
    out=base/'analysis';target=out/'retained_geometry.json';assert not target.exists();target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summary=dict(passed=True,cases=len(rows),seeds=sorted({r['seed'] for r in rows}),source_sha256=sha(Path(__file__)),
        cases_sha256=sha(out/'strict_cases.json'),geometry_sha256=sha(target),min_alm_lower=min(r['alm_exact_lower_float'] for r in rows),max_alm_lower=max(r['alm_exact_lower_float'] for r in rows))
    (out/'geometry_audit.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
