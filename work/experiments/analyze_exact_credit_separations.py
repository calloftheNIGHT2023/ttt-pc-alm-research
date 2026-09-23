"""Inspect every strict separation against the three strong single libraries."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from run_credit_wall_budget import load_inputs
from exact_credit_hull import RationalBank,unpack,pack
from audit_joint_credit_minimax import exact_mixture,rational_optimum


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/exact_credit_hull';inp=base/'development'
    assert read(base/'audit/summary.json')['passed'];inputs,hashes=load_inputs(root);paired=read(inp/'paired.json');rows=[]
    for block in paired:
        method=block['comparator']
        if method not in ['adam60_native','pc_native','nodual_native']:continue
        for case in block['strict_separations']:
            assert case['alm']=='positive' and case['comparator']=='nonpositive';seed=case['seed'];key=case['pattern'];x,v,regs,banks=inputs[seed]
            native=next(r for r in read(inp/f'bank_{seed}_alm_native.json') if r['pattern']==key);other=next(r for r in read(inp/f'bank_{seed}_{method}.json') if r['pattern']==key)
            reg=regs[native['index']];assert reg.tobytes().hex()==key;ar=native['result'];br=other['result'];residual=[unpack(t) for t in br['witness']['proof']['residual']]
            integers,den=RationalBank(banks[method]).values(residual);assert max(integers)<=0
            weights=np.array(ar['weights']);credit=exact_mixture(banks['alm_native'],weights)
            value=rational_optimum(x,v,reg,credit);products=[sum((credit[j][i]*residual[j*len(x)+i] for i in range(len(x))),F(0)) for j in range(4)]
            at_witness=sum(products,F(0));assert at_witness>=value>0
            src=root/hashes[seed]['banks']['alm_native']['file']
            with np.load(src) as z:labels=z['labels'];steps=z['steps']
            mass={str(label):float(weights[labels==label].sum()) for label in sorted(set(labels))}
            rows.append(dict(seed=seed,index=native['index'],pattern=key,comparator=method,
                alm_exact_lower=pack(value),alm_exact_lower_float=float(value),comparator_upper=br['witness']['proof']['max_credit'],
                negative_directions=sum(t<0 for t in integers),zero_directions=sum(t==0 for t in integers),directions=len(integers),
                witness_nonzero_residuals=sum(t!=0 for t in residual),residual_float=np.array([float(t) for t in residual]).reshape(4,-1).tolist(),
                alm_at_witness=pack(at_witness),alm_at_witness_float=float(at_witness),alm_layer_products=[pack(t) for t in products],
                alm_layer_products_float=[float(t) for t in products],weight_mass_by_credit_kind=mass,
                same_alm_residual_status=next(r for r in read(inp/f'bank_{seed}_alm_residual.json') if r['pattern']==key)['result']['status']))
    assert len(rows)==16
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'strict_cases.json').exists();(out/'strict_cases.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summary=dict(passed=True,strict_method_cases=len(rows),unique_task_regions=len({(r['seed'],r['pattern']) for r in rows}),
        seeds=sorted({r['seed'] for r in rows}),min_alm_lower=min(r['alm_exact_lower_float'] for r in rows),max_alm_lower=max(r['alm_exact_lower_float'] for r in rows),
        all_baseline_upper_exact_zero=all(r['comparator_upper']==['0','1'] for r in rows),all_same_alm_residual_positive=all(r['same_alm_residual_status']=='positive' for r in rows),
        source_sha256=sha(Path(__file__)),audit_sha256=sha(base/'audit/summary.json'),cases_sha256=sha(out/'strict_cases.json'),scope='Post-hoc complete strict-case explanation, not confirmatory prevalence or multiplier necessity')
    (out/'case_analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
