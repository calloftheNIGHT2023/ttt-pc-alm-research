"""Explain all 16 previous strict cases; don't confuse witness violation with proof."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from exact_credit_hull import RationalBank,unpack,pack
from audit_joint_credit_minimax import exact_mixture,rational_optimum
from audit_baseline_history import RowDyadicDots,direct_witness
from run_baseline_history import OLD


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/baseline_history';inp=base/'development';prior=root/'results/exact_credit_hull/development'
    assert read(base/'audit/summary.json')['passed'];cases=read(base/'audit/old_strict_cases.json');caps={(r['seed'],r['method']):r for r in read(inp/'captures.json')};rows=[]
    for case in cases:
        seed=case['seed'];method=case['full_method'];key=case['pattern'];cap=caps[seed,method];path=inp/cap['arrays_file'];assert sha(path)==cap['arrays_sha256']
        with np.load(path) as z:bank=z['bank'];steps=z['steps'];labels=z['labels']
        with np.load(root/f'results/common_pool_credit/development/pool_{seed}.npz') as z:x=z['x'];v=z['v'];regs=z['regs']
        comparator=OLD[method]
        old=next(r for r in read(prior/f'bank_{seed}_{comparator}.json') if r['pattern']==key);new=next(r for r in read(inp/f'bank_{seed}_{method}.json') if r['pattern']==key)
        residual=[unpack(t) for t in old['result']['witness']['proof']['residual']];values,den=RationalBank(bank).values(residual);ids=np.flatnonzero(np.array([q>0 for q in values]))
        info=dict(**case,index=new['index'],full_directions=len(bank),old_witness_still_valid=len(ids)==0,
            directions_violating_old_witness=len(ids),old_witness_max_over_full=pack(F(max(values),den)),
            first_old_witness_violation_step=int(steps[ids].min()) if len(ids) else None,
            old_witness_violation_steps=sorted(set(int(t) for t in steps[ids])),
            warning='A positive dot at one old witness is not a global positive certificate')
        info['prior_comparator_status']=info['comparator'];info['comparator']=comparator
        reg=regs[new['index']];assert reg.tobytes().hex()==key
        if not len(ids):
            direct_witness(x,v,reg,RowDyadicDots(bank),old['result']['witness']);info['supplemental_old_witness_reverified']=True
        if new['effective_status']=='positive':
            r=new['result'] if new['result']['status']=='positive' else new['inherited'];weights=np.array(r['weights']);selected=np.flatnonzero(weights>0)
            a=exact_mixture(bank,weights);lower=rational_optimum(x,v,reg,a);assert lower>0
            info.update(full_exact_lower=pack(lower),full_exact_lower_float=float(lower),proof_direction_count=len(selected),
                proof_steps=sorted(set(int(t) for t in steps[selected])),proof_latest_step=int(steps[selected].max()),
                proof_weight_by_kind={str(label):float(weights[labels==label].sum()) for label in sorted(set(labels))})
        rows.append(info)
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'strict_cases.json').exists();(out/'strict_cases.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    summary=dict(passed=True,cases=len(rows),status_counts=dict(Counter(r['full_status'] for r in rows)),old_witnesses_still_valid=sum(r['old_witness_still_valid'] for r in rows),
        source_sha256=sha(Path(__file__)),audit_sha256=sha(base/'audit/summary.json'),cases_sha256=sha(out/'strict_cases.json'),scope='Complete post-hoc explanation of the predeclared 16 prior cases; proof latest step is not minimal possible step')
    (out/'case_analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
