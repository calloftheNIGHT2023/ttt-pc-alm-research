"""All eight old cases: witness violation is separate from a global proof."""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np
from exact_credit_hull import RationalBank,unpack,pack
from audit_joint_credit_minimax import exact_mixture,rational_optimum
import joint_forward_realization as model

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/joint_forward_realization';inp=base/'development'
    assert read(base/'audit/summary.json')['passed'];old=read(root/'results/baseline_history/analysis/retained_geometry.json');generations=read(inp/'generations.json');rows=[]
    for case in old:
        seed=case['seed'];index=case['index'];key=case['pattern'];gen=next(r for r in generations if r['seed']==seed)
        assert sha(inp/gen['arrays_file'])==gen['arrays_sha256']
        with np.load(inp/gen['arrays_file']) as z:newbank=z['bank'];point_ids=z['point_ids'];labels=z['labels']
        with np.load(root/'results/baseline_history/development'/gen['old_bank_source']['arrays_file']) as z:oldbank=z['bank']
        with np.load(root/f'results/common_pool_credit/development/pool_{seed}.npz') as z:x=z['x'];v=z['v'];reg=z['regs'][index]
        assert reg.tobytes().hex()==key;residual=[unpack(t) for t in case['residual']];values,den=RationalBank(newbank).values(residual);positive=np.flatnonzero(np.array([t>0 for t in values]))
        info=dict(seed=seed,index=index,pattern=key,new_directions=len(newbank),new_negative=sum(t<0 for t in values),new_zero=sum(t==0 for t in values),new_positive=len(positive),
            new_max_at_old_witness=pack(F(max(values),den)),violating_points=sorted(set(int(t) for t in point_ids[positive])),violating_kinds=sorted(set(str(t) for t in labels[positive])),
            warning='Actual joint direction positive at the old witness is not by itself a global positive certificate',methods={})
        for method,bank in [('constructed_joint',newbank),('old_plus_constructed_joint',np.concatenate([oldbank,newbank]))]:
            row=read(inp/f'bank_{seed}_{method}.json')[index];assert row['pattern']==key;detail=dict(status=row['effective_status'],raw_status=row['result']['status'])
            if row['effective_status']=='positive':
                r=row['result'] if row['result']['status']=='positive' else row['inherited'];weights=np.array(r['weights']);a=exact_mixture(bank,weights);lower=rational_optimum(x,v,reg,a);assert lower>0
                detail.update(exact_lower=pack(lower),exact_lower_float=float(lower),support_directions=int(np.count_nonzero(weights)),
                    old_history_weight=float(weights[:len(oldbank)].sum()) if method.startswith('old_plus') else 0.,
                    new_probe_weight=float(weights[len(oldbank):].sum()) if method.startswith('old_plus') else float(weights.sum()))
            elif row['effective_status']=='nonpositive':detail['common_upper']=row['result']['witness']['proof']['max_credit']
            info['methods'][method]=detail
        rows.append(info)
    out=base/'analysis';out.mkdir(parents=True,exist_ok=True);path=out/'old_cases.json';assert not path.exists();path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    result=dict(passed=True,cases=len(rows),cases_with_actual_witness_violation=sum(r['new_positive']>0 for r in rows),
        statuses={m:{s:sum(r['methods'][m]['status']==s for r in rows) for s in ['positive','nonpositive','unknown']} for m in model.METHODS},
        source_sha256=sha(Path(__file__)),audit_sha256=sha(base/'audit/summary.json'),cases_sha256=sha(path))
    (out/'case_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
