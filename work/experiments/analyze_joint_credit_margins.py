"""Same-pool certified margins and an ideal exact-oracle iteration bound.

No timing or actual iteration claim. Only regions solved by both LPs and
certified positive by both are paired; all excluded cases are counted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/joint_credit_minimax/development';p=json.loads((inp/'protocol.json').read_text());audit=root/'results/joint_credit_minimax/audit/summary.json'
    assert json.loads(audit.read_text())['passed'];banks={(r['seed'],r['method']):r for r in json.loads((inp/'banks.json').read_text())}
    rows=[];counts=dict(regions=0,native_already_certified=0,paired_positive=0,both_nonpositive=0,unpaired_positive=0)
    for seed in p['seeds']:
        aa=banks[seed,'alm_native'];bb=banks[seed,'alm_residual'];a=json.loads((inp/aa['rows_file']).read_text());b=json.loads((inp/bb['rows_file']).read_text())
        assert len(a)==len(b)
        for left,right in zip(a,b):
            counts['regions']+=1;assert left['pattern']==right['pattern']
            if left['old_positive']:counts['native_already_certified']+=1;continue
            assert not right['old_positive']
            lp=left['joint'];rp=right['joint'];assert lp['success'] and rp['success']
            if not lp['positive'] and not rp['positive']:counts['both_nonpositive']+=1;continue
            if not (lp['positive'] and rp['positive']):counts['unpaired_positive']+=1;continue
            gl=lp['proof']['exact_convex_lower']['value'];gr=rp['proof']['exact_convex_lower']['value'];assert gl>0 and gr>0
            # |a_k|_inf<=1 and |r|_inf<=1 imply universal M=d*n=16 for both.
            # 2*M^2*ln(K)/gamma^2 is a sufficient ideal exact-oracle threshold.
            tl=2*16**2*np.log(aa['directions'])/gl**2;tr=2*16**2*np.log(bb['directions'])/gr**2
            rows.append(dict(seed=seed,index=left['index'],pattern=left['pattern'],native_gamma=gl,residual_gamma=gr,
                margin_ratio=gl/gr,ideal_bound_native=float(tl),ideal_bound_residual=float(tr),ideal_bound_ratio=float(tl/tr)))
            counts['paired_positive']+=1
    ratios=np.array([r['ideal_bound_ratio'] for r in rows]);margins=np.array([r['margin_ratio'] for r in rows])
    result=dict(passed=True,counts=counts,paired_regions=len(rows),native_smaller_sufficient_bound=int(np.sum(ratios<1)),
        bound_ratio_quantiles=np.quantile(ratios,[0,.25,.5,.75,1]).tolist(),margin_ratio_quantiles=np.quantile(margins,[0,.25,.5,.75,1]).tolist(),
        native_bound_quantiles=np.quantile([r['ideal_bound_native'] for r in rows],[0,.5,1]).tolist(),
        source_sha256=sha(Path(__file__)),audit_sha256=sha(audit),protocol_sha256=sha(inp/'protocol.json'),
        scope='Paired positive-margin diagnostic and conservative sufficient bounds under exact arithmetic; NOT measured iteration savings or a comparison of optimal complexity')
    out=root/'results/joint_credit_minimax/analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'margin_summary.json').exists()
    (out/'margin_rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');result['rows_sha256']=sha(out/'margin_rows.json')
    (out/'margin_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
