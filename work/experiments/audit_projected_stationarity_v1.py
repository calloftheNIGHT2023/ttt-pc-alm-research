"""301 independent forward-sensitivity rational audit, with an activation radius."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from diagnose_counterfactual_conditional_risk_v1 import read, save, sha


def audit(b,x,v):
    b=list(map(lambda a:F(float(a)),b)); bound=F(.12); epsilon=F(.001)
    gradient=[F(0)]*4; loss=F(0); maximum=F(0); radii=[]; knots=0
    for xx,vv in zip(x,v):
        value=F(float(xx)); derivative=[F(0)]*4
        for j,bj in enumerate(b):
            pre=value+bj
            pre_jac=[d+int(i==j) for i,d in enumerate(derivative)]
            distance=min(abs(pre-k) for k in [F(0),F(1,2),F(1)])
            norm=sum(abs(d) for d in pre_jac)
            knots+=int(distance==0)
            if norm: radii.append(distance/(2*norm))
            if pre<=0 or pre>=1: value=F(0); derivative=[F(0)]*4
            elif pre<F(1,2): value=2*pre; derivative=[2*d for d in pre_jac]
            else: value=2-2*pre; derivative=[-2*d for d in pre_jac]
        error=value-F(float(vv)); maximum=max(maximum,abs(error))
        residual=max(error-epsilon,F(0))+min(error+epsilon,F(0))
        loss+=residual**2/(2*len(x))
        for j,d in enumerate(derivative): gradient[j]+=d*residual/len(x)
    # Normal-cone condition; do not use the production projection function.
    conditions=[(-bound<=bj<=bound) and (gj==0 or (bj==-bound and gj>0) or (bj==bound and gj<0))
                for bj,gj in zip(b,gradient)]
    return gradient,loss,maximum,knots,all(conditions),min(radii) if radii else F(1)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve()
    inp=root/'results/projected_stationarity/development_v1'
    out=root/'results/projected_stationarity/audit_v1'; out.mkdir(parents=True,exist_ok=False)
    summary=read(inp/'summary.json'); assert summary['passed']
    for name,digest in summary['outputs_sha256'].items(): assert sha(inp/name)==digest,name
    raw=root/'results/certificate_activity_attribution/development'
    originals={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    rows=[]; inputs={}; cached={}; certified=Counter(); unique=set()
    for r in read(inp/'exact_states.json'):
        seed=r['seed']; source=raw/originals[seed]['file']
        if seed not in cached:
            assert sha(source)==originals[seed]['sha256']; inputs[str(source.relative_to(root))]=sha(source)
            with np.load(source,allow_pickle=False) as z:
                cached[seed]={n:z[n] for n in ['x_observed','v_observed','prefix_b','anchor_b']}
        data=cached[seed]; phase,step,origin=r['location']
        b=data[('prefix' if phase==0 else 'anchor')+'_b'][step-1,origin]
        g,loss,maximum,knots,kkt,radius=audit(b,data['x_observed'],data['v_observed'])
        assert list(map(str,g))==r['gradient_fractions']
        assert str(loss)==r['exact_loss'] and str(maximum)==r['exact_max_raw']
        assert knots==r['exact_knots'] and kkt==r['exact_kkt']
        if kkt and knots==0:
            assert radius>0
            certified[seed]+=1; unique.add((seed,b.tobytes()))
        rows.append(dict(seed=seed,location=r['location'],exact_kkt=kkt,activation_knots=knots,
            gradient_fractions=list(map(str,g)),activation_stability_radius=str(radius),
            activation_stability_radius_float=float(radius),smooth_box_local_minimum=kkt and knots==0))
    assert len(rows)==62 and sum(certified.values())==24
    save(out/'certificates.json',rows)
    save(out/'inputs.json',inputs)
    positive=[r for r in rows if r['smooth_box_local_minimum']]
    result=dict(passed=True,locations=len(rows),exact_gradient_components=4*len(rows),certified_positions=len(positive),
        certified_tasks=len(certified),certified_counts_by_seed=dict(certified),unique_certified_parameter_states=len(unique),
        minimum_positive_radius=min(r['activation_stability_radius_float'] for r in positive),
        exact_forward_sensitivity_matches_reverse=True,source_sha256=sha(Path(__file__)),
        input_summary_sha256=sha(inp/'summary.json'),query_targets_accessed=False,reference_accessed=False,
        outputs_sha256={n:sha(out/n) for n in ['certificates.json','inputs.json']})
    save(out/'summary.json',result)
    print(result,flush=True)


if __name__=='__main__': main()
