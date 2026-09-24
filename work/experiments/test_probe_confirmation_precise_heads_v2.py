"""Rational analytic fixtures and corrupt-state rejection for precise audit."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from diagnose_probe_meta_ridge_arithmetic import high_precision
from probe_confirmation_precise_heads_v2 import replay
import probe_credit_confirmation_suite as suite
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def analytic(model,x,v,q):
    bank=[[F(float(t)) for t in row] for row in model.bank.numpy()]
    projection=[[F(float(t)) for t in row] for row in model.projection.numpy()]
    weights=[F(float(t)) for t in model.mean_weights.numpy()]
    def features(z):
        phi=[];mu=[]
        for point in z:
            raw=[]
            for biases in bank:
                h=F(float(point))
                for b in biases:
                    z=h+b
                    h=F(0) if z<=0 or z>=1 else 2*z if z<=F(1,2) else 2-2*z
                raw.append(h)
            mean=sum(raw)/len(raw);centered=[t-mean for t in raw]
            phi.append([sum(centered[i]*projection[i][j] for i in range(len(bank))) for j in range(2)])
            mu.append(mean+sum(t*w for t,w in zip(centered,weights)))
        return phi,mu
    # log_ridge=0 => exp=1, clamped to the exact stored binary value of .1.
    phi,mu=features(x);penalty=F(.1)
    matrix=[[sum(row[i]*row[j] for row in phi)+(penalty if i==j else 0) for j in range(2)] for i in range(2)]
    rhs=[sum(row[j]*(F(float(target))-m) for row,target,m in zip(phi,v,mu)) for j in range(2)]
    det=matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]
    beta=[(rhs[0]*matrix[1][1]-matrix[0][1]*rhs[1])/det,
        (matrix[0][0]*rhs[1]-rhs[0]*matrix[1][0])/det]
    qp,qmu=features(q)
    pred=[min(F(1),max(F(0),m+sum(a*b for a,b in zip(row,beta)))) for row,m in zip(qp,qmu)]
    return np.array([float(t) for t in pred]),np.array([float(t) for t in beta])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();base=root/'results/probe_credit_confirmation';out=base/'precise_head_selftests_v2'
    assert not out.exists();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    tensor=lambda a:torch.tensor(a,dtype=torch.float64)
    model=SimpleNamespace(bank=tensor([[0,0,0,0],[.0625,-.125,0,.0625],[-.125,0,.0625,0]]),
        projection=tensor([[1,.5],[-.5,.25],[.25,-1]]),mean_weights=tensor([.125,-.25,.5]),log_ridge=tensor([0]))
    q=np.array([0,.0625,.125,.25,.5,.75,1.]);tests={}
    for name,x,v in [
        ('distinct_support',np.array([.125,.25,.375,.625]),np.array([.1,.8,.2,.9])),
        ('repeated_support',np.full(4,.125),np.array([.2,.3,.4,.5])),
        ('near_repeated_support',np.array([.125,np.nextafter(.125,1),.5,.75]),np.array([.2,.201,.4,.9])),
    ]:
        expected,beta=analytic(model,x,v,q)
        p70,b70,_=high_precision(model,x,v,q,70);p110,b110,_=high_precision(model,x,v,q,110)
        np.testing.assert_allclose(p70,expected,rtol=0,atol=1e-60)
        np.testing.assert_allclose(p110,expected,rtol=0,atol=1e-60)
        np.testing.assert_allclose(b70,beta,rtol=0,atol=1e-60)
        np.testing.assert_allclose(b110,beta,rtol=0,atol=1e-60)
        tests[name]=dict(passed=True,independent_rational_primal_2x2_equals_high_precision_dual=True)
    inp=base/'predictions';diag=base/'head_discrepancy_diagnosis'
    loaded,_=suite.resources.legacy.oldfit.meta.load(root)
    record=next(r for r in json.loads((diag/'heads.json').read_text()) if r['method']=='cold__meta_ridge64')
    assert sha(inp/record['saved_file'])==record['saved_sha256']
    with np.load(inp/record['saved_file']) as z:a={k:z[k].copy() for k in z.files}
    cfg=record['configuration'];x,v,q=a['x_observed'],a['v_observed'],a['q_observed']
    commit=json.loads((inp/'tasks/5500953/commit.json').read_text())
    meta=next(r['metadata'] for r in json.loads((inp/commit['rows_file']).read_text()) if r['method']==cfg['name'])
    prediction,states,note=replay(cfg,x,v,q,a,meta,loaded)
    assert note['high_precision_used'] and note['original_tolerances_passed']
    tests['observed_reference_discrepancy_resolved']=dict(passed=True,diagnostic=note)
    bad={k:z.copy() for k,z in a.items()};bad['prediction'][30]+=.0001
    try:replay(cfg,x,v,q,bad,meta,loaded)
    except AssertionError:tests['corrupted_prediction_rejected']=dict(passed=True)
    else:raise AssertionError('Prediction corruption accepted')
    bad={k:z.copy() for k,z in a.items()};bad['fast_0'][0,0,0]+=.0001
    try:replay(cfg,x,v,q,bad,meta,loaded)
    except AssertionError:tests['corrupted_state_rejected']=dict(passed=True)
    else:raise AssertionError('State corruption accepted')
    out.mkdir();exclusive_json(out/'tests.json',tests)
    result=dict(passed=True,tests=len(tests),query_targets_accessed=False,audit_gate_passed=False,
        source_sha256={n:sha(Path(__file__).parent/n) for n in [Path(__file__).name,'probe_confirmation_precise_heads_v2.py','diagnose_probe_meta_ridge_arithmetic.py','probe_confirmation_independent_heads.py']},
        tests_sha256=sha(out/'tests.json'),original_head_atol=1e-9,original_head_rtol=1e-10)
    exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
