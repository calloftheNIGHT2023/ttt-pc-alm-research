"""Extend the support-selected witness to64 using rational arithmetic only."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import sys
import time
import numpy as np
import multiplier_fixed_point_exact as core
from audit_multiplier_fixed_point import rational_trace,exact_forward
from run_multiplier_fixed_point_screen import sha,dump

def main():
    sys.set_int_max_str_digits(0)
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/multiplier_fixed_point';out=base/'witness';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    audit=json.loads((base/'audit/summary.json').read_text());assert audit['passed'];src=Path(__file__).parent
    assert sha(src/'audit_multiplier_fixed_point.py')==audit['source_sha256']
    for name,h in json.loads((base/'continuation/protocol.json').read_text())['source_sha256'].items():assert sha(src/name)==h,name
    selection=json.loads((base/'audit/witness_selection.json').read_text());assert len(selection)==1
    p=dict(source_sha256=sha(Path(__file__)),audit_sha256=sha(base/'audit/summary.json'),selection_sha256=sha(base/'audit/witness_selection.json'),
        selected=selection[0],steps=64,scope='same support-selected example, exact mathematical continuation; no new effectiveness selection')
    dump(out/'protocol.json',p);seed=selection[0]['seed'];restart=selection[0]['restart']
    certs=json.loads((base/'exact/certificates.json').read_text());c=next(t for t in certs if t['seed']==seed and t['restart']==restart)
    a=np.load(base/'screen'/f'{seed}.npz');x=[F(float(t)) for t in a['x']];v=[F(float(t)) for t in a['v']];state=core.decode(c['state']);start=time.perf_counter()
    records=rational_trace(x,v,state,64);dump(out/'trajectory.json',records)
    initial=json.dumps(records[0]['state'],sort_keys=True)
    firstprimal=next(r['step'] for r in records if json.dumps(r['state'],sort_keys=True)!=initial)
    firstbranch=next(r['step'] for r in records if r['changed_parameter_branch'])
    feasible=[r for r in records if core.unpack(r['support_error'])<=core.EPS]
    assert feasible and firstbranch==7
    pred,pattern,jac=exact_forward(x,state[0]);res=[(1 if p>=t else -1)*max(F(0),abs(p-t)-core.EPS) for p,t in zip(pred,v)]
    grad=[sum((r*row[j] for r,row in zip(res,jac)),F(0))/len(x) for j in range(4)]
    assert not any(grad)
    # At exact zero gradient, the specified zero-moment Adam and positive-
    # damping GN updates are exactly zero. This is NOT all BP algorithms.
    floats=np.load(base/'continuation'/f'{seed}_alm64.npz');idx=list(floats['restarts']).index(restart)
    difference=0.
    for r in records:
        b,h=core.decode(r['state']);step=r['step']
        difference=max(difference,float(np.max(abs(np.array([float(t) for t in b])-floats['b'][step,idx]))),
            float(np.max(abs(np.array([[float(t) for t in row] for row in h])-floats['h'][step,:,idx]))))
    result=dict(passed=True,seed=seed,restart=restart,first_primal_move=firstprimal,first_forward_branch_change=firstbranch,
        first_strict_support_feasible=feasible[0]['step'],minimum_support_error=float(min(core.unpack(r['support_error']) for r in records)),
        exact_zero_gradient=True,violated_observation_jacobians_all_zero=all(not any(j) for r,j in zip(res,jac) if r),
        no_claim_all_BP=True,max_float_vs_exact_primal_error=difference,seconds=time.perf_counter()-start,
        trajectory_sha256=sha(out/'trajectory.json'),protocol_sha256=sha(out/'protocol.json'))
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
