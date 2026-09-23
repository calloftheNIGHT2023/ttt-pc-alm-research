"""Moment-step bounds, extreme distributions and inherited frozen oracle."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import brentq
from scipy.special import logsumexp
import credit_moment_step as model
import verify_finite_credit_game as guard


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def guarded_solve(x,v,regs,bank):
    with patch.object(guard,'model',model):return guard.guarded_solve(x,v,regs,bank)


def verify():
    rng=np.random.default_rng(5900001);checks=dict(moment_cases=0,no_overshoot=0,root_upper=0,partition_upper=0,potential_lower=0,guarded_solves=0,repeat_arrays=0)
    minslack=np.inf;maxover=-np.inf
    for k in [2,9,64]:
        for scale in [1.,1e-8]:
            for skew in [0.,20.,100.]:
                for fraction in [.25,.9,1-1e-6]:
                    g=rng.uniform(-1,1,k)*scale;g[0]=scale;g[-1]=-scale
                    logs=rng.normal(size=k);logs[-1]+=skew
                    p=model.probabilities(logs[None])[0];mu=float(p@g);delta=mu+fraction*(g.max()-mu)
                    new,meta=model.update(logs[None],g[None],delta);assert meta['status'][0]==0
                    q=model.probabilities(new)[0];span=np.ptp(g);gg=(g-g.min())/span
                    mean=float(p@gg);var=float(p@(gg-mean)**2);target=(delta-g.min())/span;c=target-mean;b=float(meta['beta'][0])
                    post=float(q@gg);assert mean-1e-12<=post<=target+1e-12
                    maxover=max(maxover,post-target);checks['no_overshoot']+=1
                    fun=lambda beta:float(model.probabilities((logs+beta*gg)[None])[0]@gg-target)
                    high=max(1.,b)
                    while fun(high)<0:high*=2
                    root=brentq(fun,0,high,xtol=1e-12);assert b<=root+1e-10;checks['root_upper']+=1
                    lz=float(logsumexp(logs+b*gg)-logsumexp(logs))
                    bound=mean*b+var*np.expm1(b)-var*b
                    assert lz<=bound+1e-10;checks['partition_upper']+=1
                    lower=(var+c)*b-c;decrease=b*gg.max()-lz
                    assert lower>=-1e-12 and decrease>=lower-1e-10
                    minslack=min(minslack,decrease-lower);checks['potential_lower']+=1;checks['moment_cases']+=1
    logs=np.array([[0.,0.],[0.,0.],[0.,-1e6],[0.,0.]])
    g=np.array([[-1.,-2.],[0.,0.],[-1.,1.],[np.nan,1.]])
    _,meta=model.update(logs,g);assert np.array_equal(meta['status'],[1,1,2,3])
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=model.original.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24)
    positive=0
    for n in [4,8]:
        x=xx[:n];v=vv[:n];regs=np.array([model.original.base.pattern(x,b) for b in np.r_[teacher[None],rng.uniform(-.12,.12,(7,4))]],np.uint8)
        bank=rng.normal(size=(9,4,n));bank/=abs(bank).max((1,2),keepdims=True)
        arrays,meta=guarded_solve(x,v,regs,bank);again,repeated=guarded_solve(x,v,regs,bank);checks['guarded_solves']+=2
        for name,value in arrays.items():assert value.tobytes()==again[name].tobytes();checks['repeat_arrays']+=1
        assert meta['proofs']==repeated['proofs'] and meta['zero_variance']==meta['nonfinite_failures']==meta['already_at_target']==0
        fixed=model.original.float_optimum(x,v,regs,arrays['credit']);assert np.max(abs(fixed-arrays['best_value']))<1e-10
        positive+=meta['positive']
    return dict(passed=True,checks=checks,degenerate_cases=4,max_post_minus_target=maxover,min_potential_lower_slack=minslack,positive=positive,
        scope='Old primitive and algebraic distributions only; inherited oracle independently audited in 222')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/credit_halfspace_projection/development/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=root/'results/round_225_audit.json';assert json.loads(audit.read_text())['passed'];result=verify()
    for name in ['audit_credit_halfspace_projection.py','audit_credit_halfspace_resources.py','plot_credit_halfspace_projection.py','audit_research_round_225.py',
        'credit_moment_step.py',Path(__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    result.update(source_sha256=hashes,parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit))
    out=root/'results/credit_moment_step/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
