"""KKT, scalar-root, KL potential, oracle and forbidden-call preflight."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.optimize import brentq
import credit_halfspace_projection as model
import verify_finite_credit_game as guard


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def guarded_solve(x,v,regs,bank):
    with patch.object(guard,'model',model):return guard.guarded_solve(x,v,regs,bank)


def verify():
    rng=np.random.default_rng(5900001);checks=dict(root_cases=0,kkt=0,potential=0,pinsker_progress=0,guarded_solves=0,repeat_arrays=0)
    maxroot=0.;maxmean=0.;minpotential=np.inf
    for k in [2,9,64]:
        for _ in range(16):
            g=rng.uniform(-1,1,k);g[0]=1;g[-1]=-1
            logs=rng.normal(0,2,k);w=model.probabilities(logs[None])[0]
            delta=float((w@g+g.max())/2)
            new,m=model.project(logs[None],g[None],delta);q=model.probabilities(new)[0]
            assert m['status'][0]==0 and abs(q.sum()-1)<1e-12 and np.all(q>0)
            span=g.max()-g.min();gg=(g-g.min())/span;target=(delta-g.min())/span
            fun=lambda beta: float(model.probabilities((logs+beta*gg)[None])[0]@gg-target)
            hi=max(1.,float(m['beta'][0])*2);ref=brentq(fun,0,hi,xtol=1e-13)
            maxroot=max(maxroot,abs(ref-m['beta'][0]));maxmean=max(maxmean,abs(q@g-delta))
            assert abs(ref-m['beta'][0])<1e-10 and abs(q@g-delta)<1e-11;checks['root_cases']+=1
            kkt=np.log(q/w)-m['beta'][0]*gg;assert np.ptp(kkt)<1e-10;checks['kkt']+=1
            targetw=.5*q;targetw[g.argmax()]+=.5
            kl=lambda a,b:float(np.sum(a*np.log(a/b)))
            decrease=kl(targetw,w)-kl(targetw,q);move=kl(q,w)
            assert decrease>=move-1e-10;minpotential=min(minpotential,decrease-move);checks['potential']+=1
            lower=2*(delta-w@g)**2/span**2;assert move>=lower-1e-10;checks['pinsker_progress']+=1
    logs=np.zeros((3,4));g=np.array([[-1,-2,-3,-4],[0,0,0,0],[np.nan,0,0,0]])
    _,meta=model.project(logs,g);assert np.array_equal(meta['status'],[1,1,3])
    # Original seed construction is separate from algebraic root-test RNG consumption.
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=model.original.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24)
    positives=0
    for n in [4,8]:
        x=xx[:n];v=vv[:n]
        regs=np.array([model.original.base.pattern(x,b) for b in np.r_[teacher[None],rng.uniform(-.12,.12,(7,4))]],np.uint8)
        bank=rng.normal(size=(9,4,n));bank/=abs(bank).max((1,2),keepdims=True)
        arrays,meta=guarded_solve(x,v,regs,bank);again,repeated=guarded_solve(x,v,regs,bank);checks['guarded_solves']+=2
        for name,value in arrays.items():assert value.tobytes()==again[name].tobytes();checks['repeat_arrays']+=1
        assert meta['proofs']==repeated['proofs'] and meta['bracket_failures']==meta['nonfinite_failures']==0
        values=model.original.float_optimum(x,v,regs,arrays['credit'])
        assert np.max(abs(values-arrays['best_value']))<1e-10;positives+=meta['positive']
    return dict(passed=True,checks=checks,max_beta_error=maxroot,max_mean_error=maxmean,min_pythagorean_slack=minpotential,
        positive=positives,limited_and_nonfinite_cases=3,scope='Old primitive; inherited response already independently LP audited in 222')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/finite_credit_game/development/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=root/'results/round_223_audit.json';assert json.loads(audit.read_text())['passed']
    result=verify()
    for name in ['audit_finite_credit_game.py','audit_finite_credit_resources.py','plot_finite_credit_game.py','audit_research_round_223.py',
                 'credit_halfspace_projection.py',Path(__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    result.update(source_sha256=hashes,parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit))
    out=root/'results/credit_halfspace_projection/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
